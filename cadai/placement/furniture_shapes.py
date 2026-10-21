"""Furniture shapes across the archive, blocks and loose lines alike (ADR-025).

The dataset build (dataset.pair_rows) finds every piece of furniture in the raw plans: block
references (furniture.instances) and groups of touching loose lines the cleanup deletes
(furniture.loose_instances), each with a shape picture (shapes.picture). Exact signatures split one
bed into many keys (other segment counts, other duvet lines), so the signatures are grouped into
shape classes: in order of how often they are seen, a signature joins the first class whose leader
looks the same (shapes.distance <= CLASS_DIST), or leads a new one.

The engineer names classes on the "Mobilya şekilleri" page (most seen first, the first CATALOG_N);
a name given to a class's leader names every member, in every project. resolved_types() is that
mapping, used by the dataset build; Matcher names furniture of a new drawing by its nearest named
leader.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np

from cadai.cad.render import render
from cadai.i18n import say
from cadai.cad.scene import Scene
from cadai.corpus import process
from cadai.placement import furniture, shapes

SHAPE_DIR = furniture.FURNITURE_DIR / "shapes"         # per pair: <key>.npz; per class: <id>.png
CLASSES = furniture.FURNITURE_DIR / "shape_classes.json"
CATALOG = furniture.FURNITURE_DIR / "shape_catalog.json"
PROPOSALS = furniture.FURNITURE_DIR / "shape_proposals.json"
CLASS_DIST = 0.25            # as for exploded symbols (shapes.MATCH_DIST); checked on the archive in ADR-025
CATALOG_N = 300
MIN_PROJECTS = 2             # a class on the page is seen in at least this many projects
EXAMPLES = 3                 # drawings kept per signature for its thumbnail
NOT_FURNITURE = {"none", "annotation", "door", "window", "stair"}


def save_pair(key: str, project: str, raw_sha: str, furn: list[dict]) -> None:
    """Shapes of one pair's furniture (from dataset.pair_rows) -> SHAPE_DIR/<key>.npz."""
    SHAPE_DIR.mkdir(parents=True, exist_ok=True)
    furn = [f for f in furn if "shape" in f]
    np.savez_compressed(
        SHAPE_DIR / f"{key}.npz", project=project, raw=raw_sha,
        sig=np.array([f["sig"] for f in furn], dtype=str),
        block=np.array([f.get("block", "") for f in furn], dtype=str),
        grid=np.array([f["shape"]["grid"] for f in furn], dtype=np.float32).reshape(-1, shapes.GRID * shapes.GRID),
        ratio=np.array([f["shape"]["ratio"] for f in furn]), size=np.array([f["shape"]["size"] for f in furn]),
        xy=np.array([[f["x"], f["y"]] for f in furn]).reshape(-1, 2),
        rect=np.array([[f["short"], f["long"], f["angle"]] for f in furn]).reshape(-1, 3),
        page=np.array([f.get("page", [False] * 3) for f in furn], dtype=bool).reshape(-1, 3))


def file_id(sig: str) -> str:
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


def group(sigs: list[str], grids: np.ndarray, ratios: np.ndarray, sizes: np.ndarray) -> list[int]:
    """Class index of every signature (given most seen first): leader clustering by shape."""
    lg, lr, ls, lead_of = [], [], [], []
    out = []
    for g, r, s in zip(grids, ratios, sizes):
        k = -1
        if lg:
            S = np.array(ls)
            near = np.flatnonzero(np.abs(np.log(S / s)) <= 2 * CLASS_DIST)       # size alone would exceed the limit
            if len(near):
                d = shapes.distance(g.reshape(shapes.GRID, shapes.GRID), r, s, 0, np.array(lg)[near],
                                    np.array(lr)[near], S[near], np.zeros(len(near)))
                j = int(np.argmin(d))
                if d[j] <= CLASS_DIST:
                    k = lead_of[near[j]]
        if k < 0:
            k = len(lg)
            lg.append(g), lr.append(r), ls.append(s), lead_of.append(k)
        out.append(k)
    return out


def cluster_job(job) -> dict:
    """Shape classes of all furniture the last dataset build found -> CLASSES, CATALOG, thumbnails."""
    files = sorted(SHAPE_DIR.glob("*.npz"))
    if not files:
        raise RuntimeError(str(say("Mobilya şekli yok; önce yerleştirme veri setini kurun.",
                                   "No furniture shapes; build the placement dataset first.")))
    job.progress(0, len(files) + 2)
    count, projects, names = Counter(), defaultdict(set), defaultdict(Counter)
    pic: dict[str, tuple] = {}
    off_page: dict[str, bool] = {}           # thin, a text frame or a door (furniture._Marks)
    example: dict[str, tuple] = {}
    for i, f in enumerate(files, 1):
        with np.load(f) as z:
            project, raw = str(z["project"]), str(z["raw"])
            for k, sig in enumerate(z["sig"]):
                sig = str(sig)
                count[sig] += 1
                projects[sig].add(project)
                if z["block"][k]:
                    names[sig][str(z["block"][k])] += 1
                if sig not in pic:
                    pic[sig] = (z["grid"][k], float(z["ratio"][k]), float(z["size"][k]))
                    off_page[sig] = bool(z["page"][k].any()) if "page" in z else False
                ex = example.setdefault(sig, [])
                if len(ex) < EXAMPLES and all(e[0] != raw for e in ex):          # other drawings, for the thumbnail
                    ex.append((raw, *map(float, z["xy"][k]), *map(float, z["rect"][k])))
        job.progress(i)
    sigs = sorted(count, key=lambda s: (-len(projects[s]), -count[s]))
    cls = group(sigs, np.array([pic[s][0] for s in sigs]), np.array([pic[s][1] for s in sigs]),
                np.array([pic[s][2] for s in sigs]))
    members: dict[int, list[str]] = defaultdict(list)
    for s, k in zip(sigs, cls):
        members[k].append(s)
    classes = []
    for k, ms in members.items():
        lead = ms[0]
        classes.append({"sig": lead, "id": file_id(lead), "members": ms, "instances": sum(count[m] for m in ms),
                        "projects": len(set().union(*(projects[m] for m in ms))),
                        "short_m": round(example[lead][0][3], 2), "long_m": round(example[lead][0][4], 2),
                        "names": sum((names[m] for m in ms), Counter()).most_common(3), "example": example[lead]})
    classes.sort(key=lambda c: (-c["projects"], -c["instances"]))
    CLASSES.write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    job.progress(len(files) + 1)
    shown = [c for c in classes if c["projects"] >= MIN_PROJECTS and not off_page[c["sig"]]][:CATALOG_N]
    del pic
    _thumbnails(shown)
    CATALOG.write_text(json.dumps([{k: v for k, v in c.items() if k not in ("members", "example")} | {
        "member_count": len(c["members"]), "thumb": True} for c in shown], ensure_ascii=False, indent=1), encoding="utf-8")
    job.progress(len(files) + 2)
    loose = sum(c["instances"] for c in classes if c["sig"].startswith("~"))
    n = sum(count.values())
    job.log(say(f"{len(sigs)} imza → {len(classes)} şekil sınıfı ({n} örnek, {loose} tanesi dağınık çizgi); sayfada {len(shown)} sınıf",
                f"{len(sigs)} signatures → {len(classes)} shape classes ({n} examples, {loose} loose lines); {len(shown)} classes on the page"))
    return {"signatures": len(sigs), "classes": len(classes), "shown": len(shown)}


def _thumbnails(classes: list[dict], load=None) -> None:
    """240 px picture of each class's leader in its plan, the piece outlined in red. A drawing that
    cannot be loaded (one raw scene's texts need 9 GB as a fixed-width array) is skipped for the
    leader's next example."""
    from PIL import Image, ImageDraw
    load = load or (lambda raw: Scene.load(process.scene_path(raw)))
    todo = {c["id"]: list(c["example"]) for c in classes}
    failed: set[str] = set()
    while todo:
        by_raw = defaultdict(list)
        for cid, exs in list(todo.items()):
            exs[:] = [e for e in exs if e[0] not in failed]
            if not exs:
                del todo[cid]
                continue
            by_raw[exs[0][0]].append((cid, exs[0]))
        for raw, items in by_raw.items():
            try:
                scene = load(raw)
            except MemoryError:
                failed.add(raw)
                continue
            for cid, (_, x, y, short, long_, angle) in items:
                r = max(long_ * 0.9, 1.0)
                target = SHAPE_DIR / f"{cid}.png"
                render(scene, target, np.full(scene.n, 1, dtype=np.int8), bbox=(x - r, y - r, x + r, y + r), width=240,
                       colours={1: (150, 150, 160)})
                img = Image.open(target).convert("RGB")
                sc = 240 / (2 * r)
                a = np.radians(angle)
                u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
                corners = [np.array([x, y]) + su * long_ / 2 * u + sv * short / 2 * v
                           for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                px = [((q[0] - (x - r)) * sc, ((y + r) - q[1]) * sc) for q in corners]
                ImageDraw.Draw(img).polygon(px, outline=(220, 0, 0), width=2)
                img.save(target)
                del todo[cid]
            del scene


# --- the lexicon page (cadai/web/app.py LEXICONS) ------------------------------------------------
LEXICON_DIR = SHAPE_DIR
FURNITURE_TYPES = furniture.FURNITURE_TYPES


def load_types() -> dict[str, str]:
    return furniture.load_types()


def set_type(sig: str, furniture_type: str, source: str = "engineer") -> None:
    furniture.set_type(sig, furniture_type, source)


def load_proposals() -> dict[str, dict]:
    return json.loads(PROPOSALS.read_text(encoding="utf-8")) if PROPOSALS.exists() else {}


# --- names for the placement -------------------------------------------------------------------------
def _classes() -> list[dict]:
    return json.loads(CLASSES.read_text(encoding="utf-8")) if CLASSES.exists() else []


def class_types(classes: list[dict], confirmed: dict[str, str]) -> list[str | None]:
    """Type of each class: the leader's confirmed type, else the most common confirmed member type."""
    out = []
    for c in classes:
        if c["sig"] in confirmed:
            out.append(confirmed[c["sig"]])
            continue
        votes = Counter(confirmed[m] for m in c["members"] if m in confirmed)
        out.append(votes.most_common(1)[0][0] if votes else None)
    return out


def resolved_types() -> dict[str, str]:
    """{signature: type} for every signature of a named class, plus the confirmed ones themselves."""
    confirmed = furniture.load_types()
    classes = _classes()
    out = {}
    for c, t in zip(classes, class_types(classes, confirmed)):
        if t:
            out.update({m: t for m in c["members"]})
    return {**out, **confirmed}


class Matcher:
    """Names furniture of a new drawing: its signature's class, else the nearest named leader."""

    def __init__(self):
        classes = _classes()
        types = class_types(classes, furniture.load_types())
        self.known = resolved_types()
        named = [(c, t) for c, t in zip(classes, types) if t]
        self.types = [t for _, t in named]
        self.leaders = []
        if named:
            pics = {}
            for f in SHAPE_DIR.glob("*.npz"):
                with np.load(f) as z:
                    want = {c["sig"] for c, _ in named} - set(pics)
                    for k, s in enumerate(z["sig"]):
                        if str(s) in want and str(s) not in pics:
                            pics[str(s)] = (z["grid"][k], float(z["ratio"][k]), float(z["size"][k]))
                if len(pics) == len(named):
                    break
            keep = [i for i, (c, _) in enumerate(named) if c["sig"] in pics]
            self.types = [self.types[i] for i in keep]
            self.leaders = [pics[named[i][0]["sig"]] for i in keep]

    def types_for(self, furn: list[dict]) -> dict[str, str]:
        """{signature: type} for these instances (they need "shape" when their signature is new)."""
        out = {}
        for f in furn:
            if f["sig"] in self.known:
                out[f["sig"]] = self.known[f["sig"]]
            elif self.leaders and "shape" in f:
                sh = f["shape"]
                d = shapes.distance(sh["grid"].reshape(shapes.GRID, shapes.GRID), sh["ratio"], sh["size"], 0,
                                    np.array([l[0] for l in self.leaders]), np.array([l[1] for l in self.leaders]),
                                    np.array([l[2] for l in self.leaders]), np.zeros(len(self.leaders)))
                j = int(np.argmin(d))
                if d[j] <= CLASS_DIST:
                    out[f["sig"]] = self.types[j]
        return out
