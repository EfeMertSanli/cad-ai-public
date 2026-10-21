"""Furniture catalogue from the architects' raw drawings (plan-placement §2, docs/findings.md).

Sockets follow furniture (bed head, counter, TV wall), but block names in raw drawings are
meaningless ("GVXVVS", "A$C2AC03DC6"), so furniture is recognised by geometry:

- every top-level block reference inside a floor plan (alignment label not out-of-view) of
  0.3-4.5 m becomes an instance;
- its signature ignores position, rotation and mirroring: the two sides of its minimum-area
  rectangle (5 cm), the number of segments and the total line length (10 cm). The same bed drawn
  by two architects under two names collapses to one entry;
- per signature: projects, instances, size, the share of its lines the engineer deleted (furniture
  is deleted, doors and windows stay), the block names seen, and a thumbnail.
Types are assigned once (an LLM proposes from the thumbnails, the engineer confirms), stored in
DATA/furniture/types.json ({signature: type}).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from cadai import config
from cadai.i18n import say
from cadai.cad.render import render
from cadai.cad.scene import Scene
from cadai.cleanup.align import DELETE, KEEP, OUT_OF_VIEW
from cadai.corpus import process

FURNITURE_DIR = config.DATA / "furniture"
CATALOG = FURNITURE_DIR / "catalog.json"
TYPES = FURNITURE_DIR / "types.json"
PROPOSALS = FURNITURE_DIR / "proposals.json"

FURNITURE_TYPES = {
    "bed_double": say("Çift kişilik yatak", "Double bed"), "bed_single": say("Tek kişilik yatak", "Single bed"),
    "wardrobe": say("Gardırop / dolap", "Wardrobe / cupboard"), "sofa": say("Kanepe / koltuk", "Sofa / armchair"),
    "table": say("Masa (yemek)", "Table (dining)"), "desk": say("Çalışma masası", "Desk"),
    "tv_unit": say("TV ünitesi", "TV unit"), "kitchen_counter": say("Mutfak tezgahı", "Kitchen counter"),
    "sink": say("Evye", "Kitchen sink"), "stove": say("Ocak / fırın", "Hob / oven"), "fridge": say("Buzdolabı", "Fridge"),
    "dishwasher": say("Bulaşık makinesi", "Dishwasher"), "washing_machine": say("Çamaşır makinesi", "Washing machine"),
    "wc": say("Klozet", "WC pan"), "washbasin": say("Lavabo", "Washbasin"), "shower": say("Duş", "Shower"),
    "bathtub": say("Küvet", "Bathtub"), "boiler": say("Kombi", "Boiler"),
    "door": say("Kapı (mimari)", "Door (architectural)"), "window": say("Pencere (mimari)", "Window (architectural)"),
    "stair": say("Merdiven (mimari)", "Stair (architectural)"), "other_furniture": say("Diğer mobilya", "Other furniture"),
    "annotation": say("Yazı / işaret / ölçü", "Text / mark / dimension"),
    "none": say("Bilinmiyor / önemsiz", "Unknown / irrelevant"),
}
MIN_M, MAX_M = 0.3, 4.5


def _signature(pts: np.ndarray, n_segs: int, total_len: float) -> tuple[str, float, float, float]:
    """Rotation and mirror free key: minimum-area rectangle sides, segment count, line length.
    Returns (key, short side, long side, angle of the long side in degrees)."""
    centred = pts - pts.mean(axis=0)
    best = None
    for deg in range(0, 90, 2):                                # 2-degree search is enough at 5 cm
        a = np.radians(deg)
        rot = centred @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        w, h = np.ptp(rot[:, 0]), np.ptp(rot[:, 1])
        if best is None or w * h < best[0]:
            best = (w * h, w, h, deg)
    _, w, h, deg = best
    short, long_ = sorted((w, h))
    angle = deg if w >= h else deg + 90
    key = f"{round(short / 0.05)}x{round(long_ / 0.05)}-n{min(n_segs, 999)}-l{round(total_len / 0.1)}"
    return key, float(short), float(long_), float(angle % 180)


def shape_of(segs: np.ndarray) -> dict:
    """The shape picture of some segments (shapes.picture on points every 3 cm), for grouping
    furniture drawn alike in different files (ADR-025)."""
    from cadai.cleanup.align import _sample
    from cadai.placement import shapes
    pts, _ = _sample(segs[np.isfinite(segs).all(axis=1)], SAMPLE_M)
    grid, ratio, size = shapes.picture(pts)
    return {"grid": grid.astype(np.float32).ravel(), "ratio": float(ratio), "size": float(size)}


SAMPLE_M = 0.03


def instances(scene: Scene, labels: np.ndarray, with_shape: bool = False) -> list[dict]:
    """Block references inside floor plans, with signature, centre, size and engineer's verdict.
    `with_shape` adds the shape picture ("shape", see shape_of)."""
    out = []
    has_block = (scene.block_top >= 0) & (labels != OUT_OF_VIEW)
    if not has_block.any():
        return out
    order = np.argsort(scene.top, kind="stable")
    sorted_top = scene.top[order]
    seg_order = np.argsort(scene.seg_item, kind="stable")
    seg_sorted = scene.seg_item[seg_order]
    seg_len = np.hypot(scene.segs[:, 2] - scene.segs[:, 0], scene.segs[:, 3] - scene.segs[:, 1])
    marks = _Marks(scene) if with_shape else None
    for t in np.unique(scene.top[has_block]):
        lo, hi = np.searchsorted(sorted_top, [t, t + 1])
        items = order[lo:hi]
        a, b = np.searchsorted(seg_sorted, items), np.searchsorted(seg_sorted, items + 1)
        segs_idx = np.concatenate([seg_order[x:y] for x, y in zip(a, b)]) if len(items) else np.zeros(0, int)
        if len(segs_idx) < 2:
            continue
        segs = scene.segs[segs_idx]
        pts = np.vstack([segs[:, :2], segs[:, 2:]])
        ext = np.ptp(pts, axis=0)
        if not (MIN_M <= ext.max() <= MAX_M):
            continue
        total = float(seg_len[segs_idx].sum())
        key, short, long_, angle = _signature(pts, len(segs_idx), total)
        lab = labels[items]
        decided = np.isin(lab, (KEEP, DELETE))
        out.append({
            "sig": key, "top": int(t), "block": scene.meta["blocks"][scene.block_top[items[0]]],
            "x": float(pts[:, 0].mean()), "y": float(pts[:, 1].mean()), "short": short, "long": long_, "angle": angle,
            "deleted": float((lab[decided] == DELETE).mean()) if decided.any() else None,
            **({"shape": shape_of(segs), "page": marks.of(items, pts, short)} if with_shape else {}),
        })
    return out


PAGE_MIN_SHORT_M = 0.25        # a WC or washbasin is ~0.35 deep; thinner pieces are sills, frames, hatching


class _Marks:
    """What keeps a piece off the Mobilya şekilleri page (ADR-025): it is thin, a frame with a text
    inside (room name boxes) or a door (leaf with its swing). The piece still counts as furniture for
    the placement: leaving these out of the data lost wet-socket F1 0.450 -> 0.399."""

    def __init__(self, scene: Scene):
        kinds = np.array(scene.meta["types"])[scene.etype]
        sweep = np.abs(np.degrees(scene.sweep))
        self.swing = ((kinds == "ARC") & (scene.radius >= SWING_RADIUS_M[0]) & (scene.radius <= SWING_RADIUS_M[1])
                      & (sweep >= SWING_SWEEP_DEG[0]) & (sweep <= SWING_SWEEP_DEG[1]))
        bb = scene.bboxes()
        texts = np.flatnonzero((scene.text != "") & np.isfinite(bb[:, 0]))
        self.text_xy = np.column_stack([(bb[texts, 0] + bb[texts, 2]) / 2, (bb[texts, 1] + bb[texts, 3]) / 2])

    def of(self, items, pts: np.ndarray, short: float) -> list[bool]:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        framed = bool(len(self.text_xy) and ((self.text_xy > lo) & (self.text_xy < hi)).all(axis=1).any())
        return [short < PAGE_MIN_SHORT_M, framed, bool(self.swing[np.asarray(items)].any())]


# --- furniture drawn as loose lines (ADR-025) --------------------------------------------------
# Most architects explode their furniture: a bed is a few polylines on layer "0" or "MIMARI". Inside
# a floor plan, what our cleanup deletes and is not text, hatch or dimension is grouped like an
# exploded symbol: same-layer parts that touch. Door leaves with their swing are left out.
LOOSE_TOUCH_M = 0.02
LOOSE_MIN_SHORT_M = 0.15
LOOSE_SKIP = ("HATCH", "SOLID", "DIMENSION", "TEXT", "MTEXT", "ATTRIB", "POINT", "INSERT")
SWING_RADIUS_M = (0.5, 1.3)            # a door leaf's arc: this radius, a quarter turn
SWING_SWEEP_DEG = (75.0, 105.0)


def loose_instances(scene: Scene, where: np.ndarray, with_shape: bool = False) -> list[dict]:
    """Groups of touching loose lines inside `where` (a floor plan minus what the cleanup kept)
    of furniture size; same fields as instances(), "sig" starts with "~", "top" is -1."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    from cadai.cleanup.align import _sample
    kinds = np.array(scene.meta["types"])[scene.etype]
    length = scene.lengths()
    cand = np.flatnonzero(where & (scene.block_top < 0) & (scene.text == "") & ~np.isin(kinds, LOOSE_SKIP)
                          & np.isfinite(length) & (length > 0) & (length <= 2 * MAX_M))
    if not len(cand):
        return []
    seg_mask = np.isin(scene.seg_item, cand) & np.isfinite(scene.segs).all(axis=1)
    pts, seg_of = _sample(scene.segs[seg_mask], SAMPLE_M)
    owner = np.searchsorted(cand, scene.seg_item[seg_mask][seg_of])
    pairs = cKDTree(pts).query_pairs(LOOSE_TOUCH_M, output_type="ndarray")
    a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
    keep = (a != b) & (scene.layer[cand[a]] == scene.layer[cand[b]])
    graph = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(len(cand), len(cand)))
    _, comp = connected_components(graph, directed=False)
    order = np.argsort(comp[owner], kind="stable")
    starts = np.searchsorted(comp[owner][order], np.arange(comp.max() + 2))
    swing = ((kinds == "ARC") & (scene.radius >= SWING_RADIUS_M[0]) & (scene.radius <= SWING_RADIUS_M[1])
             & (np.abs(np.degrees(scene.sweep)) >= SWING_SWEEP_DEG[0]) & (np.abs(np.degrees(scene.sweep)) <= SWING_SWEEP_DEG[1]))
    marks = _Marks(scene) if with_shape else None
    seg_order = np.argsort(scene.seg_item, kind="stable")
    seg_lo = np.searchsorted(scene.seg_item[seg_order], cand)
    seg_hi = np.searchsorted(scene.seg_item[seg_order], cand + 1)
    out = []
    for c in range(comp.max() + 1):
        p_idx = order[starts[c]:starts[c + 1]]
        if len(p_idx) < 4:
            continue
        g_pts = pts[p_idx]
        ext = np.ptp(g_pts, axis=0)
        if not (MIN_M <= ext.max() <= MAX_M):
            continue
        local = np.flatnonzero(comp == c)
        members = cand[local]
        if swing[members].any():
            continue
        seg_idx = np.concatenate([seg_order[seg_lo[k]:seg_hi[k]] for k in local])
        key, short, long_, angle = _signature(g_pts, len(seg_idx), float(length[members].sum()))
        if short < LOOSE_MIN_SHORT_M:
            continue
        inst = {"sig": "~" + key, "top": -1, "block": "", "items": members.tolist(),
                "x": float((g_pts[:, 0].max() + g_pts[:, 0].min()) / 2), "y": float((g_pts[:, 1].max() + g_pts[:, 1].min()) / 2),
                "short": short, "long": long_, "angle": angle, "deleted": None}
        if with_shape:
            inst["shape"] = shape_of(scene.segs[seg_idx])
            inst["page"] = marks.of(members, g_pts, short)
        out.append(inst)
    return out


def harvest_job(job) -> dict:
    from cadai.cleanup.train import labelled_pairs
    FURNITURE_DIR.mkdir(parents=True, exist_ok=True)
    pairs = labelled_pairs()
    job.progress(0, len(pairs) + 1)
    stats: dict[str, dict] = defaultdict(lambda: {"projects": set(), "instances": 0, "names": Counter(),
                                                  "deleted": [], "short": [], "long": []})
    example: dict[str, tuple[str, dict]] = {}
    per_pair = {}
    for i, pair in enumerate(pairs, 1):
        scene = Scene.load(process.scene_path(pair["raw"]))
        with np.load(process.label_path(pair["project"], pair["final"])) as z:
            labels = z["labels"]
        insts = instances(scene, labels)
        per_pair[f"{pair['project']}__{process.key(pair['final'])}"] = insts
        for inst in insts:
            s = stats[inst["sig"]]
            s["projects"].add(pair["project"])
            s["instances"] += 1
            s["names"][inst["block"]] += 1
            s["short"].append(inst["short"])
            s["long"].append(inst["long"])
            if inst["deleted"] is not None:
                s["deleted"].append(inst["deleted"])
            example.setdefault(inst["sig"], (pair["raw"], inst))
        job.progress(i)
        if job.cancelled:
            return {}
    keys = sorted(stats, key=lambda k: (-len(stats[k]["projects"]), -stats[k]["instances"]))
    common = [k for k in keys if len(stats[k]["projects"]) >= 2 or stats[k]["instances"] >= 5]
    by_scene: dict[str, list[str]] = defaultdict(list)
    for k in common:
        by_scene[example[k][0]].append(k)
    for sha, ks in by_scene.items():
        scene = Scene.load(process.scene_path(sha))
        for k in ks:
            inst = example[k][1]
            mark = np.where(scene.top == inst["top"], 1, 2).astype(np.int8)
            r = max(inst["long"] * 0.9, 1.0)
            render(scene, FURNITURE_DIR / f"{_file_id(k)}.png", mark,
                   bbox=(inst["x"] - r, inst["y"] - r, inst["x"] + r, inst["y"] + r), width=240,
                   colours={2: (170, 170, 185), 1: (210, 0, 0)}, thick={1: 2})
    catalog = [{
        "sig": k, "id": _file_id(k), "projects": len(stats[k]["projects"]), "instances": stats[k]["instances"],
        "short_m": round(float(np.median(stats[k]["short"])), 2), "long_m": round(float(np.median(stats[k]["long"])), 2),
        "deleted_share": round(float(np.mean(stats[k]["deleted"])), 2) if stats[k]["deleted"] else None,
        "names": stats[k]["names"].most_common(4), "thumb": k in by_scene.get(example[k][0], []),
    } for k in keys]
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    (FURNITURE_DIR / "instances.json").write_text(json.dumps(per_pair), encoding="utf-8")
    job.progress(len(pairs) + 1)
    job.log(say(f"{len(pairs)} ham çizim, {len(keys)} farklı şekil, {len(common)} tanesi tekrar ediyor",
                f"{len(pairs)} raw drawings, {len(keys)} distinct shapes, {len(common)} repeat"))
    return {"pairs": len(pairs), "signatures": len(keys), "common": len(common)}


def _file_id(sig: str) -> str:
    import hashlib
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


def load_types() -> dict[str, str]:
    return json.loads(TYPES.read_text(encoding="utf-8")) if TYPES.exists() else {}


def load_proposals() -> dict[str, dict]:
    return json.loads(PROPOSALS.read_text(encoding="utf-8")) if PROPOSALS.exists() else {}


def set_type(sig: str, furniture_type: str, source: str = "engineer") -> None:
    if furniture_type not in FURNITURE_TYPES:
        raise ValueError(furniture_type)
    FURNITURE_DIR.mkdir(parents=True, exist_ok=True)
    types = load_types()
    types[sig] = furniture_type
    TYPES.write_text(json.dumps(types, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(FURNITURE_DIR / "types_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"sig": sig, "type": furniture_type, "source": source}, ensure_ascii=False) + "\n")


# --- the cleanup's furniture tool (ADR-021) ---------------------------------------------------
# Until the engineer has named the catalogue, "furniture" is what the archive shows: block shapes
# that engineers deleted in training projects (at least DELETE_MIN_SEEN times, on average at least
# DELETE_MIN_SHARE of their lines). Only training projects count, so the test projects stay unseen.
DELETE_SIGNATURES = FURNITURE_DIR / "delete_signatures.json"
DELETE_MIN_SEEN = 2
DELETE_MIN_SHARE = 0.8


def delete_signatures() -> set[str]:
    """Signatures the cleanup deletes: engineer-confirmed furniture types (not doors, windows,
    stairs), plus the shapes engineers deleted in the archive's training projects."""
    if DELETE_SIGNATURES.exists():
        sigs = set(json.loads(DELETE_SIGNATURES.read_text(encoding="utf-8")))
    else:
        from cadai.cleanup.train import split_of
        source = FURNITURE_DIR / "instances.json"                # written by harvest_job
        per_pair = json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}
        seen: dict[str, list[float]] = defaultdict(list)
        for key, insts in per_pair.items():
            if split_of(key.split("__")[0]) != "train":
                continue
            for inst in insts:
                if inst["deleted"] is not None:
                    seen[inst["sig"]].append(inst["deleted"])
        sigs = {s for s, v in seen.items() if len(v) >= DELETE_MIN_SEEN and np.mean(v) >= DELETE_MIN_SHARE}
        if per_pair:
            DELETE_SIGNATURES.write_text(json.dumps(sorted(sigs)), encoding="utf-8")
    architectural = {"door", "window", "stair"}
    for sig, kind in load_types().items():
        if kind in architectural:
            sigs.discard(sig)
        elif kind != "none":
            sigs.add(sig)
    return sigs


def detect(scene: Scene, where: np.ndarray, sigs: set[str] | None = None) -> np.ndarray:
    """Items of block references inside `where` whose shape is a furniture signature."""
    sigs = delete_signatures() if sigs is None else sigs
    labels = np.where(where, KEEP, OUT_OF_VIEW).astype(np.int8)
    out = np.zeros(scene.n, dtype=bool)
    tops = [inst["top"] for inst in instances(scene, labels) if inst["sig"] in sigs]
    if tops:
        out[np.isin(scene.top, tops)] = True
    return out & where
