"""Electrical symbol catalogue from the engineers' finals (plan-placement §2, docs/findings.md).

The office draws devices with a fixed block library (the same anonymous blocks appear in almost
every final), so one mapping block -> device type labels most devices of the whole archive.

harvest: every block reference in every final becomes an instance (block name, layer, centre,
size). Per block name: files, instances, layers, median size, and a thumbnail of one instance in
its surroundings. Types are assigned once, by the engineer on the website or by an LLM, and
stored in DATA/symbols/types.json ({block name: type}).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

from cadai import config
from cadai.i18n import say
from cadai.cad.render import render
from cadai.cad.scene import Scene
from cadai.corpus import inventory, process

SYMBOL_DIR = config.DATA / "symbols"
CATALOG = SYMBOL_DIR / "catalog.json"
TYPES = SYMBOL_DIR / "types.json"            # confirmed: {block: type}
PROPOSALS = SYMBOL_DIR / "proposals.json"    # suggested by an LLM, shown pre-selected, not used until confirmed

# Device types the placement model learns. "other" = electrical but not placed per room
# (panels, risers); "none" = not a device (title block parts, arrows, architecture leftovers).
DEVICE_TYPES = {
    "socket": say("Priz", "Socket"),
    "socket_wet": say("Etanj priz", "Wet-area socket"),
    "socket_power": say("Kuvvet / özel priz (fırın, klima)", "Power socket (oven, A/C)"),
    "switch": say("Anahtar", "Switch"),
    "light_ceiling": say("Tavan armatürü", "Ceiling light"),
    "light_wall": say("Aplik", "Wall light"),
    "light_emergency": say("Acil aydınlatma", "Emergency light"),
    "sensor": say("Sensör / sensörlü armatür", "Sensor / sensor light"),
    "data": say("Data / telefon", "Data / phone"),
    "tv": say("TV / anten", "TV / antenna"),
    "intercom": say("Diafon / zil", "Intercom / bell"),
    "fire": say("Yangın / duman", "Fire / smoke"),
    "panel": say("Pano / tablo", "Panel / board"),
    "junction": say("Buat / kutu", "Junction box"),
    "other": say("Diğer elektrik", "Other electrical"),
    "none": say("Cihaz değil", "Not a device"),
    "unsure": say("Emin değilim", "Not sure"),
}
UNCONFIRMED = {"unsure"}             # chosen by the engineer but not a type: never used for training

# Words in the text next to a symbol -> proposed type. Specific phrases first (first match wins).
HINT_WORDS = [
    ("ETANJ PRIZ", "socket_wet"), ("TRIFAZE", "socket_power"), ("OCAK", "socket_power"), ("FIRIN", "socket_power"),
    ("KLIMA", "socket_power"), ("CAMASIR", "socket_power"), ("CAM.M", "socket_power"), ("BULASIK", "socket_power"),
    ("BLS.M", "socket_power"), ("KOMBI", "socket_power"), ("PRIZ", "socket"),
    ("ANAHTAR", "switch"), ("KOMUTATOR", "switch"), ("VAVIYEN", "switch"), ("OTOMATIGI BUTONU", "switch"),
    ("APLIK", "light_wall"), ("ACIL", "light_emergency"), ("SENSOR", "sensor"), ("AVIZE", "light_ceiling"),
    ("ARMATUR", "light_ceiling"), ("FLORESAN", "light_ceiling"), ("SPOT", "light_ceiling"),
    ("YANGIN", "fire"), ("DUMAN", "fire"), ("ALARM", "fire"),
    ("ZAP", "tv"), ("RG6", "tv"), ("ANTEN", "tv"), ("TV", "tv"), ("CAT6", "data"), ("DATA", "data"),
    ("TELEFON", "data"), ("DIAFON", "intercom"), ("ZIL", "intercom"), ("BUAT", "junction"),
    ("TABLO", "panel"), ("PANO", "panel"),
    ("ILETKEN", "none"), ("SIGORTA", "none"), ("SALTER", "none"), ("AMPERMETRE", "none"), ("VOLTMETRE", "none"),
    ("KAZIK", "none"), ("TOPRAKLAMA", "other"),
]
LAMP_LABEL = re.compile(r"^[A-Z]{1,2}\s*\d?\s*X?\s*\d{1,3}\s*W$")       # "J 60 W", "P 2X40 W"


def hint_type(text: str) -> str | None:
    t = " ".join(text.split())
    if LAMP_LABEL.match(t):
        return "light_ceiling"
    for word, typ in HINT_WORDS:
        if word in t:
            return typ
    return None


def propose_from_hints(catalog: list[dict]) -> dict[str, dict]:
    """Proposals for catalogue rows from the texts written next to their instances (lamp labels in
    plans, descriptions in legend tables). Confidence = share of the row's hints that agree."""
    out = {}
    for c in catalog:
        votes: Counter = Counter()
        for text, n in c.get("hints", []):
            typ = hint_type(text)
            if typ:
                votes[typ] += n
        if not votes:
            continue
        typ, n = votes.most_common(1)[0]
        share = n / sum(c_n for _, c_n in c.get("hints", []))
        conf = say("yüksek", "high") if share >= 0.6 else say("orta", "medium") if share >= 0.3 else say("düşük", "low")
        texts = ", ".join(f'"{t}"' for t, _ in c["hints"][:3])
        out[c["block"]] = {"type": typ, "confidence": conf.both(),
                           "why": say(f"yanındaki yazılar: {texts}", f"nearby text: {texts}").both(),
                           "source": "text-hint"}
    return out


def propose_job(job) -> dict:
    """Write hint proposals for every catalogue row that has no proposal yet (hand-written ones stay)."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    current = load_proposals()
    new = {k: v for k, v in propose_from_hints(catalog).items() if k not in current}
    PROPOSALS.write_text(json.dumps({**new, **current}, ensure_ascii=False, indent=1), encoding="utf-8")
    job.log(say(f"{len(new)} yeni öneri; toplam {len(new) + len(current)}", f"{len(new)} new proposals; {len(new) + len(current)} in all"))
    return {"new": len(new), "total": len(new) + len(current)}
MAX_SYMBOL_M = 3.0               # larger block references are architecture, frames or title blocks


def instances(scene: Scene) -> list[dict]:
    """Every symbol of a final: block references (by name) and exploded symbols (by shape)."""
    return block_instances(scene) + exploded_instances(scene)


def block_instances(scene: Scene) -> list[dict]:
    """One row per top-level block reference: name, layer, centre and size in metres."""
    out = []
    has_block = scene.block_top >= 0
    if not has_block.any():
        return out
    bb = scene.bboxes()
    tops = np.unique(scene.top[has_block])
    order = np.argsort(scene.top, kind="stable")
    sorted_top = scene.top[order]
    for t in tops:
        lo, hi = np.searchsorted(sorted_top, [t, t + 1])
        items = order[lo:hi]
        b = bb[items]
        if not np.isfinite(b).any():
            continue
        x0, y0, x1, y1 = np.nanmin(b[:, 0]), np.nanmin(b[:, 1]), np.nanmax(b[:, 2]), np.nanmax(b[:, 3])
        size = max(x1 - x0, y1 - y0)
        if size > MAX_SYMBOL_M:
            continue
        name = scene.meta["blocks"][scene.block_top[items[0]]]
        out.append({"block": name, "layer": scene.meta["layers"][scene.layer[items[0]]], "top": int(t),
                    "items": items.tolist(), "x": float((x0 + x1) / 2), "y": float((y0 + y1) / 2),
                    "size": float(size)})
    return out


ELECTRIC_LAYER_WORDS = ["LINYE", "SEMBOL", "ZAYIF", "AYD", "PRIZ", "ANAHTAR", "ELEKTRIK", "ARMATUR", "AKIM",
                        "TOPRAK", "ZIL", "ACIL", "TELEFON", "DATA", "ANTEN", "YANGIN", "KUVVETLI"]
TOUCH_M = 0.03                    # parts closer than this belong to one exploded symbol
PART_MAX_M = 1.0                  # longer loose lines are wires, not symbol parts
EXPLODED_MIN_M, EXPLODED_MAX_M = 0.05, 1.0


def exploded_instances(scene: Scene, with_points: bool = False, max_m: float | None = None) -> list[dict]:
    """Exploded symbols: loose short parts on electrical layers that touch each other (ADR-009).
    The key is "~" + a rotation and mirror free signature (furniture._signature), plus how many
    circles and hatches the symbol has, so the same switch drawn in any direction is one entry.
    `with_points` adds the symbol's points every 2 cm ("pts"), for shape matching (shapes.py);
    `max_m` lets bigger groups through (a symbol with wire stubs attached, shapes.peel)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from cadai.corpus.inventory import fold
    from cadai.placement.furniture import _signature

    electric = np.array([any(w in fold(n) for w in ELECTRIC_LAYER_WORDS) for n in scene.meta["layers"]])
    length = scene.lengths()
    cand = np.flatnonzero((scene.block_top < 0) & electric[scene.layer] & (scene.text == "")
                          & (length > 0) & (length <= PART_MAX_M))
    if not len(cand):
        return []
    pos = {int(i): k for k, i in enumerate(cand)}
    seg_mask = np.isin(scene.seg_item, cand)
    segs, owner = scene.segs[seg_mask], scene.seg_item[seg_mask]
    d = segs[:, 2:] - segs[:, :2]
    n = np.clip(np.ceil(np.hypot(d[:, 0], d[:, 1]) / 0.02).astype(int), 1, 100) + 1
    rep = np.repeat(np.arange(len(segs)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
    pts = segs[rep, :2] + d[rep] * t[:, None]
    pt_item = np.array([pos[int(i)] for i in owner[rep]])
    pairs = cKDTree(pts).query_pairs(TOUCH_M, output_type="ndarray")
    a, b = pt_item[pairs[:, 0]], pt_item[pairs[:, 1]]
    keep = (a != b) & (scene.layer[cand[a]] == scene.layer[cand[b]])     # one symbol, one layer (ADR-024)
    graph = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(len(cand), len(cand)))
    _, comp = connected_components(graph, directed=False)
    types = scene.meta["types"]
    out = []
    for c in np.unique(comp):
        members = cand[comp == c]
        m_pts = pts[np.isin(pt_item, np.flatnonzero(comp == c))]
        ext = np.ptp(m_pts, axis=0)
        if not (EXPLODED_MIN_M <= ext.max() <= (max_m or EXPLODED_MAX_M)):
            continue
        kinds = [types[scene.etype[i]] for i in members]
        if len(members) < 2 and kinds[0] not in ("CIRCLE", "HATCH"):
            continue                                            # a lone short line is a wire end, not a symbol
        key, short, long_, _ = _signature(m_pts, int(np.isin(scene.seg_item, members).sum()), float(length[members].sum()))
        key = f"~{key}-c{kinds.count('CIRCLE')}-h{kinds.count('HATCH')}"
        out.append({"block": key, "layer": scene.meta["layers"][scene.layer[members[0]]], "top": -1,
                    "items": members.tolist(), "x": float(m_pts[:, 0].mean()), "y": float(m_pts[:, 1].mean()),
                    "size": float(ext.max()), **({"pts": m_pts} if with_points else {})})
    return out


def symbol_id(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]


def _context(scene: Scene, insts: list[dict]) -> None:
    """Adds to each instance: `arch` = architecture items within 1.5 m (high = inside a plan, low = a
    legend table), and `hint` = the text written right of it on the same row (lamp labels like
    "J 60 W", legend descriptions like "AVİZE")."""
    if not insts:
        return
    bb = scene.bboxes()
    centres = np.column_stack([(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2])
    arch_layer = np.array(["MIMARI" in inventory.fold(n) for n in scene.meta["layers"]])
    arch = np.flatnonzero(arch_layer[scene.layer] & np.isfinite(centres[:, 0]))
    arch_tree = cKDTree(centres[arch]) if len(arch) else None
    txt = np.flatnonzero((scene.text != "") & np.isfinite(bb[:, 0]))
    tpos = np.column_stack([bb[txt, 0], (bb[txt, 1] + bb[txt, 3]) / 2])
    text_tree = cKDTree(tpos) if len(txt) else None
    for inst in insts:
        xy = [inst["x"], inst["y"]]
        inst["arch"] = int(arch_tree.query_ball_point(xy, 1.5, return_length=True)) if arch_tree else 0
        inst["hint"] = None
        if text_tree is None:
            continue
        dist, idx = text_tree.query([inst["x"] + inst["size"] / 2, inst["y"]], k=8, distance_upper_bound=3)
        for d, i in zip(np.atleast_1d(dist), np.atleast_1d(idx)):
            if not np.isfinite(d):
                break
            dx, dy = tpos[i, 0] - inst["x"], abs(tpos[i, 1] - inst["y"])
            if 0 < dx < 3 and dy < max(inst["size"], 0.15):
                inst["hint"] = inventory.fold(scene.text[txt[i]]).strip()[:40]
                break


def harvest_job(job) -> dict:
    SYMBOL_DIR.mkdir(parents=True, exist_ok=True)
    inv = inventory.load()
    finals = [(p["id"], f["sha"]) for p in inv["projects"] for f in p["files"] if f["role"] == "final"]
    finals = [(pid, sha) for pid, sha in finals if (process.scene_info(sha) or {}).get("ok")]
    job.progress(0, len(finals) + 1)
    stats: dict[str, dict] = defaultdict(lambda: {"files": set(), "instances": 0, "layers": Counter(),
                                                  "sizes": [], "hints": Counter()})
    example: dict[str, tuple[str, dict]] = {}
    for i, (pid, sha) in enumerate(finals, 1):
        scene = Scene.load(process.scene_path(sha))
        insts = instances(scene)
        _context(scene, insts)
        for inst in insts:
            s = stats[inst["block"]]
            s["files"].add(sha)
            s["instances"] += 1
            s["layers"][inst["layer"]] += 1
            s["sizes"].append(inst["size"])
            if inst["hint"]:
                s["hints"][inst["hint"]] += 1
            best = example.get(inst["block"])
            if best is None or inst["arch"] > best[1]["arch"]:      # show it inside a plan
                example[inst["block"]] = (sha, inst)
        job.progress(i)
        if job.cancelled:
            return {}
    names = sorted(stats, key=lambda n: (-len(stats[n]["files"]), -stats[n]["instances"]))
    by_scene: dict[str, list[str]] = defaultdict(list)
    for n in names:
        if len(stats[n]["files"]) >= 2:
            by_scene[example[n][0]].append(n)
    for sha, blocks in by_scene.items():
        scene = Scene.load(process.scene_path(sha))
        for n in blocks:
            inst = example[n][1]
            mark = np.full(scene.n, 2, dtype=np.int8)
            mark[inst["items"]] = 1
            r = max(inst["size"] * 2.0, 1.2)
            render(scene, SYMBOL_DIR / f"{symbol_id(n)}.png", mark,
                   bbox=(inst["x"] - r, inst["y"] - r, inst["x"] + r, inst["y"] + r), width=240,
                   colours={2: (170, 170, 185), 1: (210, 0, 0)}, thick={1: 2})
    catalog = []
    for n in names:
        s = stats[n]
        catalog.append({"block": n, "id": symbol_id(n), "files": len(s["files"]), "instances": s["instances"],
                        "layers": dict(s["layers"].most_common(3)), "size_m": round(float(np.median(s["sizes"])), 3),
                        "hints": s["hints"].most_common(5), "thumb": len(s["files"]) >= 2})
    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    job.progress(len(finals) + 1)
    common = sum(1 for c in catalog if c["files"] >= 2)
    job.log(say(f"{len(finals)} son çizim, {len(catalog)} farklı blok, {common} tanesi en az 2 projede",
                f"{len(finals)} final drawings, {len(catalog)} distinct blocks, {common} in at least 2 projects"))
    return {"finals": len(finals), "blocks": len(catalog)}


def load_proposals() -> dict[str, dict]:
    return json.loads(PROPOSALS.read_text(encoding="utf-8")) if PROPOSALS.exists() else {}


def load_types(confirmed_only: bool = False) -> dict[str, str]:
    types = json.loads(TYPES.read_text(encoding="utf-8")) if TYPES.exists() else {}
    if confirmed_only:
        types = {k: v for k, v in types.items() if v not in UNCONFIRMED}
    return types


def set_type(block: str, device_type: str, source: str = "engineer") -> None:
    if device_type not in DEVICE_TYPES:
        raise ValueError(device_type)
    SYMBOL_DIR.mkdir(parents=True, exist_ok=True)
    types = load_types()
    types[block] = device_type
    TYPES.write_text(json.dumps(types, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(SYMBOL_DIR / "types_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"block": block, "type": device_type, "source": source}, ensure_ascii=False) + "\n")
