"""Placement dataset (plan-placement §2-3, ADR-007/008/010, ADR-022).

Per aligned pair and floor plan:
- one row per room: type and name, area, bounding box, the devices the engineer drew in it (raw
  coordinates, devices.py) and the raw drawing's furniture in it (furniture.py) -> rooms.jsonl;
- the room's candidate positions with their features (candidates.py) and, per device class,
  whether the engineer drew one within 40 cm -> candidates/<pair>.npz.

Rooms are read from **our cleanup** of the raw drawing (the toolbox, ADR-021), not from the lines
the engineer kept: a new drawing only ever has our cleanup, so the model must learn on the same
kind of plan (ADR-022). The floor plans are the engineer's (the views the alignment found), so
building the dataset needs no Claude call.

Pairs run in parallel worker processes (each writes its own part file); the parts are merged in a
fixed order at the end. Device types: confirmed symbol types only, unless `use_proposals` (then
everything written is marked and must not be used to train a model that ships).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

from cadai import config
from cadai import i18n
from cadai.i18n import say
from cadai.cad.scene import Scene
from cadai.cleanup.align import KEEP, OUT_OF_VIEW
from cadai.corpus import process
from cadai.placement import candidates as cand_mod
from cadai.placement import devices, furniture, furniture_shapes, rooms, symbols

PLACEMENT_DIR = config.DATA / "placement"
DATASET = PLACEMENT_DIR / "rooms.jsonl"
CANDIDATE_DIR = PLACEMENT_DIR / "candidates"
PART_DIR = PLACEMENT_DIR / "parts"
KEEP_DIR = PLACEMENT_DIR / "keep"            # cleanup keep masks per raw drawing and cleanup model
QUALITY = PLACEMENT_DIR / "quality.json"
SNAP_M = 0.4               # devices drawn on a wall belong to the room on either side within this
MATCH_M = 0.4              # a candidate is positive when a device of the class is this close
CLASSES = cand_mod.WALL_CLASSES + cand_mod.CEILING_CLASSES


def base_drawing(pair: dict, raw: Scene, labels: np.ndarray, base: str = "tools") -> np.ndarray:
    """Keep mask of the plan the rooms are read from: our toolbox cleanup inside the engineer's
    views ("tools"), or the lines the engineer kept ("engineer", the older dataset)."""
    if base == "engineer":
        return labels == KEEP
    from cadai.cleanup import apply, tools, train
    cleanup = apply.latest_model()
    cached = KEEP_DIR / f"{pair['raw']}__{cleanup.name}.npy"       # the same cleanup on the same plan
    if cached.exists():
        return np.load(cached)
    p, *_ = apply.predict(raw, cleanup, raw_features=train.feature_matrix(pair["raw"]))
    result = tools.run(raw, labels != OUT_OF_VIEW, p)
    keep = result.decision != tools.DELETE
    KEEP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cached, keep)
    return keep


def window_points(scene: Scene, keep: np.ndarray) -> np.ndarray:
    """Points every 25 cm along the kept lines of window / opening layers."""
    from cadai.cleanup import layers
    from cadai.cleanup.align import _sample
    kind = layers.item_types(scene, layers.classify(scene)[0])
    sel = keep & np.isin(kind, ["window", "opening"])
    segs = scene.segs[sel[scene.seg_item] & np.isfinite(scene.segs).all(1)]
    return _sample(segs, 0.25)[0] if len(segs) else np.zeros((0, 2))


def wall_points(scene: Scene, keep: np.ndarray) -> np.ndarray:
    """Points every 5 cm along the kept lines (no texts)."""
    from cadai.cleanup.align import _sample
    segs = scene.segs[keep[scene.seg_item] & (scene.text == "")[scene.seg_item] & np.isfinite(scene.segs).all(1)]
    return _sample(segs, 0.05)[0] if len(segs) else np.zeros((0, 2))


def swing_points(scene: Scene, where: np.ndarray) -> np.ndarray:
    """Door swings (hinge, two arc ends) among the door, opening, wall and unknown layers' items."""
    from cadai.cleanup import elements, layers
    kind = layers.item_types(scene, layers.classify(scene)[0])
    types = np.array(scene.meta["types"])[scene.etype]
    return elements.swing_points(scene, where & np.isin(kind, ["door", "opening", "window", "wall", "unknown"]), types)


def pair_rows(pair: dict, types: dict[str, str], base: str = "tools",
              furn_types: dict[str, str] | None = None) -> tuple[list[dict], dict, dict]:
    """Rooms, candidates and a quality record of one pair. `furn_types`: furniture signature -> type
    (default: the engineer's confirmed types)."""
    info = process.label_info(pair["project"], pair["final"])
    raw = Scene.load(process.scene_path(pair["raw"]))
    final = Scene.load(process.scene_path(pair["final"]))
    with np.load(process.label_path(pair["project"], pair["final"])) as z:
        labels = z["labels"]
    keep = base_drawing(pair, raw, labels, base)
    windows = window_points(raw, keep)
    swings = swing_points(raw, labels != OUT_OF_VIEW)
    walls = wall_points(raw, keep)
    devs = devices.locate(final, info["transforms"], types, raw, by_shape=True)
    furn = furniture.instances(raw, labels, with_shape=True) + \
        furniture.loose_instances(raw, (labels != OUT_OF_VIEW) & ~keep, with_shape=True)       # ADR-025
    furn_types = furniture_shapes.resolved_types() if furn_types is None else furn_types
    furn_shapes = furn                            # every piece, for the shape classes (furniture_shapes)
    furn = [f for f in furn if furn_types.get(f["sig"]) not in furniture_shapes.NOT_FURNITURE]
    rows, cand = [], {"X": [], "Y": [], "xy": [], "room": []}
    seen: list[rooms.Rooms] = []                  # view regions overlap: a room is taken once
    dev_xy = np.array([[d["x"], d["y"]] for d in devs]).reshape(-1, 2)
    dev_cls = np.array([d["type"] for d in devs])
    furn_xy = np.array([[f["x"], f["y"]] for f in furn]).reshape(-1, 2)
    placed = np.zeros(len(devs), dtype=bool)
    for v, t in enumerate(info["transforms"]):
        region = tuple(t["region"])
        rm = rooms.read_rooms(raw, keep, region, swings)
        sides = cand_mod.door_sides(swings, walls, rm)
        d_room, f_room = rm.at(dev_xy, snap_m=SNAP_M), rm.at(furn_xy, snap_m=SNAP_M)
        for r in rm.rooms:
            if any(prev.at(np.array([r["centre"]]))[0] for prev in seen):
                continue
            in_room = d_room == r["id"]
            placed |= in_room
            room_furn = [f for f, k in zip(furn, f_room) if k == r["id"]]
            rows.append({
                "project": pair["project"], "final": pair["final"], "view": v, "room": r["id"],
                "type": r["type"], "names": r["names"], "area_m2": r["area_m2"], "bbox": r["bbox"],
                "centre": r["centre"],
                "devices": [{k: d[k] for k in ("type", "x", "y")} for d, ok in zip(devs, in_room) if ok],
                "furniture": [{"sig": f["sig"], "type": furn_types.get(f["sig"]), "x": round(f["x"], 3),
                               "y": round(f["y"], 3), "short": round(f["short"], 2), "long": round(f["long"], 2),
                               "angle": round(f["angle"], 1)} for f in room_furn],
            })
            xy, X = cand_mod.room_candidates(rm, r, room_furn, furn_types, windows, sides)
            if not len(xy):
                continue
            Y = np.zeros((len(xy), len(CLASSES)), dtype=bool)
            for c, name in enumerate(CLASSES):
                pts = dev_xy[in_room & (dev_cls == name)]
                if len(pts):
                    d, _ = cKDTree(pts).query(xy)
                    Y[:, c] = d <= MATCH_M
            cand["X"].append(X)
            cand["Y"].append(Y)
            cand["xy"].append(xy)
            cand["room"].append(np.full(len(xy), len(rows) - 1))
        seen.append(rm)
    stacked = {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in cand.items()}
    stacked["furniture"] = furn_shapes            # with shape pictures, for furniture_shapes
    quality = {"project": pair["project"], "final": pair["final"], "split": pair["split"], "views": len(info["transforms"]),
               "rooms": len(rows), "typed": sum(1 for r in rows if r["type"]), "devices": len(devs),
               "devices_in_rooms": int(placed.sum()),
               "by_type": {c: [int((dev_cls == c).sum()), int((placed & (dev_cls == c)).sum())] for c in sorted(set(dev_cls))}}
    return rows, stacked, quality


def _key(pair: dict) -> str:
    return f"{pair['project']}__{process.key(pair['final'])}"


def _pair_task(pair: dict, types: dict[str, str], base: str, use_proposals: bool,
               furn_types: dict[str, str] | None = None) -> dict:
    """Worker process: one pair -> parts/<key>.json (rooms, quality) + candidates/<key>.npz (local room indices)."""
    try:
        rows, cand, quality = pair_rows(pair, types, base, furn_types)
    except Exception as exc:  # noqa: BLE001 - one bad pair is reported, the rest continue
        return {"ok": False, "key": _key(pair), "error": f"{type(exc).__name__}: {exc}"}
    for r in rows:
        r["proposals_used"] = use_proposals or furn_types is not None
        r["split"] = pair["split"]
    (PART_DIR / f"{_key(pair)}.json").write_text(json.dumps({"rows": rows, "quality": quality}, ensure_ascii=False),
                                                 encoding="utf-8")
    furniture_shapes.save_pair(_key(pair), pair["project"], pair["raw"], cand["furniture"])
    if len(cand["X"]):
        np.savez_compressed(CANDIDATE_DIR / f"{_key(pair)}.npz", X=cand["X"], Y=cand["Y"], xy=cand["xy"],
                            room=cand["room"], proposals_used=use_proposals or furn_types is not None)
    return {"ok": True, "key": _key(pair), **{k: quality[k] for k in ("rooms", "typed", "devices", "devices_in_rooms")}}


def build_job(job, use_proposals: bool = False, base: str = "tools", limit: int | None = None,
              furniture_proposals: bool = False) -> dict:
    """`furniture_proposals`: also use the LLM's furniture names (not low confidence) where the engineer
    has not confirmed one; rows are then marked proposals_used (an experiment, not a shipping model)."""
    from cadai.cleanup.train import labelled_pairs
    types = symbols.load_types(confirmed_only=True)
    if use_proposals:
        types = {**{b: p["type"] for b, p in symbols.load_proposals().items()}, **types}
    if not types:
        raise RuntimeError(str(say("Onaylı sembol tipi yok; önce Semboller sayfasında tipleri onaylayın.",
                                   "No confirmed symbol types; confirm the types on the Symbols page first.")))
    pairs = labelled_pairs()[:limit] if limit else labelled_pairs()
    for d in (CANDIDATE_DIR, PART_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for old in list(d.glob("*.npz")) + list(d.glob("*.json")):   # indices refer to this run's rooms.jsonl
            old.unlink()

    def describe(task, res):
        if not res.get("ok"):
            text = f"✗ {res.get('key', task[0]['project'])}: {res.get('error')}"
            return say(text, text)
        return say(f"✓ {res['key']}: {res['rooms']} oda ({res['typed']} tipli), "
                   f"{res['devices_in_rooms']}/{res['devices']} cihaz odalarda",
                   f"✓ {res['key']}: {res['rooms']} rooms ({res['typed']} typed), "
                   f"{res['devices_in_rooms']}/{res['devices']} devices in rooms")

    tools_base = base == "tools"
    job.log(say(f"{len(pairs)} çift; odalar {'temizlik araçlarımızdan' if tools_base else 'mühendisin çizgilerinden'}",
                f"{len(pairs)} pairs; rooms from {'our cleanup tools' if tools_base else 'the engineer lines'}"))
    furn_types = None
    if furniture_proposals:
        furn_types = {**{k: v["type"] for k, v in furniture.load_proposals().items() if i18n.bi(v.get("confidence") or "")["tr"] != "düşük"},
                      **furniture.load_types()}
        job.log(say(f"mobilya tipleri: {len(furn_types)} (LLM önerileriyle, deney)",
                    f"furniture types: {len(furn_types)} (with LLM proposals, experiment)"))
    process._run_pool(job, [(p, types, base, use_proposals, furn_types) for p in pairs], _pair_task, describe)
    # merge in a fixed order; candidate room indices become rooms.jsonl line numbers
    n_rooms, qualities = 0, []
    tmp = DATASET.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for pair in pairs:
            part = PART_DIR / f"{_key(pair)}.json"
            if not part.exists():
                continue
            data = json.loads(part.read_text(encoding="utf-8"))
            for r in data["rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            npz = CANDIDATE_DIR / f"{_key(pair)}.npz"
            if npz.exists():
                with np.load(npz) as z:
                    arrays = {k: z[k] for k in z.files}
                arrays["room"] = arrays["room"] + n_rooms
                np.savez_compressed(npz, **arrays)
            n_rooms += len(data["rows"])
            qualities.append(data["quality"])
    tmp.replace(DATASET)
    QUALITY.write_text(json.dumps(qualities, ensure_ascii=False, indent=0), encoding="utf-8")
    total = {k: sum(q[k] for q in qualities) for k in ("rooms", "typed", "devices", "devices_in_rooms")}
    job.log(say(f"{len(qualities)}/{len(pairs)} çift: {total['rooms']} oda ({total['typed']} tipli), "
                f"{total['devices_in_rooms']}/{total['devices']} cihaz odalarda",
                f"{len(qualities)}/{len(pairs)} pairs: {total['rooms']} rooms ({total['typed']} typed), "
                f"{total['devices_in_rooms']}/{total['devices']} devices in rooms"))
    return {"pairs": len(pairs), "ok_pairs": len(qualities), **total, "base": base, "proposals_used": use_proposals}
