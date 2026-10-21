"""Devices of an engineer's final, located in the architect's raw drawing (plan-placement §2, ADR-007).

The raw drawing still has the furniture (bed, counter, WC…), the final has the devices. The
cleanup alignment (cleanup/align.py) already found, per floor plan, the transform raw -> final;
its inverse puts every device of the final onto the raw plan it belongs to. A device is assigned to
the view whose region (raw coordinates) contains its back-mapped position; devices outside every
view (legend tables, schematics) are dropped.

Device types come from the confirmed symbol types (symbols.py). With `use_proposals=True` the
LLM proposals are used too: only for looking at the data, never for training.
"""
from __future__ import annotations

import json

import numpy as np

from cadai import config
from cadai.i18n import say
from cadai.cad.scene import Scene
from cadai.corpus import process
from cadai.placement import symbols

DEVICE_DIR = config.DATA / "devices"


def inverse(t: dict, xy: np.ndarray) -> np.ndarray:
    """final -> raw for one alignment transform (see align.Transform.apply)."""
    p = xy - (t["tx"], t["ty"])
    for _ in range(t["rot"] % 4):                      # undo quarter turns: (x, y) -> (y, -x)
        p = np.column_stack([p[:, 1], -p[:, 0]])
    return p / t["scale"]


LOCAL_M = 2.0              # the engineer's lines around a device that must sit on the architect's
ON_LINE_M = 0.05           # ... within this, after mapping back
DUPLICATE_M = 0.25         # same type this close in raw coordinates: one device drawn on several floors
MIN_AGREE = 0.4            # share of those lines; below it the device is in no plan (schematics, tables)


def locate(final: Scene, transforms: list[dict], types: dict[str, str], raw: Scene | None = None,
           by_shape: bool = False) -> list[dict]:
    """Devices of the final in raw coordinates. With `raw`, every device takes the transform under
    which the engineer's lines around it land on the architect's lines (a region may cover several
    floors, and riser diagrams have devices too); devices whose surroundings match no plan are
    dropped. Without `raw`: the first transform whose region contains the back-mapped position.
    `by_shape`: exploded symbols are also read by their shape (shapes.py, ADR-024)."""
    from cadai.placement import shapes
    out = []
    found = shapes.typed_instances(final, types, shapes.references(types)) if by_shape and shapes.SHAPES.exists()         else symbols.instances(final)
    insts = [i for i in found if types.get(i["block"]) not in (None, "none")]
    if not insts or not transforms:
        return out
    xy = np.array([[i["x"], i["y"]] for i in insts])
    best = np.full(len(insts), -1)
    raw_xy = np.zeros_like(xy)
    if raw is None:
        for k, t in enumerate(transforms):
            p = inverse(t, xy)
            x0, y0, x1, y1 = t["region"]
            inside = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1) & (best < 0)
            best[inside] = k
            raw_xy[inside] = p[inside]
    else:
        best, raw_xy = _by_local_lines(final, raw, transforms, xy, insts)
    taken: dict[str, list] = {}           # raw positions per type (raw coordinates are shared by every view)
    for inst, k, p in zip(insts, best, raw_xy):
        if k < 0:
            continue
        # floors drawn alike (1st..4th floor) all map onto the architect's one typical plan: count once
        same = taken.setdefault(types[inst["block"]], [])
        if any(np.hypot(*(p - q)) < DUPLICATE_M for q in same):
            continue
        same.append(p)
        out.append({"type": types[inst["block"]], "block": inst["block"], "x": round(float(p[0]), 3),
                    "y": round(float(p[1]), 3), "view": int(k), "size": round(inst["size"] / transforms[k]["scale"], 3)})
    return out


def _by_local_lines(final: Scene, raw: Scene, transforms: list[dict], xy: np.ndarray, insts: list[dict]):
    from scipy.spatial import cKDTree

    from cadai.cleanup.align import _sample
    from cadai.corpus.inventory import fold
    # the final's architecture: no texts, no blocks (symbols), nothing on electrical layers (cables)
    electric = np.array([any(w in fold(n) for w in symbols.ELECTRIC_LAYER_WORDS) for n in final.meta["layers"]])
    lines = (final.text == "") & (final.block_top < 0) & ~electric[final.layer]
    fseg = final.segs[lines[final.seg_item] & np.isfinite(final.segs).all(1)]
    fpts, _ = _sample(fseg, 0.25)
    # the architect's lines inside the views only, every 5 cm (a whole sheet at 2 cm took 9 GB)
    rs = raw.segs[np.isfinite(raw.segs).all(1)]
    in_views = np.zeros(len(rs), dtype=bool)
    for t in transforms:
        x0, y0, x1, y1 = t["region"]
        in_views |= (np.maximum(rs[:, 0], rs[:, 2]) >= x0) & (np.minimum(rs[:, 0], rs[:, 2]) <= x1) &                     (np.maximum(rs[:, 1], rs[:, 3]) >= y0) & (np.minimum(rs[:, 1], rs[:, 3]) <= y1)
    rpts, _ = _sample(rs[in_views], 0.05)
    if not len(rpts):
        return np.full(len(xy), -1), np.zeros_like(xy)
    raw_tree = cKDTree(rpts)
    near = cKDTree(fpts).query_ball_point(xy, LOCAL_M) if len(fpts) else [[] for _ in xy]
    best = np.full(len(xy), -1)
    raw_xy = np.zeros_like(xy)
    score = np.zeros(len(xy))
    for k, t in enumerate(transforms):
        p = inverse(t, xy)
        x0, y0, x1, y1 = t["region"]
        inside = (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
        for i, nb in enumerate(near):
            if len(nb) < 4 or not inside[i]:
                continue
            d, _ = raw_tree.query(inverse(t, fpts[nb]), distance_upper_bound=ON_LINE_M)
            agree = float(np.isfinite(d).mean())
            if agree >= MIN_AGREE and agree > score[i]:
                score[i], best[i], raw_xy[i] = agree, k, p[i]
    return best, raw_xy


def build_job(job, use_proposals: bool = False) -> dict:
    """Devices in raw coordinates for every aligned pair -> DATA/devices/<project>__<final>.json."""
    from cadai.cleanup.train import labelled_pairs
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    types = symbols.load_types(confirmed_only=True)
    if use_proposals:
        types = {**{b: p["type"] for b, p in symbols.load_proposals().items()}, **types}
    if not types:
        raise RuntimeError(str(say("Onaylı sembol tipi yok; önce Semboller sayfasında tipleri onaylayın.",
                                   "No confirmed symbol types; confirm the types on the Symbols page first.")))
    pairs = labelled_pairs()
    job.progress(0, len(pairs))
    total = 0
    for i, pair in enumerate(pairs, 1):
        info = process.label_info(pair["project"], pair["final"])
        final = Scene.load(process.scene_path(pair["final"]))
        raw = Scene.load(process.scene_path(pair["raw"]))
        devs = locate(final, info["transforms"], types, raw, by_shape=True)
        total += len(devs)
        out = DEVICE_DIR / f"{pair['project']}__{process.key(pair['final'])}.json"
        out.write_text(json.dumps({"pair": pair, "proposals_used": use_proposals, "devices": devs},
                                  ensure_ascii=False), encoding="utf-8")
        job.progress(i)
    job.log(say(f"{len(pairs)} çift, {total} cihaz ham çizim koordinatında", f"{len(pairs)} pairs, {total} devices in raw drawing coordinates"))
    return {"pairs": len(pairs), "devices": total}
