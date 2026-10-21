"""Exploded symbols that are the same drawing in different files (ADR-024).

An exploded symbol's catalogue key (symbols.exploded_instances) counts its segments and line
length exactly, so one switch drawn by two people (a circle tessellated into 35 or 39 segments, a
slightly longer stem) becomes two keys, and only the one the engineer confirmed is a device. Here
every key gets a picture of its shape: points turned onto the long side of the minimum-area
rectangle, scaled to it and counted on a 16 x 16 grid. Mirrors and half turns of the grid are the
same shape. An unconfirmed group takes the type of the nearest confirmed key when the two pictures,
proportions, sizes and circle counts are close enough (MATCH_DIST, chosen by leaving each confirmed
key out and checking it is found again). A group that is a symbol with a wire stub attached is
grown part by part from a circle or dot until it looks like a confirmed symbol (peel).

Only the engineer's finals are read this way (devices.locate): it decides which of the engineer's
drawn items are devices, it does not place anything.
"""
from __future__ import annotations

import json

import numpy as np

from cadai.cad.scene import Scene
from cadai.i18n import say
from cadai.corpus import inventory, process
from cadai.placement import symbols

SHAPES = symbols.SYMBOL_DIR / "shapes.npz"
GRID = 16
PER_KEY = 3                      # instances kept per key (the same key in other files)
MATCH_DIST = 0.25                # leave-one-key-out on the 111 confirmed keys: 28/28 right up to here, errors from 0.27


def picture(pts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(unit-length 16 x 16 grid of the points on their own long axis, short / long, long side in m)."""
    a, short, long_ = long_axis(pts)
    c = pts - pts.mean(axis=0)
    u = c @ np.array([np.cos(a), np.sin(a)])              # along the long side
    v = c @ np.array([-np.sin(a), np.cos(a)])
    u, v = u - (u.max() + u.min()) / 2, v - (v.max() + v.min()) / 2
    half = max(long_ / 2, 1e-6)
    ix = np.clip(((u / half + 1) / 2 * GRID).astype(int), 0, GRID - 1)
    iy = np.clip(((v / half + 1) / 2 * GRID).astype(int), 0, GRID - 1)
    g = np.zeros((GRID, GRID))
    np.add.at(g, (iy, ix), 1.0)
    g = np.minimum(g, 3.0)                                 # long straight runs should not outweigh a circle
    g = g + 0.5 * (np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1))   # 1-cell blur
    g /= max(np.linalg.norm(g), 1e-9)
    return g, short / max(long_, 1e-9), long_


def long_axis(pts: np.ndarray) -> tuple[float, float, float]:
    """(angle in radians of the long side, short side, long side) of the minimum-area rectangle;
    exact: one side of that rectangle lies on a convex hull edge."""
    from scipy.spatial import ConvexHull, QhullError
    try:
        hull = pts[ConvexHull(pts).vertices]
    except (QhullError, ValueError):                       # collinear points: a line
        hull = pts
    e = np.roll(hull, -1, axis=0) - hull
    angles = np.unique(np.round(np.arctan2(e[:, 1], e[:, 0]) % (np.pi / 2), 6))
    best = None
    for a in angles:
        u = hull @ np.array([np.cos(a), np.sin(a)])
        v = hull @ np.array([-np.sin(a), np.cos(a)])
        w, h = np.ptp(u), np.ptp(v)
        if best is None or w * h < best[0] - 1e-12:
            best = (w * h, a if w >= h else a + np.pi / 2, min(w, h), max(w, h))
    return best[1], best[2], best[3]


def variants(g: np.ndarray) -> np.ndarray:
    """The four mirror / half-turn copies of a grid (the long axis is fixed), flattened."""
    return np.stack([g, g[:, ::-1], g[::-1, :], g[::-1, ::-1]]).reshape(4, -1)


def circles(key: str) -> int:
    return int(key.rsplit("-c", 1)[1].split("-")[0]) if "-c" in key else 0


def distance(qg, qratio, qsize, qc, rg, rratio, rsize, rc) -> np.ndarray:
    """Distance of one query shape to many reference shapes. rg: (n, 256) grids."""
    pic = np.sqrt(np.maximum(0, 2 - 2 * (variants(qg) @ rg.T).max(axis=0)))     # 0 same .. 1.41 disjoint
    return pic + 0.6 * np.abs(qratio - rratio) + 0.5 * np.abs(np.log(qsize / rsize)) + 0.15 * np.abs(qc - rc)


def collect_job(job) -> dict:
    """Shape pictures of every exploded key in every final -> DATA/symbols/shapes.npz."""
    inv = inventory.load()
    finals = [f["sha"] for p in inv["projects"] for f in p["files"] if f["role"] == "final"]
    finals = [sha for sha in finals if (process.scene_info(sha) or {}).get("ok")]
    job.progress(0, len(finals))
    keys, grids, ratios, sizes = [], [], [], []
    seen: dict[str, int] = {}
    for i, sha in enumerate(finals, 1):
        scene = Scene.load(process.scene_path(sha))
        for inst in symbols.exploded_instances(scene, with_points=True):
            k = inst["block"]
            if seen.get(k, 0) >= PER_KEY:
                continue
            seen[k] = seen.get(k, 0) + 1
            g, ratio, size = picture(inst["pts"])
            keys.append(k)
            grids.append(g.ravel())
            ratios.append(ratio)
            sizes.append(size)
        job.progress(i)
        if job.cancelled:
            return {}
    np.savez_compressed(SHAPES, keys=np.array(keys), grids=np.array(grids, dtype=np.float32),
                        ratios=np.array(ratios), sizes=np.array(sizes))
    job.log(say(f"{len(finals)} son çizim, {len(seen)} patlatılmış sembol, {len(keys)} şekil resmi",
                f"{len(finals)} final drawings, {len(seen)} exploded symbols, {len(keys)} shape pictures"))
    return {"finals": len(finals), "keys": len(seen)}


def _load():
    d = np.load(SHAPES)
    keys = d["keys"]
    return keys, d["grids"], d["ratios"], d["sizes"], np.array([circles(k) for k in keys])


def nearest(types: dict[str, str], exclude: str | None = None, data=None) -> dict[str, tuple[str, float, str]]:
    """For every exploded key: (type, distance, reference key) of the nearest confirmed key's shape.
    `exclude` leaves one confirmed key out of the references and only looks that key up (evaluation)."""
    keys, grids, ratios, sizes, circ = data or _load()
    ref = np.array([k in types and types[k] != "unsure" and k != exclude for k in keys])
    ask = keys == exclude if exclude else np.ones(len(keys), dtype=bool)
    if not ref.any():
        return {}
    rg, rr, rs, rc, rk = grids[ref], ratios[ref], sizes[ref], circ[ref], keys[ref]
    out: dict[str, tuple[str, float, str]] = {}
    for k, g, r, s, c in zip(keys[ask], grids[ask], ratios[ask], sizes[ask], circ[ask]):
        d = distance(g.reshape(GRID, GRID), r, s, c, rg, rr, rs, rc)
        j = int(np.argmin(d))
        if k not in out or d[j] < out[k][1]:
            out[str(k)] = (types[rk[j]], float(d[j]), str(rk[j]))
    return out


def evaluate(types: dict[str, str]) -> list[dict]:
    """Leave each confirmed key out; where does its nearest other confirmed key land, how far?"""
    rows, data = [], _load()
    for k, t in types.items():
        if t == "unsure" or not k.startswith("~"):
            continue
        got = nearest(types, exclude=k, data=data).get(k)
        if got:
            rows.append({"key": k, "type": t, "found": got[0], "dist": got[1], "ref": got[2]})
    return rows


PEEL_MAX_PARTS = 30
PEEL_MAX_M = 2.5                 # groups up to this size may hide a symbol with wire stubs attached


_CACHE: dict[int, "References"] = {}


def references(types: dict[str, str]) -> "References":
    """References for these confirmed types, built once per process (pool workers get the dict per task)."""
    k = hash(frozenset(types.items()))
    if k not in _CACHE:
        _CACHE.clear()
        _CACHE[k] = References(types)
    return _CACHE[k]


class References:
    """Confirmed shapes, loaded once, for matching symbols found while reading a drawing."""

    def __init__(self, types: dict[str, str], data=None):
        keys, grids, ratios, sizes, circ = data or _load()
        ref = np.array([k in types and types[k] != "unsure" for k in keys])
        self.keys, self.grids, self.ratios, self.sizes, self.circ = keys[ref], grids[ref], ratios[ref], sizes[ref], circ[ref]

    def match(self, pts: np.ndarray, n_circles: int) -> tuple[str, float]:
        """(nearest confirmed key, distance) for a point cloud."""
        if len(self.keys) == 0 or len(pts) < 3 or np.ptp(pts, axis=0).max() < 1e-3:
            return "", np.inf
        g, r, s = picture(pts)
        d = distance(g, r, s, n_circles, self.grids, self.ratios, self.sizes, self.circ)
        j = int(np.argmin(d))
        return str(self.keys[j]), float(d[j])


def part_points(scene: Scene, items) -> dict[int, np.ndarray]:
    """Points every 2 cm along each item's segments."""
    out = {}
    for i in items:
        sg = scene.segs[scene.seg_item == i]
        sg = sg[np.isfinite(sg).all(axis=1)]
        d = sg[:, 2:] - sg[:, :2]
        n = np.clip(np.ceil(np.hypot(d[:, 0], d[:, 1]) / 0.02).astype(int), 1, 100) + 1
        out[int(i)] = np.vstack([sg[k, :2] + d[k] * np.linspace(0, 1, n[k])[:, None] for k in range(len(sg))]) \
            if len(sg) else np.zeros((0, 2))
    return out


def peel(scene: Scene, items: list[int], refs: References) -> tuple[str, float, list[int]] | None:
    """A confirmed symbol hidden in a group of touching parts (a switch with a wire stub and the
    next tap dot attached). From a seed part (a circle, hatch or arc; any part in small groups) the
    symbol is grown by adding the touching part nearest to the seed, and every step is compared
    with the confirmed shapes. (key, distance, kept items) of the best step, or None."""
    from scipy.spatial import cKDTree
    if len(items) > PEEL_MAX_PARTS:
        return None
    pts = part_points(scene, items)
    items = [i for i in items if len(pts[i])]
    kind = {i: scene.meta["types"][scene.etype[i]] for i in items}
    if len(items) <= 8:
        seeds = items
    else:
        seeds = [i for i in items if kind[i] in ("CIRCLE", "HATCH", "ARC")][:8]
    owner = np.concatenate([[i] * len(pts[i]) for i in items])
    tree = cKDTree(np.vstack([pts[i] for i in items]))
    touch = {i: {int(o) for o in owner[np.unique(np.concatenate(tree.query_ball_point(pts[i], symbols.TOUCH_M)))]} - {i}
             for i in items}
    best = ("", np.inf, [])
    for seed in seeds:
        centre = pts[seed].mean(axis=0)
        keep, cloud = [seed], [pts[seed]]
        while True:
            if len(keep) >= 2:
                key, d = refs.match(np.vstack(cloud), sum(kind[i] == "CIRCLE" for i in keep))
                if d < best[1]:
                    best = (key, d, list(keep))
            nxt = {j for i in keep for j in touch[i]} - set(keep)
            if not nxt:
                break
            j = min(nxt, key=lambda j: np.hypot(*(pts[j] - centre).T).min())
            keep.append(j)
            cloud.append(pts[j])
    return best if best[1] <= MATCH_DIST else None


def typed_instances(scene: Scene, types: dict[str, str], refs: References) -> list[dict]:
    """symbols.instances with exploded symbols read by shape: a group whose key is not confirmed
    takes the key of the confirmed shape it is (or hides, see peel), centred on the kept parts."""
    out = symbols.block_instances(scene)
    for inst in symbols.exploded_instances(scene, max_m=PEEL_MAX_M):
        if inst["block"] in types:
            if inst["size"] <= symbols.EXPLODED_MAX_M:
                out.append(inst)
            continue
        found = peel(scene, inst["items"], refs)
        if found is None:
            if inst["size"] <= symbols.EXPLODED_MAX_M:
                out.append(inst)
            continue
        key, dist, keep = found
        pts = np.vstack(list(part_points(scene, keep).values()))
        ext = np.ptp(pts, axis=0)
        out.append({**inst, "block": key, "items": keep, "x": float(pts[:, 0].mean()), "y": float(pts[:, 1].mean()),
                    "size": float(ext.max()), "shape_of": inst["block"], "shape_dist": round(dist, 3)})
    return out
