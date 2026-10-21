"""Which items of the architect's drawing go into the engineer's base, given the layer types (ADR-019).

On a kept layer the engineer still leaves out a few kinds of parts, seen in the archive:
- hatching: a HATCH stays as its boundary only; exploded hatch lines (ArchiCAD writes column
  hatching as loose diagonal lines on a pen layer) go: short lines at a slant with many parallel
  neighbours;
- door swings: arcs on door and wall layers (some offices draw doors on the wall layer; true ARCs, or polylines whose points lie on a quarter-ish
  circle of door size: bulged polylines, or exploded into short pieces), the leaf and frame
  pieces inside the swing's quarter (on wall layers only the leaf), and small marker circles;
- short marks on opening layers sticking out of the wall (finish.opening_ticks);
- every text except room names, dimensions, dashed lines (except stairs);
- everything outside the floor plans (sections, elevations, site plan): the cleanup model's view
  part decides what is a plan.
Layers of unknown type are decided item by item by the cleanup model (its best guess).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from cadai.cad.scene import ATOMIC, TEXT, Scene
from cadai.cleanup.finish import opening_ticks, room_names
from cadai.cleanup.layers import KEEP_TYPES, item_types

HATCH_MAX_M = 1.5          # an exploded hatch line is short
HATCH_SIBLINGS = 4         # ... and has this many parallel lines nearby
HATCH_NEAR_M = 1.0
HATCH_NEIGHBOURS = 64       # nearest lines looked at
SWING_R = (0.5, 1.3)       # door leaf widths, metres
SWING_SWEEP = (60, 120)    # degrees
MARKER_R = 0.3


def _angles(scene: Scene) -> np.ndarray:
    s = scene.segs
    return np.degrees(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])) % 180


def hatch_lines(scene: Scene, cand: np.ndarray) -> np.ndarray:
    """Single-segment slanted short lines among `cand` with >= HATCH_SIBLINGS parallel lines within 1 m."""
    nseg = np.bincount(scene.seg_item, minlength=scene.n)
    length = scene.lengths()
    first = np.full(scene.n, -1)
    first[scene.seg_item[::-1]] = np.arange(len(scene.seg_item))[::-1]
    ang = np.full(scene.n, np.nan)
    ok = first >= 0
    ang[ok] = _angles(scene)[first[ok]]
    slant = np.minimum(ang % 90, 90 - ang % 90) > 10
    pool = np.flatnonzero(cand & (nseg == 1) & (length <= HATCH_MAX_M) & slant)
    out = np.zeros(scene.n, dtype=bool)
    if len(pool) <= HATCH_SIBLINGS:
        return out
    seg = scene.segs[first[pool]]
    mid = np.column_stack([(seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2])
    # the nearest HATCH_NEIGHBOURS lines within 1 m only: an all-pairs search runs out of memory on
    # densely hatched sheets (tens of thousands of lines per square metre)
    k = min(HATCH_NEIGHBOURS, len(pool))
    dist, nb = cKDTree(mid).query(mid, k=k, distance_upper_bound=HATCH_NEAR_M)
    a = np.r_[ang[pool], np.nan]                       # index len(pool) = "no neighbour"
    d = np.abs(a[nb] - a[:-1, None])
    parallel = (np.minimum(d, 180 - d) < 1.5) & np.isfinite(dist)
    out[pool[parallel.sum(axis=1) - 1 >= HATCH_SIBLINGS]] = True
    return out


ARC_PIECE_M = 0.25         # exploded arcs: chains of short straight pieces
LEAF_END_M = 0.15          # margin around a swing's quarter for its leaf and frame pieces


def _swing(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(centre, the two arc ends, inlier mask) when enough of the points lie on one circle of door
    size through a door-like angle. Exploded doors come as one touching group of arc, leaf and frame
    pieces, so the fit is repeated without the points farthest off the circle."""
    keep = np.ones(len(pts), dtype=bool)
    for _ in range(6):
        if keep.sum() < 5:
            return None
        q = pts[keep]
        A = np.column_stack([2 * q, np.ones(len(q))])
        sol, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
        centre = sol[:2]
        r_all = np.hypot(*(pts - centre).T)
        radius = np.median(r_all[keep])
        dev = np.abs(r_all - radius)
        if dev[keep].max() <= 0.02:
            break
        keep &= dev <= max(np.percentile(dev[keep], 70), 0.02)
    else:
        return None
    if not SWING_R[0] <= radius <= SWING_R[1]:
        return None
    keep = dev <= 0.02
    ang = np.arctan2(pts[keep, 1] - centre[1], pts[keep, 0] - centre[0])
    ref = np.angle(np.exp(1j * ang).mean())                         # middle direction of the arc
    rel = np.degrees(np.angle(np.exp(1j * (ang - ref))))
    if not (SWING_SWEEP[0] <= rel.max() - rel.min() <= SWING_SWEEP[1]) or keep.sum() < 8:
        return None
    on = pts[keep]
    return centre, on[[np.argmin(rel), np.argmax(rel)]], keep


def _swing_fits(scene: Scene, door: np.ndarray, types: np.ndarray):
    """Every door swing among `door` items: (items, points, owner of each point, (centre, arc ends, on))."""
    from cadai.cleanup.review import object_groups

    nseg = np.bincount(scene.seg_item, minlength=scene.n)
    order = np.argsort(scene.seg_item, kind="stable")
    starts = np.searchsorted(scene.seg_item[order], np.arange(scene.n + 1))

    def points(idx):
        """Both ends of every segment, and the item each point belongs to."""
        segs = np.vstack([scene.segs[order[starts[i]:starts[i + 1]]] for i in idx])
        owner = np.concatenate([np.full(starts[i + 1] - starts[i], i) for i in idx])
        return np.vstack([segs[:, :2], segs[:, 2:]]), np.concatenate([owner, owner])

    candidates = [[i] for i in np.flatnonzero(door & (types == "ARC") | (door & (nseg >= 4) & ~scene.closed))]
    pieces = door & (nseg == 1) & (scene.lengths() <= ARC_PIECE_M) & (types != "ARC")
    gid = object_groups(scene, pieces)
    g_order = np.argsort(gid, kind="stable")
    g_sorted = gid[g_order]
    bounds = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if g_sorted[a] >= 0 and b - a >= 4:
            candidates.append(list(g_order[a:b]))
    for idx in candidates:
        pts, owner = points(idx)
        found = _swing(pts)
        if found:
            yield idx, pts, owner, found


def swing_points(scene: Scene, door: np.ndarray, types: np.ndarray) -> np.ndarray:
    """(n, 3, 2): per door swing the hinge (arc centre) and the two arc ends (placement: the switch
    goes beside the door on its latch side, ADR-023)."""
    out = [np.vstack([centre, ends]) for _, _, _, (centre, ends, _) in _swing_fits(scene, door, types)]
    return np.array(out).reshape(-1, 3, 2)


def door_swings(scene: Scene, door: np.ndarray, types: np.ndarray, frame: np.ndarray | None = None) -> np.ndarray:
    """On door layers: swing arcs (ARCs, bulged polylines, or exploded into short pieces), the leaf
    and frame pieces inside the swing's quarter, and small marker circles."""
    out = door & (types == "CIRCLE") & (scene.radius > 0) & (scene.radius <= MARKER_R)
    nseg = np.bincount(scene.seg_item, minlength=scene.n)
    boxes, hinges = [], []
    for idx, pts, owner, found in _swing_fits(scene, door, types):
        if found:
            centre, arc_ends, on = found
            off = np.unique(owner[~on])
            out[np.setdiff1d(np.unique(owner[on]), off)] = True        # every point of the item on the circle
            q = np.vstack([centre, arc_ends, pts[on]])
            boxes.append(np.r_[q.min(0) - LEAF_END_M, q.max(0) + LEAF_END_M])
            hinges.append(centre)
    if not boxes:
        return out
    # the leaf and the frame pieces: items of door layers (`frame`) inside the swing's quarter; on
    # other layers (doors drawn on the wall layer) only straight leaves inside it, so wall ends stay
    frame = door if frame is None else frame & door
    bb = scene.bboxes()
    first = np.full(scene.n, -1)
    first[scene.seg_item[::-1]] = np.arange(len(scene.seg_item))[::-1]
    leafish = (nseg == 1) & (scene.lengths() >= SWING_R[0] - LEAF_END_M)
    idx = np.flatnonzero(leafish)
    seg = scene.segs[first[idx]]
    tree = cKDTree(np.array(hinges))
    at_hinge = np.minimum(tree.query(seg[:, :2])[0], tree.query(seg[:, 2:])[0]) <= LEAF_END_M
    leafish[idx[~at_hinge]] = False                                   # a leaf starts at the hinge
    rest = np.flatnonzero(door & ~out & np.isfinite(bb[:, 0]) & (frame | leafish))
    for x0, y0, x1, y1 in boxes:
        inside = (bb[rest, 0] >= x0) & (bb[rest, 1] >= y0) & (bb[rest, 2] <= x1) & (bb[rest, 3] <= y1)
        out[rest[inside]] = True
    return out


def select(scene: Scene, layer_types, in_plan: np.ndarray, model_keep: np.ndarray) -> tuple[np.ndarray, dict]:
    """Keep mask over items and counts of what each rule did."""
    kind = item_types(scene, layer_types)
    types = np.array(scene.meta["types"])[scene.etype]
    is_text = np.isin(types, list(TEXT))
    keep = in_plan & (np.isin(kind, list(KEEP_TYPES)) | ((kind == "unknown") & model_keep))
    keep &= ~is_text & ~np.isin(types, list(ATOMIC))
    stats = {"by_layer_type": int(keep.sum())}
    if scene.dashed is not None:
        dashed = keep & scene.dashed & (kind != "stair")
        keep &= ~dashed
        stats["dashed"] = int(dashed.sum())
    hatch = hatch_lines(scene, keep & np.isin(kind, ["wall", "column", "unknown"]))
    keep &= ~hatch
    doorish = np.isin(kind, ["door", "opening", "window"])
    swings = door_swings(scene, keep & (doorish | np.isin(kind, ["wall", "unknown"])), types, frame=doorish)
    keep &= ~swings
    ticks = opening_ticks(scene, keep.astype(np.int8), in_plan)       # sill / jamb marks sticking out of walls
    keep &= ~ticks
    names = room_names(scene, keep.astype(np.int8), in_plan)
    keep[names] = True
    stats.update(hatch_lines=int(hatch.sum()), door_swings=int(swings.sum()), opening_ticks=int(ticks.sum()), room_names=int(len(names)),
                 kept=int(keep.sum()))
    return keep, stats


FILL_SHORT_M = 0.8          # a column or a shear wall is at most this thick
FILL_LONG_M = 8.0           # ... and at most this long
FILL_MIN_M = 0.25           # smaller fills are symbols (level marks, section arrows: 20 cm, in blocks)
FILL_SHARE = 0.7            # of the hatch's boundary length


def structural_fills(scene: Scene, cand: np.ndarray) -> np.ndarray:
    """Hatches among `cand` that are columns / shear walls, whatever their layer: their boundary
    loops are narrow (short side <= 80 cm, long side <= 8 m), like a column or a wall piece, not a
    room-sized floor pattern; loops of at least 4 sides, at least 25 cm long, outside blocks (fills in
    blocks are symbols: level marks, section arrows, north arrows). Revit / AIA exports draw them as one filled region per floor on a
    detail layer ("A-DETL-GENF", Ali Kemal ARSLAN). The engineer keeps their outline (ADR-018, ADR-021)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    types = np.array(scene.meta["types"])[scene.etype]
    out = np.zeros(scene.n, dtype=bool)
    order = np.argsort(scene.seg_item, kind="stable")
    starts = np.searchsorted(scene.seg_item[order], np.arange(scene.n + 1))
    for i in np.flatnonzero(cand & (types == "HATCH") & (scene.block_top < 0)):
        segs = scene.segs[order[starts[i]:starts[i + 1]]]
        segs = segs[np.isfinite(segs).all(axis=1)]
        if len(segs) < 3:
            continue
        ends = np.round(np.vstack([segs[:, :2], segs[:, 2:]]) / 0.005).astype(np.int64)
        _, node = np.unique(ends, axis=0, return_inverse=True)
        node = node.ravel()
        m = len(segs)
        graph = coo_matrix((np.ones(m), (node[:m], node[m:])), shape=(node.max() + 1,) * 2)
        _, comp = connected_components(graph, directed=False)
        loop = comp[node[:m]]
        seg_len = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        narrow = 0.0
        for k in np.unique(loop):
            pts = np.vstack([segs[loop == k, :2], segs[loop == k, 2:]])
            short, long_ = _min_rect(pts)
            if (loop == k).sum() >= 4 and short <= FILL_SHORT_M and FILL_MIN_M <= long_ <= FILL_LONG_M:
                narrow += seg_len[loop == k].sum()
        out[i] = narrow >= FILL_SHARE * seg_len.sum()
    return out


def _min_rect(pts: np.ndarray) -> tuple[float, float]:
    """Sides of the minimum-area rectangle around the points (5-degree search)."""
    c = pts - pts.mean(axis=0)
    best = None
    for deg in range(0, 90, 5):
        a = np.radians(deg)
        r = c @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        w, h = np.ptp(r[:, 0]), np.ptp(r[:, 1])
        if best is None or w * h < best[0]:
            best = (w * h, min(w, h), max(w, h))
    return best[1], best[2]
