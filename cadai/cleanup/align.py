"""Cleanup labels from a raw architect drawing and the engineer's final electrical drawing.

The engineer copies the floor plans out of the architect's file, explodes them, moves them to
other layers and coordinates, deletes what a drafter would delete, and draws the electrical
project on top. Handles don't survive (measured on real pairs), so labels come from geometry:

1. Both scenes are in metres (scene.py infers units). For each quarter-turn rotation and a few
   scale ratios, distinctive straight segments (same direction and length in both drawings) vote
   for the translation final - raw. Votes are binned at 2 cm.
2. The strongest translation peaks are verified: raw geometry is sampled every 5 cm, moved, and
   looked up among points sampled every 2 cm along the final's segments (KD-tree, 1.5 cm
   tolerance). Several peaks can be accepted: the engineer often moves each floor separately.
3. Per raw item, the share of its sample points found in the final decides the label:
   keep (>= 0.8), delete (<= 0.2, inside a floor plan), out_of_view (outside), partial (in between:
   excluded from training). "Inside a floor plan" = within 3 m of kept geometry shorter than 20 m
   (_view_mask, ADR-012).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from cadai.cad.scene import Scene
from cadai.i18n import say

KEEP, DELETE, OUT_OF_VIEW, PARTIAL, EMPTY = 1, 0, 2, 3, -1
LABEL_NAMES = {KEEP: "keep", DELETE: "delete", OUT_OF_VIEW: "out_of_view", PARTIAL: "partial", EMPTY: "empty"}

TOL_M = 0.015
RAW_STEP_M = 0.05
FINAL_STEP_M = 0.02
MIN_SEG_M = 0.3
BIN_M = 0.02
SCALES = (1.0, 10.0, 0.1, 100.0, 0.01)
MIN_VIEW_MATCH_M = 15.0          # a view must share at least this much matched line length
VIEW_MARGIN_M = 1.0
MIN_VOTES = 40                   # real pairs: 5th percentile of the strongest peak is ~94
VIEW_SHARE = 0.1                 # a view needs this share of the strongest view's matched length
VOTE_SHARE = 0.25                # a candidate needs this share of the strongest peak's votes


@dataclass
class Transform:
    rot: int            # quarter turns applied to raw (counter-clockwise)
    scale: float        # applied to raw, before rotation
    tx: float
    ty: float
    votes: int = 0
    matched_m: float = 0.0
    region: tuple[float, float, float, float] | None = None   # raw coordinates, metres

    def apply(self, xy: np.ndarray) -> np.ndarray:
        p = xy * self.scale
        for _ in range(self.rot):
            p = np.column_stack([-p[:, 1], p[:, 0]])
        return p + (self.tx, self.ty)

    def as_dict(self) -> dict:
        return {"rot": self.rot, "scale": self.scale, "tx": round(self.tx, 3), "ty": round(self.ty, 3),
                "votes": self.votes, "matched_m": round(self.matched_m, 1),
                "region": [round(v, 2) for v in self.region] if self.region else None}


# ------------------------------------------------------------------------------- sampling
def _sample(segs: np.ndarray, step: float, max_per_seg: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Points every `step` along each segment (both ends included), up to `max_per_seg` per segment
    (100 m at 2 cm). Returns (points, segment index)."""
    if not len(segs):
        return np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    n = np.clip(np.ceil(length / step).astype(np.int64), 1, max_per_seg) + 1
    seg_idx = np.repeat(np.arange(len(segs)), n)
    starts = np.repeat(np.cumsum(n) - n, n)
    t = (np.arange(n.sum()) - starts) / np.repeat(n - 1, n)
    p0, p1 = segs[seg_idx, :2], segs[seg_idx, 2:]
    return p0 + (p1 - p0) * t[:, None], seg_idx


def _signatures(segs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinctive key per straight segment (direction 1 degree, length 1 cm) and its midpoint."""
    d = segs[:, 2:] - segs[:, :2]
    length = np.hypot(d[:, 0], d[:, 1])
    ok = length >= MIN_SEG_M
    ang = (np.degrees(np.arctan2(d[ok, 1], d[ok, 0])) % 180).round().astype(np.int64) % 180
    key = ang * 10_000_000 + (length[ok] * 100).round().astype(np.int64)
    mid = (segs[ok, :2] + segs[ok, 2:]) / 2
    return key, mid


def _vote(raw_segs: np.ndarray, fin_key: np.ndarray, fin_mid: np.ndarray, scale: float, rot: int,
          rng: np.random.Generator) -> Counter:
    t = Transform(rot, scale, 0.0, 0.0)
    moved = np.column_stack([t.apply(raw_segs[:, :2]), t.apply(raw_segs[:, 2:])])
    key, mid = _signatures(moved)
    votes: Counter = Counter()
    order = np.argsort(fin_key)
    fk, fm = fin_key[order], fin_mid[order]
    uniq, start, count = np.unique(fk, return_index=True, return_counts=True)
    lookup = {k: (s, c) for k, s, c in zip(uniq.tolist(), start.tolist(), count.tolist())}
    raw_order = np.argsort(key)
    rk, rm = key[raw_order], mid[raw_order]
    r_uniq, r_start, r_count = np.unique(rk, return_index=True, return_counts=True)
    for k, s, c in zip(r_uniq.tolist(), r_start.tolist(), r_count.tolist()):
        hit = lookup.get(k)
        if hit is None or c * hit[1] > 400:          # common lengths carry no information
            continue
        a = rm[s:s + c]
        b = fm[hit[0]:hit[0] + hit[1]]
        diff = (b[None, :, :] - a[:, None, :]).reshape(-1, 2)
        for bx, by in np.round(diff / BIN_M).astype(np.int64).tolist():
            votes[(bx, by)] += 1
    return votes


def _peaks(votes: Counter, n: int = 12) -> list[tuple[int, int, int]]:
    """Strongest bins, merging each with its 8 neighbours, without repeating a neighbourhood."""
    out, taken = [], set()
    for (bx, by), _ in votes.most_common(400):
        if (bx, by) in taken:
            continue
        total = sum(votes.get((bx + i, by + j), 0) for i in (-1, 0, 1) for j in (-1, 0, 1))
        out.append((total, bx, by))
        taken.update((bx + i, by + j) for i in range(-3, 4) for j in range(-3, 4))
        if len(out) >= n:
            break
    return sorted(out, reverse=True)


def _region(pts: np.ndarray, cell: float = 2.0) -> tuple[float, float, float, float] | None:
    """Bounding box of the main cluster of matched points (a view), ignoring scattered chance hits
    elsewhere on the sheet: occupancy grid -> connected parts -> the largest ones."""
    if len(pts) < 50:
        return None
    lo = pts.min(axis=0)
    ij = ((pts - lo) / cell).astype(np.int64)
    grid = np.zeros(ij.max(axis=0) + 1, dtype=np.int32)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    occupied = ndimage.binary_dilation(grid >= 3, iterations=1)
    parts, n = ndimage.label(occupied)
    if n == 0:
        return None
    weight = ndimage.sum(grid, parts, index=np.arange(1, n + 1))
    keep = np.flatnonzero(weight >= 0.3 * weight.max()) + 1
    sel = np.isin(parts[ij[:, 0], ij[:, 1]], keep)
    a, b = pts[sel].min(axis=0), pts[sel].max(axis=0)
    return (a[0] - VIEW_MARGIN_M, a[1] - VIEW_MARGIN_M, b[0] + VIEW_MARGIN_M, b[1] + VIEW_MARGIN_M)


def _view_mask(matched: np.ndarray, centres: np.ndarray, cell: float = 1.0, grow: int = 3) -> np.ndarray:
    """Which item centres lie in a floor plan: within `grow` m of geometry the engineer kept.
    Not a bounding box per view (one weak view spanned 420 m and turned neighbouring sheets into
    "deleted inside the plan") and no hole filling (a kept sheet frame is a closed rectangle and
    would fill a whole section sheet). Islands with < 5 % of the biggest part's matches are dropped
    (title blocks and frames that repeat on every sheet), ADR-012."""
    inside = np.zeros(len(centres), dtype=bool)
    if len(matched) < 50:
        return inside
    lo = matched.min(axis=0) - (grow + 1) * cell
    ij = ((matched - lo) / cell).astype(np.int64)
    count = np.zeros(ij.max(axis=0) + grow + 3, dtype=np.int32)
    np.add.at(count, (ij[:, 0], ij[:, 1]), 1)
    grid = ndimage.binary_dilation(count > 0, iterations=grow)
    parts, n = ndimage.label(grid)
    if n == 0:
        return inside
    weight = ndimage.sum(count, parts, index=np.arange(1, n + 1))
    grid = np.isin(parts, np.flatnonzero(weight >= 0.05 * weight.max()) + 1)
    c = np.floor((np.nan_to_num(centres, nan=-1e9) - lo) / cell).astype(np.int64)
    ok = (c[:, 0] >= 0) & (c[:, 0] < grid.shape[0]) & (c[:, 1] >= 0) & (c[:, 1] < grid.shape[1])
    inside[ok] = grid[c[ok, 0], c[ok, 1]]
    return inside


def _shadowed(raw: Scene, pts: np.ndarray, pt_item: np.ndarray, n_pts_item: np.ndarray,
              item_len: np.ndarray, k: int = 6) -> np.ndarray:
    """Items (>= 20 cm) whose sample points almost all lie on another item of a different layer."""
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=k, distance_upper_bound=TOL_M)
    own = pt_item[:, None]
    valid = np.isfinite(dist)
    other = np.where(valid, pt_item[np.minimum(idx, len(pts) - 1)], own)
    other_layer = raw.layer[other]
    covered = ((other != own) & (other_layer != raw.layer[own]) & valid).any(axis=1)
    frac = np.bincount(pt_item, weights=covered, minlength=raw.n) / np.maximum(n_pts_item, 1)
    return (frac >= 0.8) & (item_len >= 0.2)


# ------------------------------------------------------------------------------- main
def align(raw: Scene, final: Scene, log=lambda m: None) -> dict:
    fin_key, fin_mid = _signatures(final.segs)
    fin_pts, _ = _sample(final.segs, FINAL_STEP_M)
    tree = cKDTree(fin_pts)
    raw_pts, raw_pt_seg = _sample(raw.segs, RAW_STEP_M)
    raw_pt_item = raw.seg_item[raw_pt_seg]
    n_pts_item = np.bincount(raw_pt_item, minlength=raw.n)
    rng = np.random.default_rng(0)

    # 1. candidate transforms from votes
    candidates: list[Transform] = []
    for scale in SCALES:
        for rot in range(4):
            votes = _vote(raw.segs, fin_key, fin_mid, scale, rot, rng)
            for total, bx, by in _peaks(votes, 6):
                candidates.append(Transform(rot, scale, bx * BIN_M, by * BIN_M, votes=total))
    candidates.sort(key=lambda t: -t.votes)
    top_votes = candidates[0].votes if candidates else 0
    if top_votes < MIN_VOTES:
        log(say(f"en güçlü oy {top_votes} < {MIN_VOTES}: güvenilir hizalama yok",
                f"strongest vote {top_votes} < {MIN_VOTES}: no reliable alignment"))
        candidates = []
    # A weak peak is chance agreement (dense hatches, repeated details), not a moved view. The
    # engineer moves and turns floors but doesn't rescale them one by one: one scale per pair.
    # A raw drawing shrunk 10x "fits" into any dense area of the final (ADR-003, measured).
    scale = candidates[0].scale if candidates else 1.0
    candidates = [c for c in candidates if c.votes >= VOTE_SHARE * top_votes and c.scale == scale][:20]
    log(say(f"{len(candidates)} aday dönüşüm, en güçlü oy {top_votes}, ölçek {scale:g}",
            f"{len(candidates)} candidate transforms, strongest vote {top_votes}, scale {scale:g}"))

    # 2. verify, strongest first. Every final point belongs to one view only: floors look alike,
    #    so a raw floor also half-matches the final's other floors under a wrong translation.
    accepted: list[tuple[Transform, np.ndarray]] = []
    best_frac = np.zeros(raw.n)
    claimed = np.zeros(len(fin_pts), dtype=bool)
    seg_len = np.hypot(raw.segs[:, 2] - raw.segs[:, 0], raw.segs[:, 3] - raw.segs[:, 1])
    item_len = np.bincount(raw.seg_item, weights=seg_len, minlength=raw.n)
    for cand in candidates:
        moved = cand.apply(raw_pts)
        dist, idx = tree.query(moved, distance_upper_bound=TOL_M)
        hit = np.isfinite(dist)
        fresh = hit.copy()
        fresh[hit] = ~claimed[idx[hit]]
        fresh &= best_frac[raw_pt_item] < 0.8
        frac = np.bincount(raw_pt_item, weights=fresh, minlength=raw.n) / np.maximum(n_pts_item, 1)
        new = frac >= 0.8
        cand.matched_m = float(item_len[new].sum())
        strongest = max((a.matched_m for a, _ in accepted), default=0.0)
        if cand.matched_m < max(MIN_VIEW_MATCH_M, VIEW_SHARE * strongest):
            continue
        cand.region = _region(raw_pts[fresh & new[raw_pt_item]])
        if cand.region is None:
            continue
        claimed[idx[fresh]] = True
        accepted.append((cand, frac))
        best_frac = np.maximum(best_frac, frac)
        log(say(f"görünüş kabul: dönüş {cand.rot * 90}°, ölçek {cand.scale:g}, öteleme ({cand.tx:.2f}, {cand.ty:.2f}) m, "
                f"eşleşen {cand.matched_m:.0f} m çizgi",
                f"view accepted: rotation {cand.rot * 90}°, scale {cand.scale:g}, shift ({cand.tx:.2f}, {cand.ty:.2f}) m, "
                f"{cand.matched_m:.0f} m of lines matched"))

    # 3. labels
    labels = np.full(raw.n, OUT_OF_VIEW, dtype=np.int8)
    bb = raw.bboxes()
    cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
    short_kept = (best_frac >= 0.8) & (item_len <= 20.0)          # frames and axes don't define a plan
    inside = _view_mask(raw_pts[short_kept[raw_pt_item]], np.column_stack([cx, cy]))
    labels[inside & (best_frac <= 0.2)] = DELETE
    labels[inside & (best_frac > 0.2) & (best_frac < 0.8)] = PARTIAL
    labels[best_frac >= 0.8] = KEEP
    # Raw items lying on top of another raw item (area lines over wall faces, copies): the final
    # can't tell which of the two survived, and it doesn't matter visually.
    shadow = _shadowed(raw, raw_pts, raw_pt_item, n_pts_item, item_len)
    labels[shadow & (labels == KEEP)] = PARTIAL
    labels[n_pts_item == 0] = EMPTY

    counts = {LABEL_NAMES[k]: int((labels == k).sum()) for k in LABEL_NAMES}
    counts["shadowed"] = int(shadow.sum())
    return {
        "labels": labels,
        "frac": best_frac.astype(np.float32),
        "shadow": shadow,
        "transforms": [c.as_dict() for c, _ in accepted],
        "counts": counts,
        "ok": bool(accepted),
    }
