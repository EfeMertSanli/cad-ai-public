"""Finishing rules after the model, to match the engineer's base drawing (ADR-016).

Measured against the target the user showed (an engineer's clean base in AutoCAD): the model's
plans were right, but
- room names were deleted; the target keeps exactly one name per room ("Salon", "Mutfak",
  "E.Yatak Odası", "Kat Holu", "Asansör"…), no areas, levels or floor finishes;
- hatches stayed (column fills, wall hatching); the target's columns and walls are outlines only;
- dashed lines stayed inside walls (beams, lintels) and outside rooms (overhangs, projections);
  the target's only dashed lines are the upper flight of the stairs.
These are exact, not learned: hatches in a plan go, except shaft hatches (on a shaft layer or a
small square of at most 60 cm: the engineer's target draws shafts as those), and except on wall /
column layers where the hatch is the element (it becomes its outline, ADR-018); dashed items go
(except on stair layers); annotation layers (dimension, axis, level) go; shaft symbols stay; a room-name text stays only if it lies in a room read from the
cleaned geometry (sections carry room names per floor too, and the view model can take a
section for a plan), one name per room (the biggest text). Colours are made white by the writer
(apply.write_clean).
"""
from __future__ import annotations

import re

import numpy as np

from cadai.cad.scene import Scene
from cadai.corpus.inventory import fold
from cadai.placement.rooms import ROOM_TYPES

DELETE, KEEP, OUTLINE = 0, 1, 2           # same codes as apply.py
ROOM_WORDS = sorted({w for w, _ in ROOM_TYPES} | {
    "ASANSOR", "KAT HOLU", "HOL", "MERDIVEN", "CATI KAPAGI", "SAHANLIK", "GIRIS", "BALKON", "TERAS",
    "BANYO", "MUTFAK", "SALON", "KORIDOR", "DEPO", "KILER", "ANTRE", "LAVABO", "VESTIYER", "OTOPARK",
    "SIGINAK", "KAZAN", "DUKKAN", "OFIS", "CAMASIRHANE", "TESISAT", "SAFT", "BACA"})
NOT_A_NAME = re.compile(r"M2|M²|\^2|[±+]\s*\d|KOT|SERAMIK|PARKE|MERMER|LAMINAT|SAP|BETON|BOYA|KAPLAMA|\d{3,}")
STAIR_LAYER = ("MERDIVEN", "STAIR", "BASAMAK")
SHAFT_LAYER = ("BACA", "SAFT", "SHAFT", "HAVALANDIRMA")
# hatches on these layers are the element itself (a column drawn only as a filled square): keep the outline
STRUCTURE_LAYER = ("DUVAR", "WALL", "KOLON", "COLUMN", "PERDE", "BETON", "PANEL", "TUGLA", "BRIKET", "GAZBETON")
# annotation layers: nothing on them belongs in the base drawing (except room names, kept separately)
ANNOTATION_LAYER = ("OLCU", "DIM", "AKS", "GRID", "AXIS", "KOT", "LEVEL")


def is_room_name(text: str) -> bool:
    """'Salon', 'E.Yatak Odası', 'Kat Holu', 'Yatak Odası 2' yes; '18.03 m²', '±0.00', 'Seramik' no."""
    t = " ".join(fold(text).split())
    if not t or len(t) > 30 or NOT_A_NAME.search(t):
        return False
    return any(w in t for w in ROOM_WORDS)


def _layer_has(scene: Scene, words) -> np.ndarray:
    flags = np.array([any(w in fold(n) for w in words) for n in scene.meta["layers"]])
    return flags[scene.layer]


def finish(scene: Scene, applied: np.ndarray, in_plan: np.ndarray) -> tuple[np.ndarray, dict]:
    out = applied.copy()
    types = np.array(scene.meta["types"])[scene.etype]
    bb = scene.bboxes()
    w, h = bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]
    small_square = np.fmax(w, h) <= 0.6                                  # shaft / duct fills are small, often oblong
    hatch = in_plan & (types == "HATCH") & ~_layer_has(scene, SHAFT_LAYER)
    structural = hatch & _layer_has(scene, STRUCTURE_LAYER)               # first: a column is a small square too
    hatch = hatch & (structural | ~small_square)                         # other small squares: shaft symbols, stay
    dashed = np.zeros(scene.n, dtype=bool) if scene.dashed is None else scene.dashed
    dashed = dashed & ~_layer_has(scene, STAIR_LAYER)
    annotation = in_plan & _layer_has(scene, ANNOTATION_LAYER)
    shafts = shaft_items(scene, in_plan)
    ticks = opening_ticks(scene, out, in_plan)
    stats = {"hatches_deleted": int((hatch & ~structural & (out == KEEP)).sum()),
             "hatches_outlined": int(structural.sum()),
             "dashed_deleted": int((dashed & (out == KEEP)).sum()),
             "annotation_deleted": int((annotation & (out == KEEP)).sum()),
             "shaft_items_kept": int((shafts & (out == DELETE)).sum()),
             "opening_ticks_deleted": int((ticks & (out == KEEP)).sum())}
    out[(hatch & ~structural) | dashed | annotation | ticks] = DELETE
    out[structural] = OUTLINE
    out[shafts] = KEEP
    names = room_names(scene, out, in_plan)
    stats["room_names_kept"] = int(len(names))
    stats["room_names_restored"] = int((out[names] == DELETE).sum())
    out[names] = KEEP
    return out, stats


OPENING_LAYER = ("KAPI", "DOOR", "PENCERE", "WINDOW", "DOGRAMA")
TICK_M = 0.35
WALL_BODY_M = 0.12


def opening_ticks(scene: Scene, applied: np.ndarray, in_plan: np.ndarray) -> np.ndarray:
    """Short marks on door / window layers that stick out of the wall: window sill extensions outside
    the facade, jamb ticks into rooms. Single straight items under 35 cm with most of their length
    farther than 12 cm from every kept wall line. The engineer's base has none (Cevriye BERKOĞLU)."""
    from scipy.spatial import cKDTree

    from cadai.cleanup.align import _sample

    length = scene.lengths()
    nseg = np.bincount(scene.seg_item, minlength=scene.n)
    cand = in_plan & _layer_has(scene, OPENING_LAYER) & (length > 0.02) & (length <= TICK_M) & (nseg == 1)
    walls = (applied == KEEP) & _layer_has(scene, STRUCTURE_LAYER)
    if not cand.any() or not walls.any():
        return np.zeros(scene.n, dtype=bool)
    wall_pts, _ = _sample(scene.segs[walls[scene.seg_item]], 0.02)
    tree = cKDTree(wall_pts)
    seg_mask = cand[scene.seg_item]
    pts, seg = _sample(scene.segs[seg_mask], 0.02)
    owner = scene.seg_item[seg_mask][seg]
    far = tree.query(pts, distance_upper_bound=WALL_BODY_M)[0] > WALL_BODY_M
    share = np.bincount(owner, weights=far, minlength=scene.n) / np.maximum(np.bincount(owner, minlength=scene.n), 1)
    return cand & (share >= 0.5)


def shaft_items(scene: Scene, in_plan: np.ndarray) -> np.ndarray:
    """Items of ventilation shaft / chimney symbols inside a plan: an object of at most 3 m (a block,
    or loose parts on one layer touching each other: architects explode them too) with at least two
    small squares (12-60 cm) and at least two diagonals or circles. The engineer keeps these; the
    model took them for furniture (Cevriye BERKOĞLU)."""
    from cadai.cleanup.review import object_groups

    out = np.zeros(scene.n, dtype=bool)
    types = np.array(scene.meta["types"])[scene.etype]
    bb = scene.bboxes()
    w, h = bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]
    small = in_plan & (np.fmax(w, h) <= 0.8)
    square = scene.closed & (w >= 0.12) & (w <= 0.6) & (np.abs(w - h) <= 0.08)
    seg_angle = np.degrees(np.arctan2(scene.segs[:, 3] - scene.segs[:, 1], scene.segs[:, 2] - scene.segs[:, 0])) % 90
    diag_seg = np.minimum(seg_angle, 90 - seg_angle) > 20
    diagonal = (np.bincount(scene.seg_item, weights=diag_seg, minlength=scene.n) > 0) & ~scene.closed
    mark = diagonal | np.isin(types, ("CIRCLE", "ELLIPSE"))
    # one object = one block reference, or touching loose parts on one layer
    group = np.where(scene.block_top >= 0, scene.top, -1).astype(np.int64)
    loose = object_groups(scene, small & (scene.block_top < 0))
    group[loose >= 0] = loose[loose >= 0] + scene.top.max() + 1
    group[~small & (scene.block_top < 0)] = -1
    order = np.argsort(group, kind="stable")
    g_sorted = group[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    for s0, e0 in zip(starts, np.r_[starts[1:], len(order)]):
        if g_sorted[s0] < 0:
            continue
        idx = order[s0:e0]
        if square[idx].sum() < 2 or mark[idx].sum() < 2:
            continue
        b = bb[idx]
        if max(np.nanmax(b[:, 2]) - np.nanmin(b[:, 0]), np.nanmax(b[:, 3]) - np.nanmin(b[:, 1])) <= 3:
            out[idx] = True
    return out & in_plan


NEAR_M = 1.5          # a name outside closed rooms (balcony, stair hall) stays when this close to kept lines


def room_names(scene: Scene, applied: np.ndarray, in_plan: np.ndarray) -> np.ndarray:
    """Indices of the room-name texts to keep: inside a room of the cleaned plan (each different name
    once per room: an open plan has "Giriş" and "Salon" in one space), or close to kept lines for
    open spaces like balconies (one per name and spot).
    Names far from every kept line are on sheets that were cleaned away (section floor labels)."""
    from scipy.spatial import cKDTree

    from cadai.check.cleanup_checks import plan_region
    from cadai.placement import rooms as rooms_mod

    candidates = np.flatnonzero(in_plan & (scene.text != "")
                                & np.array([is_room_name(t) if t else False for t in scene.text]))
    keep_geo = (applied == KEEP) & (scene.text == "")
    region = plan_region(scene, keep_geo)
    if not len(candidates) or region is None:
        return np.zeros(0, dtype=np.int64)
    bb = scene.bboxes()[candidates]
    centres = np.column_stack([(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2])
    kept_segs = scene.segs[keep_geo[scene.seg_item]]
    near = cKDTree(np.vstack([kept_segs[:, :2], kept_segs[:, 2:], (kept_segs[:, :2] + kept_segs[:, 2:]) / 2]))
    dist, _ = near.query(centres, distance_upper_bound=NEAR_M)
    room = rooms_mod.read_rooms(scene, keep_geo, region).at(centres, snap_m=0.3)
    best: dict[tuple, int] = {}
    for i, r, d, c in zip(candidates, room, dist, centres):
        if r:
            key = ("room", int(r), " ".join(fold(scene.text[i]).split()))     # each different name once per room
        elif np.isfinite(d):
            key = ("open", " ".join(fold(scene.text[i]).split()), round(c[0]), round(c[1]))
        else:
            continue
        if key not in best or scene.text_h[i] > scene.text_h[best[key]]:
            best[key] = int(i)
    return np.array(sorted(best.values()), dtype=np.int64)
