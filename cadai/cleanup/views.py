"""View model: is an item part of a floor plan? (plan-cleanup §4.1, ADR-006)

A raw drawing holds many views side by side: floor plans, sections, elevations, site plan, roof,
details, title blocks, tables. The cleanup keeps floor plans only, and a wrong view choice deletes
thousands of items at once, so this is decided before the per-item cleanup.

Sheets usually touch each other (frames, title blocks), so cutting the drawing into clusters
fails. Instead every item gets neighbourhood features: on a 2 m grid, sums of wall-face length,
door-swing arcs, room names, dimension texts, hatches, section/site/roof layers… box-filtered
over 6 m and 20 m; plus the distance to the nearest title word of each view kind (KESİT,
GÖRÜNÜŞ, VAZİYET, ÇATI, KAT PLANI). A LightGBM model learns "inside a floor plan" from the
alignment labels (not out-of-view).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from cadai.cad.scene import Scene
from cadai.corpus.inventory import fold

VERSION = 1
CELL_M = 2.0
WINDOWS_CELLS = (3, 10)                     # 6 m and 20 m
FAR_M = 200.0

VIEW_WORDS = {
    "plan": ["KAT PLANI", "ZEMIN KAT", "NORMAL KAT", "BODRUM KAT", "ASMA KAT", "BIRINCI KAT", "IKINCI KAT", "KAT PLAN"],
    "section": ["KESIT", "SECTION"],
    "elevation": ["GORUNUS", "CEPHE", "ELEVATION"],
    "site": ["VAZIYET", "YERLESIM", "SITE PLAN"],
    "roof": ["CATI PLANI", "CATI ARASI"],
    "detail": ["DETAY", "DETAIL"],
    "table": ["MAHAL LISTESI", "ALAN HESAB", "EMSAL", "TAKS", "KAKS", "TABLO"],
}
VIEW_KEYS = list(VIEW_WORDS)

# item feature -> per-cell sum (weighted by item length where it is a share)
DENSITY = ["wall_twin", "door_arc", "text_room", "text_dim", "text_level", "is_text", "filled",
           "layer_section", "layer_site", "layer_roof", "layer_title", "layer_wall", "layer_furniture",
           "layer_sanitary", "axis_share", "one"]


def item_features(scene: Scene, X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Neighbourhood features per item. X/names: the cleanup item features (features.py)."""
    col = {n: i for i, n in enumerate(names)}
    bb = scene.bboxes()
    centres = np.column_stack([np.nan_to_num((bb[:, 0] + bb[:, 2]) / 2), np.nan_to_num((bb[:, 1] + bb[:, 3]) / 2)])
    has_geo = np.isfinite(bb[:, 0])
    length = X[:, col["length"]]
    w = np.clip(length, 0.05, 20.0)                                      # long lines count more, capped
    base = {
        "door_arc": (np.abs(X[:, col["sweep"]] - 90) < 10) & (X[:, col["radius"]] > 0.6) & (X[:, col["radius"]] < 1.2),
        "is_text": X[:, col["text_len"]] > 0,
        "one": np.ones(scene.n),
    }
    values = {k: (base[k] if k in base else X[:, col[k]]).astype(np.float64) for k in DENSITY}
    weighted = {"wall_twin", "layer_section", "layer_site", "layer_roof", "layer_title", "layer_wall",
                "layer_furniture", "layer_sanitary", "axis_share"}

    # robust grid extent: ignore stray items far from the bulk of the drawing
    ok = has_geo & np.all(np.abs(centres - np.median(centres[has_geo], axis=0)) < 5000, axis=1) if has_geo.any() else has_geo
    lo = centres[ok].min(axis=0) if ok.any() else np.zeros(2)
    ij = np.clip(((centres - lo) / CELL_M).astype(np.int64), 0, None)
    shape = ij[ok].max(axis=0) + 1 if ok.any() else np.array([1, 1])
    ij = np.minimum(ij, shape - 1)

    cols, cnames = [], []
    for name in DENSITY:
        grid = np.zeros(shape)
        v = values[name] * (w if name in weighted else 1.0)
        np.add.at(grid, (ij[ok, 0], ij[ok, 1]), v[ok])
        for size in WINDOWS_CELLS:
            sm = ndimage.uniform_filter(grid, size=2 * size + 1, mode="constant") * (2 * size + 1) ** 2
            cols.append(np.log1p(np.maximum(sm[ij[:, 0], ij[:, 1]], 0)))
            cnames.append(f"nb{size * CELL_M:g}_{name}")

    # distance to the nearest title word of each view kind
    texts = np.array([fold(t) if t else "" for t in scene.text])
    tmask = np.flatnonzero(texts != "")
    for k in VIEW_KEYS:
        hit = [i for i in tmask if any(wd in texts[i] for wd in VIEW_WORDS[k])]
        if hit:
            dist, _ = cKDTree(centres[hit]).query(centres, distance_upper_bound=FAR_M)
            cols.append(np.minimum(dist, FAR_M))
        else:
            cols.append(np.full(scene.n, FAR_M))
        cnames.append(f"dist_title_{k}")

    # where the item sits in the whole drawing
    ext = np.maximum(shape * CELL_M, 1.0)
    cols += [(centres[:, 0] - lo[0]) / ext[0], (centres[:, 1] - lo[1]) / ext[1]]
    cnames += ["rel_x", "rel_y"]
    return np.column_stack(cols).astype(np.float32), cnames
