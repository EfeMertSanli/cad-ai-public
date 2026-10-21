"""Drawing regions of a sheet: which views are there, for choosing the floor plans (ADR-021).

A raw drawing holds many views side by side (floor plans, sections, elevations, site plan, roof,
details, tables). The statistical view model (views.py) judges item by item; it is right on 60 %
of whole regions in the archive. Choosing whole regions from a picture and their titles is where an
LLM is strongest (82 % in the same test), so the sheet is cut into regions first:

- candidates: non-text items the view model gives p >= 0.3, that are small (frames, axis lines and
  long outlines would glue neighbouring sheets together) and not on dimension / axis / title layers
  (dimension chains reach across the gaps between views);
- 1 m cells with at least 1.5 m of candidate line, grown by one cell, closed, labelled; regions
  under 30 m of line are dropped;
- every item (texts and dimensions too) belongs to the region of the cell its longest segment lies in;
- titles: the largest texts inside a region or just below it ("B-B KESİTİ", "ZEMİN KAT PLANI 1/50").
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from cadai.cad.scene import Scene
from cadai.corpus.inventory import fold

CELL_M = 1.0
DENSE_M = 1.5               # metres of candidate line per cell
MIN_REGION_M = 30.0
CANDIDATE_P = 0.3
SMALL_M = 6.0
GLUE_FEATURES = ("layer_dim", "layer_grid", "layer_text", "layer_title", "block_dim", "block_grid", "block_title")
GLUE_TYPES = ("DIMENSION", "HATCH", "SOLID")


def find(scene: Scene, p_view: np.ndarray, X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[dict]]:
    """Region index per item (-1 = none) and the regions [{id, box, m, p_view}], biggest first."""
    col = {n: i for i, n in enumerate(names)}
    bb = scene.bboxes()
    diag = np.nan_to_num(np.hypot(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]), nan=1e9)
    cand = (diag < SMALL_M) & (scene.text == "") & (p_view >= CANDIDATE_P)
    for f in GLUE_FEATURES:
        cand &= X[:, col[f]] <= 0
    cand &= ~np.isin(np.array(scene.meta["types"])[scene.etype], GLUE_TYPES)
    finite = np.isfinite(scene.segs).all(1)
    s = scene.segs[finite & cand[scene.seg_item]]
    out_region = np.full(scene.n, -1)
    if len(s) == 0:
        return out_region, []
    x0, y0 = np.percentile(np.r_[s[:, 0], s[:, 2]], 0.1) - 5, np.percentile(np.r_[s[:, 1], s[:, 3]], 0.1) - 5
    x1, y1 = np.percentile(np.r_[s[:, 0], s[:, 2]], 99.9) + 5, np.percentile(np.r_[s[:, 1], s[:, 3]], 99.9) + 5
    cell = CELL_M
    while (x1 - x0) / cell * (y1 - y0) / cell > 4e6:                 # huge sheets: coarser cells
        cell *= 1.5
    W, H = int((x1 - x0) / cell) + 1, int((y1 - y0) / cell) + 1
    mid = (s[:, :2] + s[:, 2:]) / 2
    ix = np.clip(((mid[:, 0] - x0) / cell).astype(int), 0, W - 1)
    iy = np.clip(((mid[:, 1] - y0) / cell).astype(int), 0, H - 1)
    density = np.zeros((H, W))
    np.add.at(density, (iy, ix), np.hypot(*(s[:, 2:] - s[:, :2]).T))
    mask = ndimage.binary_closing(ndimage.binary_dilation(density > DENSE_M * cell, iterations=1), iterations=1)
    lab, n = ndimage.label(mask)
    # every item -> the region of its longest segment's cell
    sa = scene.segs[finite]
    ma = (sa[:, :2] + sa[:, 2:]) / 2
    jx, jy = ((ma[:, 0] - x0) / cell).astype(int), ((ma[:, 1] - y0) / cell).astype(int)
    inside = (jx >= 0) & (jx < W) & (jy >= 0) & (jy < H)
    seg_region = np.full(len(sa), -1)
    seg_region[inside] = lab[jy[inside], jx[inside]] - 1
    order = np.argsort(np.hypot(*(sa[:, 2:] - sa[:, :2]).T))      # longest last wins
    out_region[scene.seg_item[finite][order]] = seg_region[order]
    length = scene.lengths()
    regions = []
    slices = ndimage.find_objects(lab)
    for k in range(n):
        m = out_region == k
        m_len = float(length[m].sum())
        if m_len < MIN_REGION_M:
            out_region[m] = -1
            continue
        sy, sx = slices[k]
        regions.append({"id": k, "box": [x0 + sx.start * cell, y0 + sy.start * cell, x0 + sx.stop * cell, y0 + sy.stop * cell],
                        "m": round(m_len), "p_view": round(float(np.average(p_view[m], weights=length[m] + 1e-3)), 2)})
    regions.sort(key=lambda r: -r["m"])
    for r in regions:
        r["titles"] = titles(scene, r)
    return out_region, regions


def titles(scene: Scene, region: dict, limit: int = 6) -> list[str]:
    """The largest texts inside the region or within a quarter of its height below it."""
    x0, y0, x1, y1 = region["box"]
    bb = scene.bboxes()
    cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
    near = (scene.text != "") & (cx >= x0) & (cx <= x1) & (cy >= y0 - 0.25 * (y1 - y0) - 3) & (cy <= y1)
    idx = np.flatnonzero(near)
    seen, out = set(), []
    for i in idx[np.argsort(-scene.text_h[idx])]:
        t = " ".join(fold(scene.text[i]).split())[:60]
        if len(t) < 3 or t in seen or re.fullmatch(r"[\d\s.,+\-±%/]+", t):
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def overview(scene: Scene, item_region: np.ndarray, regions: list[dict], target: Path,
             chosen: set[int] | None = None, width: int = 2000) -> Path:
    """The whole sheet, region boxes with their numbers (list order). With `chosen`: kept regions
    green, the others red."""
    finite = np.isfinite(scene.segs).all(1)
    s = scene.segs[finite]
    x0, y0 = np.percentile(np.r_[s[:, 0], s[:, 2]], 0.1), np.percentile(np.r_[s[:, 1], s[:, 3]], 0.1)
    x1, y1 = np.percentile(np.r_[s[:, 0], s[:, 2]], 99.9), np.percentile(np.r_[s[:, 1], s[:, 3]], 99.9)
    sc = width / max(x1 - x0, 1e-6)
    H = int(min(max((y1 - y0) * sc, 200), 2000))
    sc = min(sc, H / max(y1 - y0, 1e-6))
    img = Image.new("RGB", (width, H), "white")
    d = ImageDraw.Draw(img)
    in_region = item_region[scene.seg_item[finite]] >= 0
    px = np.column_stack([(s[:, 0] - x0) * sc, H - (s[:, 1] - y0) * sc, (s[:, 2] - x0) * sc, H - (s[:, 3] - y0) * sc])
    step = max(1, len(px) // 400_000)
    for (a, b, c, e), r in zip(px[::step].tolist(), in_region[::step]):
        d.line((a, b, c, e), fill=(30, 30, 30) if r else (160, 160, 160))
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for n, reg in enumerate(regions):
        bx0, by0, bx1, by1 = reg["box"]
        colour = (0, 150, 60) if chosen is not None and reg["id"] in chosen else (220, 0, 0)
        a, b, c, e = (bx0 - x0) * sc, H - (by1 - y0) * sc, (bx1 - x0) * sc, H - (by0 - y0) * sc
        d.rectangle((a, b, c, e), outline=colour, width=3)
        d.rectangle((a, b, a + 52, b + 34), fill=colour)
        d.text((a + 6, b + 2), str(n), fill="white", font=font)
    img.save(target)
    return target
