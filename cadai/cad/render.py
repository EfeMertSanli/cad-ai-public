"""PNG previews of a scene, optionally coloured per item (labels, predictions)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cadai.cad.scene import Scene

# label/prediction value -> colour, drawn in this order (later on top)
LABEL_COLOURS = {
    2: (190, 190, 210),     # out of view: pale
    -1: (200, 200, 200),
    0: (220, 40, 40),       # delete: red
    3: (240, 160, 0),       # partial / uncertain: orange
    1: (60, 60, 60),        # keep: dark grey
}


def render(scene: Scene, out: Path, colour_by: np.ndarray | None = None, bbox=None,
           width: int = 2400, colours: dict | None = None, background=(255, 255, 255),
           thick: dict | None = None) -> Path:
    """Colours are drawn in dict order; `thick` maps a key to a line width (errors)."""
    segs = scene.segs
    if bbox is None:
        finite = segs[np.isfinite(segs).all(axis=1)]
        xs = np.concatenate([finite[:, 0], finite[:, 2]])
        ys = np.concatenate([finite[:, 1], finite[:, 3]])
        bbox = (np.percentile(xs, 0.2), np.percentile(ys, 0.2), np.percentile(xs, 99.8), np.percentile(ys, 99.8))
    x0, y0, x1, y1 = bbox
    scale = width / max(x1 - x0, 1e-9)
    height = max(1, int((y1 - y0) * scale))
    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)
    colours = colours or LABEL_COLOURS
    keys = colour_by[scene.seg_item] if colour_by is not None else np.ones(len(segs), dtype=int)
    px = np.column_stack([(segs[:, 0] - x0) * scale, (y1 - segs[:, 1]) * scale,
                          (segs[:, 2] - x0) * scale, (y1 - segs[:, 3]) * scale])
    inside = ((np.maximum(px[:, 0], px[:, 2]) >= 0) & (np.minimum(px[:, 0], px[:, 2]) <= width)
              & (np.maximum(px[:, 1], px[:, 3]) >= 0) & (np.minimum(px[:, 1], px[:, 3]) <= height))
    thick = thick or {}
    for key, colour in colours.items():
        sel = inside & (keys == key)
        for a, b, c, d in px[sel].tolist():
            draw.line((a, b, c, d), fill=colour, width=thick.get(key, 1))
    if isinstance(out, Path):
        out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG")                      # `out` may be a buffer (Düzelt page views)
    return out
