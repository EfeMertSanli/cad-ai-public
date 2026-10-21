"""Scene: a drawing exploded into leaf items with flattened geometry, in metres.

Every modelspace entity is walked; block references are exploded recursively (virtual entities,
nothing is written), so each leaf knows the top-level entity it came from (`top`) and its path
`<handle>/<i>/<j>` for writing a clean copy later. Children on layer "0" take their parent's
layer (AutoCAD by-block convention). Dimensions and leaders stay one leaf each.

Geometry is a flat segment list: curves are flattened with a 5 mm chord error; text becomes its
outline box. Columnar arrays, saved as one .npz per drawing, so 500k leaves load in a second.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ezdxf import path as ezpath
from ezdxf import recover

from cadai.cad.units import infer_unit

ATOMIC = {"DIMENSION", "ARC_DIMENSION", "LARGE_RADIAL_DIMENSION", "LEADER", "MULTILEADER", "ACAD_TABLE"}
TEXT = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}
MAX_DEPTH = 8
CHORD_M = 0.005


@dataclass
class Scene:
    """One drawing. Item arrays have one row per leaf; segments point to their item."""
    meta: dict
    path: np.ndarray        # str  "<handle>/<i>/..."
    top: np.ndarray         # int  index of the top-level entity (modelspace order)
    layer: np.ndarray       # int  -> meta["layers"]
    etype: np.ndarray       # int  -> meta["types"]
    block: np.ndarray       # int  innermost block -> meta["blocks"], -1 none
    block_top: np.ndarray   # int  outermost block, -1 none
    depth: np.ndarray       # int
    color: np.ndarray       # int  ACI, 256 = by layer
    text: np.ndarray        # str
    text_h: np.ndarray      # float m
    radius: np.ndarray      # float m, 0 if not an arc/circle
    sweep: np.ndarray       # float degrees
    closed: np.ndarray      # bool
    filled: np.ndarray      # bool
    seg_item: np.ndarray    # int  (M,)
    segs: np.ndarray        # float (M, 4) x1 y1 x2 y2 in metres
    dashed: np.ndarray | None = None   # bool: drawn with a dashed linetype (None in scenes cached before it)

    @property
    def n(self) -> int:
        return len(self.top)

    def save(self, target: Path) -> None:
        import json
        tmp = target.with_suffix(".part.npz")
        arrays = {k: getattr(self, k) for k in self.__dataclass_fields__
                  if k != "meta" and getattr(self, k) is not None}
        np.savez_compressed(tmp, meta=np.array(json.dumps(self.meta, ensure_ascii=False)), **arrays)
        tmp.replace(target)

    @classmethod
    def load(cls, source: Path) -> "Scene":
        import json
        with np.load(source, allow_pickle=False) as z:
            kw = {k: z[k] for k in cls.__dataclass_fields__ if k != "meta" and k in z.files}
            return cls(meta=json.loads(str(z["meta"])), **kw)

    def bboxes(self) -> np.ndarray:
        """(n, 4) min x, min y, max x, max y per item; NaN for items without geometry."""
        out = np.full((self.n, 4), np.nan)
        if len(self.segs):
            xs = np.minimum(self.segs[:, 0], self.segs[:, 2]), np.maximum(self.segs[:, 0], self.segs[:, 2])
            ys = np.minimum(self.segs[:, 1], self.segs[:, 3]), np.maximum(self.segs[:, 1], self.segs[:, 3])
            for col, vals, fn in ((0, xs[0], np.fmin), (1, ys[0], np.fmin), (2, xs[1], np.fmax), (3, ys[1], np.fmax)):
                acc = np.full(self.n, np.nan)
                fn.at(acc, self.seg_item, vals)
                out[:, col] = acc
        return out

    def lengths(self) -> np.ndarray:
        seg_len = np.hypot(self.segs[:, 2] - self.segs[:, 0], self.segs[:, 3] - self.segs[:, 1])
        return np.bincount(self.seg_item, weights=seg_len, minlength=self.n)


# ------------------------------------------------------------------------------- building
class _Leaf:
    __slots__ = ("entity", "path", "top", "layer", "block", "block_top", "depth")

    def __init__(self, entity, path, top, layer, block, block_top, depth):
        self.entity, self.path, self.top, self.layer = entity, path, top, layer
        self.block, self.block_top, self.depth = block, block_top, depth


def _walk(entity, path, top, parent_layer, block, block_top, depth, out):
    layer = entity.dxf.get("layer", "0")
    if parent_layer is not None and layer == "0":
        layer = parent_layer
    t = entity.dxftype()
    if t == "INSERT" and depth < MAX_DEPTH:
        name = entity.dxf.name
        try:
            kids = list(entity.virtual_entities())
        except Exception:  # noqa: BLE001 - a broken block reference contributes nothing
            kids = []
        for i, kid in enumerate(kids):
            _walk(kid, f"{path}/{i}", top, layer, name, block_top or name, depth + 1, out)
        for k, att in enumerate(entity.attribs):
            _walk(att, f"{path}/a{k}", top, layer, name, block_top or name, depth + 1, out)
        return
    out.append(_Leaf(entity, path, top, layer, block, block_top, depth))


def _text_of(e) -> tuple[str, float, float, tuple[float, float]]:
    if e.dxftype() == "MTEXT":
        ins = e.dxf.insert
        return e.plain_text(split=False), float(e.dxf.char_height), float(e.get_rotation()), (ins.x, ins.y)
    ins = e.dxf.insert
    if e.dxf.get("halign", 0) or e.dxf.get("valign", 0):
        ins = e.dxf.get("align_point", ins)
    return e.dxf.text, float(e.dxf.height), float(e.dxf.get("rotation", 0.0)), (ins.x, ins.y)


def _text_box(x, y, h, rot, text):
    w = max((len(s) for s in text.split("\n")), default=1) * h * 0.6
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    pts = [(x + px * c - py * s, y + px * s + py * c) for px, py in ((0, 0), (w, 0), (w, h), (0, h), (0, 0))]
    return [np.array(pts)]


def _polylines(e, chord) -> tuple[list[np.ndarray], bool]:
    t = e.dxftype()
    if t == "POINT":
        p = e.dxf.location
        return [np.array([(p.x, p.y), (p.x, p.y)])], False
    if t == "HATCH":
        return [np.array([(v.x, v.y) for v in p.flattening(chord)]) for p in ezpath.from_hatch(e)], True
    if t in ATOMIC:
        out = []
        for v in e.virtual_entities():
            try:
                if v.dxftype() in TEXT:
                    txt, h, rot, (x, y) = _text_of(v)
                    out += _text_box(x, y, h, rot, txt)
                else:
                    out += _polylines(v, chord)[0]
            except Exception:  # noqa: BLE001 - odd sub-entities of a dimension are skipped
                continue
        return out, False
    p = ezpath.make_path(e)
    return [np.array([(v.x, v.y) for v in p.flattening(chord)])], bool(p.is_closed)


SOLID_LINETYPES = {"", "CONTINUOUS", "BYBLOCK"}      # BYBLOCK: the block reference's linetype, not tracked


def _is_dashed(entity, layer: str, layer_linetype: dict) -> bool:
    lt = str(entity.dxf.get("linetype", "BYLAYER") or "BYLAYER").upper()
    if lt == "BYLAYER":
        lt = str(layer_linetype.get(layer.upper(), "CONTINUOUS")).upper()
    return lt not in SOLID_LINETYPES and lt != "BYLAYER"


def build(dxf_file: Path, unit: dict | None = None) -> Scene:
    """`unit`: force a unit (infer_unit result) instead of inferring it, e.g. for a cleaned copy
    of a drawing whose door arcs and texts are gone."""
    t0 = time.time()
    doc, auditor = recover.readfile(str(dxf_file))
    leaves: list[_Leaf] = []
    tops = []
    for i, e in enumerate(doc.modelspace()):
        tops.append(e.dxf.handle)
        try:
            _walk(e, e.dxf.handle, i, None, None, None, 0, leaves)
        except Exception:  # noqa: BLE001 - one bad entity must not lose the drawing
            continue
    t_read = time.time() - t0

    # Unit from evidence (door arcs, text heights, header).
    arcs, heights = [], []
    for lf in leaves:
        e, t = lf.entity, lf.entity.dxftype()
        if t == "ARC":
            sweep = (e.dxf.end_angle - e.dxf.start_angle) % 360
            arcs.append((float(e.dxf.radius), sweep))
        elif t in ("TEXT", "MTEXT"):
            heights.append(float(e.dxf.get("char_height" if t == "MTEXT" else "height", 0) or 0))
    unit = unit or infer_unit(arcs, heights[:5000], int(doc.header.get("$INSUNITS", 0) or 0))
    scale = unit["scale"]
    chord = CHORD_M / scale

    layers: dict[str, int] = {}
    types: dict[str, int] = {}
    blocks: dict[str, int] = {}
    cols = {k: [] for k in ("path", "top", "layer", "etype", "block", "block_top", "depth", "color",
                            "text", "text_h", "radius", "sweep", "closed", "filled", "dashed")}
    layer_linetype = {layer.dxf.name.upper(): layer.dxf.get("linetype", "CONTINUOUS") for layer in doc.layers}
    seg_item, segs = [], []
    failed = 0
    for lf in leaves:
        e, t = lf.entity, lf.entity.dxftype()
        text, text_h, radius, sweep, closed = "", 0.0, 0.0, 0.0, False
        try:
            if t in TEXT:
                text, h, rot, (x, y) = _text_of(e)
                text_h = h * scale
                polys = _text_box(x, y, h, rot, text)
            else:
                polys, closed = _polylines(e, chord)
                if t in ("ARC", "CIRCLE"):
                    radius = float(e.dxf.radius) * scale
                    sweep = 360.0 if t == "CIRCLE" else (e.dxf.end_angle - e.dxf.start_angle) % 360
                if t in ATOMIC:
                    try:
                        text = e.dxf.get("text", "") or ""
                    except Exception:  # noqa: BLE001
                        text = ""
        except Exception:  # noqa: BLE001 - unsupported geometry keeps an item without segments
            polys, failed = [], failed + 1
        idx = len(cols["top"])
        for poly in polys:
            if len(poly) == 1:
                poly = np.vstack([poly, poly])
            if len(poly) >= 2:
                p = poly * scale
                segs.append(np.hstack([p[:-1], p[1:]]))
                seg_item.append(np.full(len(p) - 1, idx, dtype=np.int32))
        cols["path"].append(lf.path)
        cols["top"].append(lf.top)
        cols["layer"].append(layers.setdefault(lf.layer, len(layers)))
        cols["etype"].append(types.setdefault(t, len(types)))
        cols["block"].append(blocks.setdefault(lf.block, len(blocks)) if lf.block else -1)
        cols["block_top"].append(blocks.setdefault(lf.block_top, len(blocks)) if lf.block_top else -1)
        cols["depth"].append(lf.depth)
        cols["color"].append(int(e.dxf.get("color", 256)))
        cols["text"].append(text.replace("\x00", ""))
        cols["text_h"].append(text_h)
        cols["radius"].append(radius)
        cols["sweep"].append(sweep)
        cols["closed"].append(closed)
        cols["filled"].append(t in ("SOLID", "TRACE") or (t == "HATCH" and bool(e.dxf.get("solid_fill", 0))))
        cols["dashed"].append(_is_dashed(e, lf.layer, layer_linetype))

    all_segs = np.vstack(segs) if segs else np.zeros((0, 4))
    meta = {
        "source": str(dxf_file),
        "unit": unit,
        "insunits": int(doc.header.get("$INSUNITS", 0) or 0),
        "layers": list(layers),
        "types": list(types),
        "blocks": list(blocks),
        "top_handles": tops,
        "audit_fixes": len(auditor.fixes),
        "failed_items": failed,
        "seconds_read": round(t_read, 1),
        "seconds_total": round(time.time() - t0, 1),
    }
    return Scene(
        meta=meta,
        path=np.array(cols["path"], dtype=str),
        top=np.array(cols["top"], dtype=np.int32),
        layer=np.array(cols["layer"], dtype=np.int32),
        etype=np.array(cols["etype"], dtype=np.int16),
        block=np.array(cols["block"], dtype=np.int32),
        block_top=np.array(cols["block_top"], dtype=np.int32),
        depth=np.array(cols["depth"], dtype=np.int8),
        color=np.array(cols["color"], dtype=np.int16),
        text=np.array(cols["text"], dtype=str),
        text_h=np.array(cols["text_h"], dtype=np.float32),
        radius=np.array(cols["radius"], dtype=np.float32),
        sweep=np.array(cols["sweep"], dtype=np.float32),
        closed=np.array(cols["closed"], dtype=bool),
        filled=np.array(cols["filled"], dtype=bool),
        dashed=np.array(cols["dashed"], dtype=bool),
        seg_item=np.concatenate(seg_item) if seg_item else np.zeros(0, dtype=np.int32),
        segs=all_segs,
    )
