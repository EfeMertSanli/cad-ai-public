"""Features per scene item for the cleanup classifier (docs/plan-cleanup.md §3).

Every item becomes one row of numbers. Groups:
- entity: type, size, length, closed/filled, arc radius and sweep, depth in blocks, colour;
- names: keywords in the layer name and the block name (Turkish folded, English too);
- layer statistics: how the whole layer looks in this drawing (size, text share, straightness),
  so a layer with an unknown name can still be recognised by its content;
- text: numbers, dimension values, levels, areas, room names, title-block words;
- context: wall twin (a parallel line at wall thickness), distance to the nearest wall-like line,
  neighbour counts, exact copies.
Nothing here looks at labels: the same function runs on a new customer's drawing.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.spatial import cKDTree

from cadai.cad.scene import Scene
from cadai.corpus.inventory import fold

VERSION = 1

ETYPES = ["LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE", "HATCH", "TEXT", "MTEXT",
          "ATTRIB", "ATTDEF", "DIMENSION", "SOLID", "POINT", "LEADER", "MULTILEADER", "INSERT", "ACAD_TABLE"]

NAME_KEYWORDS = {
    "wall": ["DUVAR", "WALL", "TUGLA", "BRIKET", "GAZBETON", "BIMS"],
    "column": ["KOLON", "COLUMN", "PERDE", "BETON", "BETONARME"],
    "door": ["KAPI", "DOOR"],
    "window": ["PENCERE", "WINDOW", "DOGRAMA", "CAM", "GLAZ"],
    "stair": ["MERDIVEN", "STAIR", "BASAMAK", "RAMPA"],
    "rail": ["KORKULUK", "RAIL", "PARAPET"],
    "plaster": ["SIVA", "PLASTER", "KAPLAMA"],
    "dim": ["OLCU", "DIM", "OLCULENDIRME", "KOTLAMA"],
    "grid": ["AKS", "GRID", "AXIS"],
    "hatch": ["TARAMA", "HATCH", "TRM", "DOLGU"],
    "furniture": ["MOBILYA", "FURN", "ESYA", "DEFRISE", "DEKOR", "KUTUPHANE", "LIB"],
    "sanitary": ["VITRIFIYE", "SIHHI", "WC", "LAVABO", "KLOZET", "EVYE", "DUS", "BANYO", "TESISAT", "SU", "PIS"],
    "text": ["YAZI", "TEXT", "NOT", "NOTE", "ETIKET", "LABEL"],
    "level": ["KOT", "LEVEL", "SEVIYE"],
    "section": ["KESIT", "SECTION", "GORUNUS", "ELEV", "CEPHE"],
    "title": ["ANTET", "TITLE", "PAFTA", "CERCEVE", "FRAME", "LOGO"],
    "room": ["MAHAL", "ROOM", "ODA"],
    "area": ["HESAP", "ALAN", "AREA", "EMSAL", "TAKS", "KAKS"],
    "roof": ["CATI", "ROOF", "SACAK", "MAHYA"],
    "site": ["VAZIYET", "SITE", "PARSEL", "IMAR", "YOL", "AGAC", "TREE", "PEYZAJ", "OTO", "ARAC"],
    "electric": ["ELEKTRIK", "ELEC", "PRIZ", "LINYE", "ZAYIF", "AYDINLATMA"],
    "line": ["CIZGI", "LINE", "HAT", "YARDIMCI", "CONSTR"],
    "defpoints": ["DEFPOINTS"],
    "slab": ["DOSEME", "SLAB", "KENAR", "SUBASMAN"],
}
KW_KEYS = list(NAME_KEYWORDS)

ROOM_WORDS = ["SALON", "YATAK", "MUTFAK", "BANYO", "WC", "HOL", "ANTRE", "BALKON", "ODA", "KILER", "DEPO",
              "OTURMA", "CALISMA", "COCUK", "EBEVEYN", "GIYINME", "TERAS", "KORIDOR", "LOBI", "DUKKAN",
              "OFIS", "ISLAK", "CAMASIR", "SIGINAK", "KAZAN", "MERDIVEN", "ASANSOR", "SAHANLIK", "YEMEK"]
TITLE_WORDS = ["OLCEK", "PAFTA", "MIMAR", "PROJE", "TARIH", "CIZEN", "KONTROL", "ADA", "PARSEL", "MAL SAHIBI",
               "PLANI", "KESIT", "GORUNUS", "CEPHE", "VAZIYET", "KAT PLANI", "BODRUM", "ZEMIN", "NORMAL"]

RE_NUMBER = re.compile(r"^[\s\d.,+\-±%]+$")
RE_DIM = re.compile(r"^\d{2,4}([.,]\d{1,2})?$")
RE_LEVEL = re.compile(r"[±+\-]\s*\d+[.,]\d{2}")
RE_AREA = re.compile(r"M2|M²|M\^2")
RE_AXIS = re.compile(r"^[A-Z]$|^\d{1,2}$|^[A-Z]'?$")


def _keyword_matrix(names: list[str]) -> np.ndarray:
    folded = [fold(n) for n in names]
    out = np.zeros((len(names), len(KW_KEYS)), dtype=np.float32)
    for i, n in enumerate(folded):
        for j, k in enumerate(KW_KEYS):
            if any(w in n for w in NAME_KEYWORDS[k]):
                out[i, j] = 1
    return out


def _text_flags(texts: np.ndarray) -> np.ndarray:
    out = np.zeros((len(texts), 9), dtype=np.float32)
    for i, t in enumerate(texts):
        if not t:
            continue
        f = fold(t).strip()
        out[i, 0] = len(f)
        out[i, 1] = bool(RE_NUMBER.match(f))
        out[i, 2] = bool(RE_DIM.match(f))
        out[i, 3] = bool(RE_LEVEL.search(f)) or "KOT" in f
        out[i, 4] = bool(RE_AREA.search(f))
        out[i, 5] = any(w in f for w in ROOM_WORDS)
        out[i, 6] = any(w in f for w in TITLE_WORDS)
        out[i, 7] = bool(RE_AXIS.match(f))
        out[i, 8] = f.count("\n") + 1
    return out


def _wall_twins(scene: Scene, length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per item: share of its straight length that has a parallel partner 5-50 cm away (a wall face);
    and the sample points that do (for distance-to-wall)."""
    segs = scene.segs
    d = segs[:, 2:] - segs[:, :2]
    seg_len = np.hypot(d[:, 0], d[:, 1])
    ok = seg_len >= 0.3
    idx = np.flatnonzero(ok)
    if not len(idx):
        return np.zeros(scene.n, dtype=np.float32), np.zeros((0, 2))
    unit = d[idx] / seg_len[idx, None]
    n = np.clip((seg_len[idx] / 0.25).astype(int), 1, 200) + 1
    rep = np.repeat(np.arange(len(idx)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / np.repeat(n - 1, n)
    pts = segs[idx[rep], :2] + d[idx[rep]] * t[:, None]
    dirs = unit[rep]
    owner = scene.seg_item[idx[rep]]
    tree = cKDTree(pts)
    pairs = tree.query_pairs(0.5, output_type="ndarray")
    has = np.zeros(len(pts), dtype=bool)
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        parallel = np.abs((dirs[a] * dirs[b]).sum(axis=1)) > 0.999
        diff = pts[b] - pts[a]
        perp = np.abs(diff[:, 0] * dirs[a, 1] - diff[:, 1] * dirs[a, 0])
        good = parallel & (perp >= 0.05) & (perp <= 0.5) & (owner[a] != owner[b])
        has[a[good]] = True
        has[b[good]] = True
    share = np.bincount(owner, weights=has, minlength=scene.n) / np.maximum(np.bincount(owner, minlength=scene.n), 1)
    return share.astype(np.float32), pts[has]


def _copies(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """Number of other items with identical geometry (1 mm), and how many of them share the layer."""
    if not len(scene.segs):
        return np.zeros(scene.n), np.zeros(scene.n)
    q = np.round(scene.segs * 1000).astype(np.int64)
    # direction-free segment key
    a, b = q[:, :2], q[:, 2:]
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    lo = np.where(swap[:, None], b, a)
    hi = np.where(swap[:, None], a, b)
    seg_key = (lo[:, 0] * 1_000_003 + lo[:, 1]) * 1_000_033 + hi[:, 0] * 7 + hi[:, 1]
    item_hash = np.zeros(scene.n, dtype=np.int64)
    np.add.at(item_hash, scene.seg_item, seg_key % (2**61 - 1))
    has_geo = np.bincount(scene.seg_item, minlength=scene.n) > 0
    _, inv, counts = np.unique(item_hash, return_inverse=True, return_counts=True)
    copies = np.where(has_geo, counts[inv] - 1, 0)
    lay_key = item_hash * 1000 + scene.layer
    _, inv2, counts2 = np.unique(lay_key, return_inverse=True, return_counts=True)
    same_layer = np.where(has_geo, counts2[inv2] - 1, 0)
    return copies.astype(np.float32), same_layer.astype(np.float32)


def compute(scene: Scene) -> tuple[np.ndarray, list[str]]:
    n = scene.n
    cols: dict[str, np.ndarray] = {}
    types = scene.meta["types"]
    type_code = np.array([ETYPES.index(t) + 1 if t in ETYPES else 0 for t in types])[scene.etype]
    cols["etype"] = type_code
    cols["depth"] = scene.depth
    cols["in_block"] = scene.block >= 0
    cols["color"] = scene.color
    cols["closed"] = scene.closed
    cols["filled"] = scene.filled

    bb = scene.bboxes()
    w, h = np.nan_to_num(bb[:, 2] - bb[:, 0]), np.nan_to_num(bb[:, 3] - bb[:, 1])
    length = scene.lengths()
    nseg = np.bincount(scene.seg_item, minlength=n)
    cols.update(width=w, height=h, diag=np.hypot(w, h), aspect=np.maximum(w, h) / np.maximum(np.minimum(w, h), 1e-3),
                length=length, nseg=nseg, radius=scene.radius, sweep=scene.sweep, text_h=scene.text_h)

    d = scene.segs[:, 2:] - scene.segs[:, :2]
    seg_len = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 90
    axis = (np.minimum(ang, 90 - ang) < 1) & (seg_len > 0)
    cols["axis_share"] = np.bincount(scene.seg_item, weights=axis * seg_len, minlength=n) / np.maximum(length, 1e-9)

    # names
    layer_kw = _keyword_matrix(scene.meta["layers"])
    block_names = scene.meta["blocks"]
    block_kw = _keyword_matrix(block_names) if block_names else np.zeros((0, len(KW_KEYS)), dtype=np.float32)
    for j, k in enumerate(KW_KEYS):
        cols[f"layer_{k}"] = layer_kw[scene.layer, j]
        bk = np.zeros(n, dtype=np.float32)
        has_b = scene.block >= 0
        if len(block_kw):
            bk[has_b] = np.maximum(block_kw[scene.block[has_b], j], block_kw[scene.block_top[has_b], j])
        cols[f"block_{k}"] = bk
    cols["layer_is_0"] = np.array([name == "0" for name in scene.meta["layers"]])[scene.layer]

    # layer statistics in this drawing
    n_layers = len(scene.meta["layers"])
    lay_count = np.bincount(scene.layer, minlength=n_layers)
    is_text = np.isin(cols["etype"], [ETYPES.index(t) + 1 for t in ("TEXT", "MTEXT", "ATTRIB")])
    is_hatch = cols["etype"] == ETYPES.index("HATCH") + 1
    cols["layer_items_log"] = np.log1p(lay_count)[scene.layer]
    cols["layer_share"] = (lay_count / max(n, 1))[scene.layer]
    for name, values in (("text", is_text), ("hatch", is_hatch), ("in_block", scene.block >= 0),
                         ("axis", cols["axis_share"]), ("closed", scene.closed)):
        cols[f"layer_mean_{name}"] = (np.bincount(scene.layer, weights=values.astype(float), minlength=n_layers)
                                      / np.maximum(lay_count, 1))[scene.layer]
    cols["layer_mean_length"] = (np.bincount(scene.layer, weights=length, minlength=n_layers)
                                 / np.maximum(lay_count, 1))[scene.layer]

    # top-level entity this leaf came from
    top_count = np.bincount(scene.top, minlength=scene.top.max() + 1 if n else 1)
    cols["top_leaves_log"] = np.log1p(top_count[scene.top])

    # text
    tf = _text_flags(scene.text)
    for j, name in enumerate(["text_len", "text_number", "text_dim", "text_level", "text_area", "text_room",
                              "text_title", "text_axis", "text_lines"]):
        cols[name] = tf[:, j]

    # context
    twin, wall_pts = _wall_twins(scene, length)
    cols["wall_twin"] = twin
    cx, cy = np.nan_to_num((bb[:, 0] + bb[:, 2]) / 2), np.nan_to_num((bb[:, 1] + bb[:, 3]) / 2)
    centres = np.column_stack([cx, cy])
    if len(wall_pts):
        dist, _ = cKDTree(wall_pts).query(centres, distance_upper_bound=50)
        cols["dist_wall"] = np.minimum(dist, 50)
    else:
        cols["dist_wall"] = np.full(n, 50.0)
    ctree = cKDTree(centres)
    cols["nbr_05"] = np.log1p(ctree.query_ball_point(centres, 0.5, return_length=True))
    cols["nbr_2"] = np.log1p(ctree.query_ball_point(centres, 2.0, return_length=True))
    copies, same_layer = _copies(scene)
    cols["copies"] = copies
    cols["copies_same_layer"] = same_layer

    names = list(cols)
    X = np.column_stack([np.asarray(cols[k], dtype=np.float32) for k in names])
    return X, names
