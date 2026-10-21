"""Placing devices on a cleaned drawing (Çizim page, ADR-023).

Input: a cleanup output folder (outputs/<name>, ADR-021). We need the raw drawing (furniture is read
from it, the cleanup deleted it) and which raw items the cleanup kept:
- new outputs store both (scene.npz, decision.npz, written by apply.clean_job);
- older ones are rebuilt: the source drawing is converted again, and a raw item counts as kept
  when its segments are in the cleaned drawing (clean.dxf).
Floor plans are the clusters of the kept drawing (every view the cleanup kept is a floor plan).
Per plan: rooms (rooms.py), candidates (candidates.py), the placement model's counts and ranking
(model.py). An LLM choosing between alternative layouts did not beat the model (ADR-023).
Output: devices.json, placed.png (rooms + devices), placed.dxf/.dwg (the cleaned drawing with
the devices on their own layers, ELK-*).
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from cadai import config
from cadai.i18n import say
from cadai.cad.scene import Scene
from cadai.placement import candidates as cand_mod
from cadai.placement import furniture, furniture_shapes, model, rooms as rooms_mod, steer
from cadai.placement.candidates import CEILING_CLASSES, FEATURES

OUTPUTS = config.DATA / "outputs"
# panel and emergency light: drafts (test 1 m F1 0.44 and 0.28, ADR-027); junction boxes wait for the wiring.
PLACE_CLASSES = ["light_ceiling", "socket", "socket_wet", "switch", "panel", "light_emergency"]
REGION_CELL_M = 1.0
REGION_GAP_CELLS = 2           # kept lines closer than this many metres belong to the same plan
MIN_PLAN_M2 = 30.0
MATCH_ROUND_M = 0.01
MIN_SCORED_DEVICES = 5        # a plan is compared with the engineer's when he drew at least this many devices on it
KEPT_SHARE = 0.8               # a raw item is kept when this share of its segments is in clean.dxf

LAYERS = {"light_ceiling": ("ELK-AYDINLATMA", 30), "socket": ("ELK-PRIZ", 1), "socket_wet": ("ELK-PRIZ-ETANJ", 6),
          "switch": ("ELK-ANAHTAR", 5), "panel": ("ELK-PANO", 8), "light_emergency": ("ELK-ACIL", 3)}
WIRE_LAYER = ("ELK-LINYE", 3)
COLOURS = {"light_ceiling": (235, 140, 0), "socket": (220, 0, 0), "socket_wet": (140, 0, 170), "switch": (0, 90, 230),
           "panel": (70, 70, 70), "light_emergency": (0, 150, 60)}
NAMES = {"light_ceiling": say("tavan armatürü", "ceiling light"), "socket": say("priz", "socket"),
         "socket_wet": say("etanj priz", "wet-area socket"), "switch": say("anahtar", "switch"),
         "panel": say("daire panosu", "apartment panel"), "light_emergency": say("acil aydınlatma", "emergency light")}


def latest_model() -> Path:
    found = sorted(p for p in config.MODEL_DIR.glob("placement-*") if (p / "report.json").exists())
    if not found:
        raise RuntimeError(str(say("Yerleştirme modeli yok; önce yerleştirme modelini eğitin.",
                                   "No placement model; train the placement model first.")))
    return found[-1]


# ------------------------------------------------------------------------------- the cleaned drawing
def _seg_keys(segs: np.ndarray) -> np.ndarray:
    """Direction-free rounded keys of segments, for matching two scenes of the same drawing."""
    a = np.round(segs[:, :2] / MATCH_ROUND_M).astype(np.int64)
    b = np.round(segs[:, 2:] / MATCH_ROUND_M).astype(np.int64)
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    lo, hi = np.where(swap[:, None], b, a), np.where(swap[:, None], a, b)
    return np.array([hash((x0, y0, x1, y1)) for x0, y0, x1, y1 in np.column_stack([lo, hi]).tolist()])


def kept_by_match(raw: Scene, clean: Scene) -> np.ndarray:
    """Raw items whose segments are (mostly) in the cleaned drawing. Texts by content and position."""
    ok_r = np.isfinite(raw.segs).all(1)
    ok_c = np.isfinite(clean.segs).all(1)
    clean_keys = set(_seg_keys(clean.segs[ok_c]).tolist())
    keys = np.zeros(len(raw.segs), dtype=np.int64)
    keys[ok_r] = _seg_keys(raw.segs[ok_r])
    present = np.isin(keys, list(clean_keys)) & ok_r
    total = np.bincount(raw.seg_item, minlength=raw.n)
    hit = np.bincount(raw.seg_item, weights=present, minlength=raw.n)
    kept = (total > 0) & (hit >= KEPT_SHARE * np.maximum(total, 1))
    # texts (room names) carry no segments in some scenes: match by text and rounded position
    is_text = raw.text != ""
    if is_text.any():
        bb_r, bb_c = raw.bboxes(), clean.bboxes()
        ct = np.flatnonzero(clean.text != "")
        have = {(str(clean.text[i]), round(float(bb_c[i, 0]), 1), round(float(bb_c[i, 1]), 1)) for i in ct}
        for i in np.flatnonzero(is_text):
            if (str(raw.text[i]), round(float(bb_r[i, 0]), 1), round(float(bb_r[i, 1]), 1)) in have:
                kept[i] = True
    return kept


def load_cleaned(out_dir: Path, log=print) -> tuple[Scene, np.ndarray]:
    """(raw scene, kept mask per raw item) of a cleanup output."""
    scene_file, dec_file = out_dir / "scene.npz", out_dir / "decision.npz"
    if scene_file.exists() and dec_file.exists():
        with np.load(dec_file) as z:
            return Scene.load(scene_file), z["kept"]
    from cadai.cad import oda
    from cadai.cad.scene import build
    info = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    src = Path(info["source"])
    if not src.exists():
        raise RuntimeError(str(say(f"Kaynak çizim bulunamadı: {src}", f"Source drawing not found: {src}")))
    log(say("eski çıktı: kaynak çizim yeniden okunuyor", "older output: reading the source drawing again"))
    raw = build(oda.to_dxf(src, out_dir / "source.dxf"))
    (out_dir / "source.dxf").unlink(missing_ok=True)
    from cadai.cad.units import UNITS
    clean = build(out_dir / "clean.dxf", unit={"unit": "cm", "scale": UNITS["cm"], "confidence": 1.0, "votes": {}})
    kept = kept_by_match(raw, clean)
    log(say(f"temiz çizimle eşleşen ham nesne: {int(kept.sum())} / {raw.n}",
            f"raw items matching the clean drawing: {int(kept.sum())} / {raw.n}"))
    raw.save(scene_file)
    np.savez_compressed(dec_file, kept=kept)
    return raw, kept


def plan_regions(scene: Scene, kept: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Bounding boxes (metres) of the clusters of kept lines: one per floor plan. A cluster is a plan
    when its lines span at least MIN_PLAN_M2 (box area: a plan's outline alone is sparse)."""
    from cadai.cleanup.align import _sample
    segs = scene.segs[kept[scene.seg_item] & np.isfinite(scene.segs).all(1)]
    if not len(segs):
        return []
    pts = _sample(segs, 0.5)[0]
    lo = pts.min(0)
    ij = np.floor((pts - lo) / REGION_CELL_M).astype(int)
    grid = np.zeros(ij.max(0) + 1, dtype=bool)
    grid[ij[:, 0], ij[:, 1]] = True
    lab, _ = ndimage.label(ndimage.binary_dilation(grid, iterations=REGION_GAP_CELLS))
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        cells = np.argwhere((lab[sl] == k) & grid[sl])
        span = (cells.max(0) - cells.min(0) + 1) * REGION_CELL_M
        if span[0] * span[1] < MIN_PLAN_M2:
            continue
        (i0, j0), (i1, j1) = cells.min(0) + (sl[0].start, sl[1].start), cells.max(0) + (sl[0].start, sl[1].start)
        out.append((float(lo[0] + i0 * REGION_CELL_M - 1), float(lo[1] + j0 * REGION_CELL_M - 1),
                    float(lo[0] + (i1 + 1) * REGION_CELL_M + 1), float(lo[1] + (j1 + 1) * REGION_CELL_M + 1)))
    return out


# ------------------------------------------------------------------------------- placing
class Placer:
    def __init__(self, model_dir: Path | None = None):
        self.dir = model_dir or latest_model()
        self.room_type = (lgb.Booster(model_file=str(self.dir / "room_type.txt"))
                          if (self.dir / "room_type.txt").exists() else None)
        self.boosters = {}
        for c in PLACE_CLASSES:
            if (self.dir / f"count_{c}.txt").exists():
                self.boosters[c] = (lgb.Booster(model_file=str(self.dir / f"count_{c}.txt")),
                                    lgb.Booster(model_file=str(self.dir / f"rank_{c}.txt")))

    def guess_type(self, X: np.ndarray) -> str | None:
        """Type of an unnamed room from its geometry (model.type_features), or None without the model."""
        if self.room_type is None or not len(X):
            return None
        return model.TYPE_NAMES[int(np.argmax(self.room_type.predict(model.type_features(X)[None, :])))]

    def room(self, xy: np.ndarray, X: np.ndarray, rules: list[dict] | None = None,
             room_type: str | None = None) -> dict[str, dict]:
        """Per class: predicted count, candidate scores and the model's pick (indices into xy).
        `rules`: the engineer's instruction as steer.py rules, applied on top of the model."""
        on_wall = X[:, FEATURES.index("on_wall")] == 1
        row = dict(zip(model.COUNT_FEATURES, model.room_row(X)))
        out = {}
        for c, (cm, rk) in self.boosters.items():
            idx = np.flatnonzero(~on_wall if c in CEILING_CLASSES else on_wall)
            room_row = np.array([[row[f] for f in cm.feature_name()]], dtype=np.float32)
            n = int(model.device_counts(cm.predict(room_row), c)[0]) if len(idx) else 0
            # columns by the model's own feature names: a model trained before a feature was added still works
            cols = [FEATURES.index(f) for f in rk.feature_name()]
            scores = rk.predict(X[np.ix_(idx, cols)]) if len(idx) else np.zeros(0)
            if rules and len(idx):
                n, scores = steer.apply(rules, c, room_type, n, X[idx], scores)
            pick = (idx[model.place_room(scores, xy[idx], n, model.SPACING_M.get(c, model.MIN_SPACING_M))]
                    if n > 0 else np.zeros(0, int))
            out[c] = {"n": n, "idx": idx, "scores": scores, "pick": pick}
        return out


def facing(rm: rooms_mod.Rooms, room_id: int, p: np.ndarray) -> float:
    """Degrees of the axis direction from a wall point into its room (the side that is the room
    30 cm out while the opposite side is not); 0 when it can't tell."""
    best, score = 0.0, -1
    for ang in (0.0, 90.0, 180.0, 270.0):
        u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        ahead, behind = rm.at(np.array([p + 0.3 * u, p - 0.3 * u]))
        s = int(ahead == room_id) * 2 - int(behind == room_id)
        if s > score:
            best, score = ang, s
    return best


def place(scene: Scene, kept: np.ndarray, placer: Placer, log=print, rules: list[dict] | None = None) -> dict:
    """Rooms and devices of every floor plan of the cleaned drawing."""
    from cadai.placement.dataset import window_points
    regions = plan_regions(scene, kept)
    log(say(f"{len(regions)} kat planı bulundu", f"{len(regions)} floor plans found"))
    bb = scene.bboxes()
    cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
    in_plans = np.zeros(scene.n, dtype=bool)
    for x0, y0, x1, y1 in regions:
        in_plans |= (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    furn_all = furniture.instances(scene, np.zeros(scene.n, dtype=np.int8), with_shape=True) +         furniture.loose_instances(scene, in_plans & ~kept, with_shape=True)                  # ADR-025
    furn_types = furniture_shapes.Matcher().types_for(furn_all)
    furn_all = [f for f in furn_all if furn_types.get(f["sig"]) not in furniture_shapes.NOT_FURNITURE]
    named = sum(1 for f in furn_all if f["sig"] in furn_types)
    log(say(f"{len(furn_all)} mobilya / vitrifiye, {named} tanesi adlı", f"{len(furn_all)} furniture / sanitary items, {named} named"))
    furn_xy = np.array([[f["x"], f["y"]] for f in furn_all]).reshape(-1, 2)
    windows = window_points(scene, kept)
    from cadai.placement.dataset import swing_points, wall_points
    swings = swing_points(scene, np.ones(scene.n, dtype=bool))
    walls = wall_points(scene, kept)
    plans = []
    for v, region in enumerate(regions):
        rm = rooms_mod.read_rooms(scene, kept, region, swings)
        sides = cand_mod.door_sides(swings, walls, rm)
        f_room = rm.at(furn_xy, snap_m=0.4) if len(furn_xy) else np.zeros(0, int)
        rooms_out = []
        for r in rm.rooms:
            room_furn = [f for f, k in zip(furn_all, f_room) if k == r["id"]]
            xy, X = cand_mod.room_candidates(rm, r, room_furn, furn_types, windows, sides)
            devs = []
            guessed = None
            if len(xy) and r["type"] is None:
                guessed = placer.guess_type(X)
                if guessed:
                    X[:, FEATURES.index("room_type")] = cand_mod.ROOM_CODES[guessed]
            if len(xy):
                res = placer.room(xy, X, rules, r["type"] or guessed)
                for c, p in res.items():
                    for i in p["pick"]:
                        d = {"type": c, "x": round(float(xy[i, 0]), 3), "y": round(float(xy[i, 1]), 3)}
                        if c not in CEILING_CLASSES:
                            d["angle"] = facing(rm, r["id"], xy[i])
                        devs.append(d)
            rooms_out.append({**{k: r[k] for k in ("id", "names", "area_m2", "bbox", "centre")},
                              "stamp_m2": r.get("stamp_m2"), "stamps": r.get("stamps", 0),
                              "type": r["type"] or guessed, "type_guessed": guessed is not None,
                              "devices": devs, "links": light_links(devs), "furniture": len(room_furn)})
        if not rooms_out:                             # frame or title lines left alone: not a plan
            continue
        plans.append({"region": list(region), "rooms": rooms_out, "_rm": rm})
        n_dev = sum(len(r["devices"]) for r in rooms_out)
        log(say(f"plan {v + 1}: {len(rooms_out)} oda, {n_dev} cihaz", f"plan {v + 1}: {len(rooms_out)} rooms, {n_dev} devices"))
    return {"plans": plans, "model": placer.dir.name}


def light_links(devs: list[dict]) -> list[list[float]]:
    """Phase 3, first piece: the archive's finals join every ceiling light to the switch of its room
    with one straight wire (75 % of wiring curves are single straight lines; light-switch is the most
    common device pair). Each light goes to the nearest switch of its own room. [x0, y0, x1, y1] metres."""
    sw = np.array([[d["x"], d["y"]] for d in devs if d["type"] == "switch"]).reshape(-1, 2)
    if not len(sw):
        return []
    out = []
    for d in devs:
        if d["type"] == "light_ceiling":
            s = sw[np.argmin(np.hypot(*(sw - (d["x"], d["y"])).T))]
            out.append([d["x"], d["y"], round(float(s[0]), 3), round(float(s[1]), 3)])
    return out


# ------------------------------------------------------------------------------- output
def sheet_scale(result: dict) -> float:
    """Pixels per metre, the same for every plan: all plans side by side about 6000 px wide,
    at most 60 px/m (a symbol stays readable)."""
    total = sum(p["region"][2] - p["region"][0] for p in result["plans"]) or 1.0
    return min(60.0, 6000.0 / total)


def picture(scene: Scene, kept: np.ndarray, result: dict, target: Path, width: int = 2400) -> Path:
    """Every plan side by side: kept architecture, rooms tinted with type, devices as symbols."""
    panels = []
    for plan in result["plans"]:
        x0, y0, x1, y1 = plan["region"]
        sc = sheet_scale(result)
        W, H = int((x1 - x0) * sc), int((y1 - y0) * sc)
        rm = plan["_rm"]
        gx, gy = np.meshgrid(x0 + (np.arange(W) + 0.5) / sc, y1 - (np.arange(H) + 0.5) / sc)
        lab = rm.at(np.column_stack([gx.ravel(), gy.ravel()])).reshape(H, W)
        arr = np.full((H, W, 3), 255, np.uint8)
        arr[lab > 0] = (255, 249, 215)
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
        segs = scene.segs[kept[scene.seg_item] & (scene.text == "")[scene.seg_item] & np.isfinite(scene.segs).all(1)]
        px = np.column_stack([(segs[:, 0] - x0) * sc, (y1 - segs[:, 1]) * sc, (segs[:, 2] - x0) * sc, (y1 - segs[:, 3]) * sc])
        inside = (px[:, [0, 2]].max(1) >= 0) & (px[:, [0, 2]].min(1) <= W) & (px[:, [1, 3]].max(1) >= 0) & (px[:, [1, 3]].min(1) <= H)
        for a, b, c, e in px[inside].tolist():
            d.line((a, b, c, e), fill=(60, 60, 60), width=1)
        r = max(4, int(sc * 0.12))
        for room in plan["rooms"]:
            for a, b, c, e in room.get("links", []):
                d.line(((a - x0) * sc, (y1 - b) * sc, (c - x0) * sc, (y1 - e) * sc), fill=(0, 160, 120), width=1)
            cx, cy = (room["centre"][0] - x0) * sc, (y1 - room["centre"][1]) * sc
            d.text((cx - 20, cy + r + 4), f"{room['type'] or '?'}{'?' if room.get('type_guessed') else ''} {room['area_m2']:.0f}m2", fill=(130, 130, 130))
            for dev in room["devices"]:
                X, Y = (dev["x"] - x0) * sc, (y1 - dev["y"]) * sc
                col = COLOURS[dev["type"]]
                if dev["type"] == "light_ceiling":
                    d.ellipse((X - r, Y - r, X + r, Y + r), outline=col, width=3)
                    d.line((X - r * .7, Y - r * .7, X + r * .7, Y + r * .7), fill=col, width=2)
                    d.line((X - r * .7, Y + r * .7, X + r * .7, Y - r * .7), fill=col, width=2)
                elif dev["type"] == "socket":
                    d.ellipse((X - r, Y - r, X + r, Y + r), fill=col)
                elif dev["type"] == "socket_wet":
                    d.rectangle((X - r, Y - r, X + r, Y + r), fill=col)
                elif dev["type"] == "panel":
                    d.rectangle((X - r * .6, Y - r * 1.4, X + r * .6, Y + r * 1.4), outline=col, width=3)
                elif dev["type"] == "light_emergency":
                    d.ellipse((X - r, Y - r, X + r, Y + r), outline=col, width=3)
                    d.ellipse((X - r * .4, Y - r * .4, X + r * .4, Y + r * .4), fill=col)
                else:
                    d.polygon([(X, Y - r - 2), (X - r, Y + r), (X + r, Y + r)], fill=col)
        panels.append(img)
    if not panels:
        panels = [Image.new("RGB", (400, 100), "white")]
    W = sum(p.width for p in panels) + 10 * len(panels)
    H = max(p.height for p in panels)
    sheet = Image.new("RGB", (W, H), "white")
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 10
    if sheet.width > 6000:
        sheet = sheet.resize((6000, int(sheet.height * 6000 / sheet.width)))
    sheet.save(target)
    return target


# The office's own symbol blocks (the most used confirmed block per type). Their geometry sits far
# from the block base (the light's X is ~3 m away), so it is stored centred: OFFICE_BLOCKS file,
# written once by harvest_office_blocks() from a final that has them. Rotation 0 of the wall symbols:
# wall on the left (-x), room on the right (+x). The switch has no office block (always exploded).
OFFICE_BLOCK_NAMES = {"light_ceiling": "A$C69B63251", "socket": "A$C452F5B38", "socket_wet": "A$C66B11EAB"}
OFFICE_BLOCKS = config.DATA / "symbols" / "office_blocks.json"


def harvest_office_blocks(final_dxf: Path) -> dict:
    """Centred lines, arcs and solid fills of the office blocks -> OFFICE_BLOCKS."""
    from ezdxf import recover
    doc, _ = recover.readfile(str(final_dxf))
    out = {}
    for cls, name in OFFICE_BLOCK_NAMES.items():
        blk = doc.blocks.get(name)
        if blk is None:
            continue
        prims, pts = [], []
        for e in blk:
            t = e.dxftype()
            if t == "LINE":
                a, b = (e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)
                prims.append(["line", *a, *b])
                pts += [a, b]
            elif t in ("ARC", "CIRCLE"):
                a0, a1 = (e.dxf.start_angle, e.dxf.end_angle) if t == "ARC" else (0.0, 360.0)
                prims.append(["arc", e.dxf.center.x, e.dxf.center.y, e.dxf.radius, a0, a1])
                pts += [(e.dxf.center[0] - e.dxf.radius, e.dxf.center[1] - e.dxf.radius),
                        (e.dxf.center[0] + e.dxf.radius, e.dxf.center[1] + e.dxf.radius)]
            elif t == "HATCH" and e.dxf.solid_fill:
                for path in e.paths:
                    v = [(float(q[0]), float(q[1])) for q in getattr(path, "vertices", [])]
                    if len(v) >= 3:
                        prims.append(["fill", v])
        if not pts:
            continue
        c = (np.min(pts, axis=0) + np.max(pts, axis=0)) / 2
        out[cls] = [_shift(q, -c[0], -c[1]) for q in prims]
    OFFICE_BLOCKS.write_text(json.dumps(out), encoding="utf-8")
    return out


def _shift(prim: list, dx: float, dy: float) -> list:
    if prim[0] == "line":
        return ["line", prim[1] + dx, prim[2] + dy, prim[3] + dx, prim[4] + dy]
    if prim[0] == "arc":
        return ["arc", prim[1] + dx, prim[2] + dy, *prim[3:]]
    return ["fill", [(x + dx, y + dy) for x, y in prim[1]]]


def _define_blocks(doc) -> dict[str, str]:
    """Block name per device type: the office's where known, otherwise a simple one of ours."""
    office = json.loads(OFFICE_BLOCKS.read_text(encoding="utf-8")) if OFFICE_BLOCKS.exists() else {}
    names = {}
    for c in LAYERS:
        name = f"CADAI_{c.upper()}"
        names[c] = name
        if name in doc.blocks:
            continue
        blk = doc.blocks.new(name)
        if c in office:
            for q in office[c]:
                if q[0] == "line":
                    blk.add_line(q[1:3], q[3:5])
                elif q[0] == "arc":
                    blk.add_arc(q[1:3], q[3], q[4], q[5])
                else:
                    blk.add_solid(q[1][:4]) if len(q[1]) <= 4 else blk.add_lwpolyline(q[1], close=True)
        elif c == "light_ceiling":
            blk.add_circle((0, 0), 15)
            blk.add_line((-10.6, -10.6), (10.6, 10.6))
            blk.add_line((-10.6, 10.6), (10.6, -10.6))
        elif c == "socket":
            blk.add_arc((5, 0), 10, 90, 270)
            blk.add_line((-5, -10), (-5, 10))
        elif c == "socket_wet":
            blk.add_arc((5, 0), 10, 90, 270)
            blk.add_line((-5, -10), (-5, 10))
            blk.add_line((-5, 0), (5, 0))
        elif c == "panel":                                # box on the wall with a diagonal
            blk.add_lwpolyline([(0, -20), (12, -20), (12, 20), (0, 20)], close=True)
            blk.add_line((0, -20), (12, 20))
        elif c == "light_emergency":                      # ceiling light with a filled centre
            blk.add_circle((0, 0), 15)
            blk.add_circle((0, 0), 5)
            blk.add_line((-10.6, -10.6), (10.6, 10.6))
            blk.add_line((-10.6, 10.6), (10.6, -10.6))
        else:                                             # switch: circle on the wall, stroke into the room
            blk.add_circle((6, 0), 6)
            blk.add_line((10, 4), (22, 14))
    return names


def write_dxf(clean_dxf: Path, result: dict, target: Path) -> dict:
    """The cleaned drawing plus the devices (office blocks, wall devices facing into the room) on
    ELK-* layers and the light-switch wires on ELK-LINYE. The scene is in metres, clean.dxf in
    centimetres (apply.write_clean to_cm)."""
    from ezdxf import recover
    doc, _ = recover.readfile(str(clean_dxf))
    msp = doc.modelspace()
    k = 100.0                                         # metres -> centimetres
    for layer, colour in list(LAYERS.values()) + [WIRE_LAYER]:
        if layer not in doc.layers:
            doc.layers.add(layer, color=colour)
    blocks = _define_blocks(doc)
    n = 0
    for plan in result["plans"]:
        for room in plan["rooms"]:
            for a, b, c, e in room.get("links", []):
                msp.add_line((a * k, b * k), (c * k, e * k), dxfattribs={"layer": WIRE_LAYER[0]})
            for dev in room["devices"]:
                msp.add_blockref(blocks[dev["type"]], (dev["x"] * k, dev["y"] * k),
                                 dxfattribs={"layer": LAYERS[dev["type"]][0], "rotation": dev.get("angle", 0.0)})
                n += 1
    doc.saveas(target)
    return {"devices": n}


def archive_devices(source: str, scene: Scene) -> list[dict] | None:
    """The engineer's devices (raw coordinates) when the drawing is an archive project's raw drawing
    with an aligned final; None otherwise."""
    from cadai.corpus import inventory, process
    from cadai.placement import devices, symbols
    try:
        rel = str(Path(source).resolve().relative_to(config.PROJECTS.resolve()))
    except ValueError:
        return None
    inv = inventory.load()
    for p in inv["projects"]:
        sha = next((f["sha"] for f in p["files"] if Path(f["rel"]) == Path(rel)), None)
        if not sha:
            continue
        for pair in p["pairs"]:
            info = process.label_info(p["id"], pair["final"]) if pair["raw"] == sha else None
            if info and info.get("ok"):
                final = Scene.load(process.scene_path(pair["final"]))
                devs = devices.locate(final, info["transforms"], symbols.load_types(confirmed_only=True), scene,
                                       by_shape=True)
                return [d for d in devs if d["type"] in PLACE_CLASSES]
    return None


def compare_picture(scene: Scene, kept: np.ndarray, result: dict, truth: list[dict], target: Path) -> dict:
    """Ours (rings) over the engineer's (filled); hits within 0.5 / 1 m per class."""
    shown = {"plans": [{**p, "rooms": [{**r, "devices": []} for r in p["rooms"]]} for p in result["plans"]]}
    picture(scene, kept, shown, target)
    img = Image.open(target)
    # redraw: simpler to draw both sets on the plan picture of each region
    panels_x, x = [], 0
    for plan in result["plans"]:
        x0, y0, x1, y1 = plan["region"]
        sc = sheet_scale(result)
        panels_x.append((x, sc))
        x += int((x1 - x0) * sc) + 10
    total_w = x - 10
    shrink = img.width / max(total_w, 1)
    d = ImageDraw.Draw(img)
    ours = [dev for p in result["plans"] for r in p["rooms"] for dev in r["devices"]]
    for (ox, sc), plan in zip(panels_x, result["plans"]):
        x0, y0, x1, y1 = plan["region"]
        rad = max(4, int(sc * 0.12)) * shrink
        def P(px, py):
            return (ox + (px - x0) * sc) * shrink, (y1 - py) * sc * shrink
        for dev in truth:
            if x0 <= dev["x"] <= x1 and y0 <= dev["y"] <= y1:
                X, Y = P(dev["x"], dev["y"])
                d.ellipse((X - rad * .8, Y - rad * .8, X + rad * .8, Y + rad * .8), fill=COLOURS[dev["type"]])
        for dev in ours:
            if x0 <= dev["x"] <= x1 and y0 <= dev["y"] <= y1:
                X, Y = P(dev["x"], dev["y"])
                d.ellipse((X - rad * 1.6, Y - rad * 1.6, X + rad * 1.6, Y + rad * 1.6), outline=COLOURS[dev["type"]], width=3)
    img.save(target)
    # the engineer draws the typical floor once: score only the plans that carry his devices
    def inside(q, region):
        return region[0] <= q["x"] <= region[2] and region[1] <= q["y"] <= region[3]
    drawn = [pl["region"] for pl in result["plans"] if sum(inside(q, pl["region"]) for q in truth) >= MIN_SCORED_DEVICES]
    ours = [q for q in ours if any(inside(q, r) for r in drawn)]
    truth = [q for q in truth if any(inside(q, r) for r in drawn)]
    score = {}
    for c in PLACE_CLASSES:
        pred = np.array([[q["x"], q["y"]] for q in ours if q["type"] == c]).reshape(-1, 2)
        tru = np.array([[q["x"], q["y"]] for q in truth if q["type"] == c]).reshape(-1, 2)
        score[c] = {"ours": len(pred), "engineer": len(tru), "hits_05": model._match(pred, tru, 0.5),
                    "hits_10": model._match(pred, tru, 1.0)}
    return score


def place_job(job, output: str, model_name: str | None = None, instruction: str | None = None,
              rules: list[dict] | None = None) -> dict:
    """`rules`: the rules the engineer confirmed on the Çizim page; without them the office rule book and the
    instruction translated by Haiku are applied (ADR-027, ADR-028)."""
    out_dir = (OUTPUTS / output).resolve()
    if OUTPUTS.resolve() not in out_dir.parents or not (out_dir / "summary.json").exists():
        raise RuntimeError(str(say(f"temizlik çıktısı yok: {output}", f"no cleanup output: {output}")))
    job.progress(0, 4)
    scene, kept = load_cleaned(out_dir, log=job.log)
    job.progress(1)
    placer = Placer(config.MODEL_DIR / model_name if model_name else None)
    if rules is not None:
        rules = steer.check(rules, PLACE_CLASSES)
        job.log(say(f"onaylı kurallar: {json.dumps(rules, ensure_ascii=False)}", f"confirmed rules: {json.dumps(rules, ensure_ascii=False)}"))
    else:
        rules = steer.load_book(PLACE_CLASSES)
        if rules:
            job.log(say(f"kural defteri: {json.dumps(rules, ensure_ascii=False)}", f"rule book: {json.dumps(rules, ensure_ascii=False)}"))
        if instruction and instruction.strip():
            said, cost = steer.translate(instruction.strip(), PLACE_CLASSES)
            rules_json = json.dumps(said, ensure_ascii=False)
            job.log(say(f"talimat: {instruction.strip()} → kural: {rules_json} (${cost:.3f})",
                        f"instruction: {instruction.strip()} → rules: {rules_json} (${cost:.3f})"))
            rules += said
    if not rules:
        job.log(say("kural yok; model kendi başına yerleştiriyor", "no rules; the model places on its own"))
    result = place(scene, kept, placer, log=job.log, rules=rules)
    job.progress(2)
    picture(scene, kept, result, out_dir / "placed.png")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    truth = archive_devices(summary["source"], scene)
    score = None
    if truth:
        score = compare_picture(scene, kept, result, truth, out_dir / "placed_compare.png")
        job.log(say("mühendisle karşılaştırma: " + ", ".join(
            f"{NAMES[c].tr} biz {v['ours']} / mühendis {v['engineer']}, 1 m içinde {v['hits_10']}" for c, v in score.items()),
            "compared with the engineer: " + ", ".join(
            f"{NAMES[c].en} ours {v['ours']} / engineer {v['engineer']}, within 1 m {v['hits_10']}" for c, v in score.items())))
    stats = write_dxf(out_dir / "clean.dxf", result, out_dir / "placed.dxf")
    try:
        from cadai.cad import oda
        oda.to_dwg(out_dir / "placed.dxf", out_dir / "placed.dwg")
    except Exception as exc:  # noqa: BLE001 - the DXF is still usable
        job.log(say(f"DWG yazılamadı: {exc}", f"could not write the DWG: {exc}"))
    job.progress(3)
    counts: dict[str, int] = {}
    for plan in result["plans"]:
        for room in plan["rooms"]:
            for d in room["devices"]:
                counts[d["type"]] = counts.get(d["type"], 0) + 1
    from datetime import datetime
    record = {"model": result["model"], "created": datetime.now().strftime("%Y%m%d-%H%M%S"),
              "instruction": (instruction or "").strip(), "rules": rules,
              "score": score, "counts": counts,
              "plans": [{k: v for k, v in p.items() if k != "_rm"} for p in result["plans"]]}
    (out_dir / "devices.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    job.log(say("cihazlar: " + ", ".join(f"{NAMES[c].tr} {n}" for c, n in counts.items()) + f" · DXF {stats['devices']}",
                "devices: " + ", ".join(f"{NAMES[c].en} {n}" for c, n in counts.items()) + f" · DXF {stats['devices']}"))
    job.progress(4)
    return {"output": output, "model": result["model"], "counts": counts,
            "plans": len(result["plans"]), "rooms": sum(len(p["rooms"]) for p in result["plans"])}
