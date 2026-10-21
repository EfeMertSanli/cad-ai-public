"""Rooms of a cleaned floor plan, on a raster (plan-placement M2.0, ADR-008; walls and doors, ADR-030).

A room is space closed by walls, its door side closed by the door (ADR-030):
1. Kept architecture (walls, windows, door frames, columns, stairs; no texts) is drawn on a 5 cm grid,
   gaps of a few centimetres between lines closed.
2. Every door swing of the raw drawing (the cleanup deletes them, the raw drawing keeps them) closes its
   doorway with a line from the hinge to the closed leaf's end: the arc end that meets a jamb (wall just
   beyond it, and wall just behind the hinge the other way).
3. The free space falls into parts. Parts touching the raster edge are outside; parts never wider than
   2 x THIN_M are the insides of walls.
4. Seeds are the raw drawing's room names and area stamps ("SALON", "18.03 m²"); a name and its stamp
   close together with no wall between are one seed. A part holding several seeds (rooms joined by an
   opening without a door swing) is shrunk step by step until every seed sits in its own core, and the
   cores grow back inside the part: the rooms meet at the narrowest opening. A part without seeds is
   split the old way (cores after shrinking CORE_M), so unnamed rooms joined by a doorway still part.
5. Closed shapes inside one room and nothing else (wardrobe, counter, shower tray drawn closed) are its
   floor: the architect's stamp counts them.
6. The room type comes from the names inside the room (ADR-026 for the open kitchen). The area stamp
   inside a room is kept with it (`stamp_m2`): the check compares the floor inside the walls with what
   the architect wrote.

Measured on area stamps (a stamp in a room whose area is within 10 % of it, the architect's convention
per project divided out, our toolbox cleanup as the base): test pairs 0.52 with the old wall-growing
reader, 0.72 with this one; without door lines 0.31.

Returns a label raster (0 = wall/outside, k = room k) with its transform, and per room: area,
centre, bounding box, type, name texts and the stamp.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from cadai.cad.scene import Scene
from cadai.corpus.inventory import fold

CELL_M = 0.05
GAP_PX = 2                 # line gaps up to ~20 cm close (drawing slack at corners and jambs)
JAMB_M = 0.15              # a door's closed end meets a jamb within this
DOOR_PAST_M = 0.1          # the door line runs this far into the jambs
SEED_JOIN_M = 1.2          # a room name and its stamp closer than this, nothing between: one seed
THIN_M = 0.2               # free parts never wider than 2 x this are the insides of walls
CORE_M = 0.55              # unnamed parts split where an opening is narrower than 2 x this
SPLIT_R = (0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1)
POCKET_M = 0.25            # closed shapes inside a room wider than 2 x this are its floor
POCKET_MAX_M2 = 3.0        # ... when smaller than this and unnamed
MIN_ROOM_M2 = 1.0
AREA_RE = re.compile(r"(\d{1,4}[.,]\d{1,2})\s*M\s*(2|²|\^2)", re.IGNORECASE)

# folded text fragment -> room type (first match wins, so specific names come first)
ROOM_TYPES = [
    ("EBEVEYN BANYO", "bathroom"), ("EBEVEYN", "bedroom"), ("YATAK", "bedroom"), ("Y.ODA", "bedroom"), ("Y. ODA", "bedroom"), ("COCUK", "bedroom"),
    ("CALISMA", "study"), ("SALON", "living"), ("OTURMA", "living"), ("YASAM", "living"), ("YEMEK", "living"),
    ("MUTFAK", "kitchen"), ("BANYO", "bathroom"), ("DUS", "bathroom"), ("WC", "wc"), ("TUVALET", "wc"),
    ("LAVABO", "wc"), ("ISLAK", "bathroom"), ("CAMASIR", "laundry"), ("GIYINME", "dressing"),
    ("VESTIYER", "dressing"), ("ANTRE", "hall"), ("HOL", "hall"), ("KORIDOR", "hall"), ("GIRIS", "hall"),
    ("BALKON", "balcony"), ("TERAS", "balcony"), ("KILER", "storage"), ("DEPO", "storage"),
    ("MERDIVEN", "stair"), ("SAHANLIK", "stair"), ("ASANSOR", "lift"), ("DUKKAN", "shop"), ("MAGAZA", "shop"),
    ("OFIS", "office"), ("BURO", "office"), ("SIGINAK", "shelter"), ("KAZAN", "plant"), ("OTOPARK", "parking"),
    ("ODA", "bedroom"),
]

# An open kitchen gets more lights and sockets than a living room of its size (archive, 11-15 m2:
# 1.7 lights against 0.7), and less than a kitchen's rules suggest (ADR-026).
OPEN_KITCHEN = "living_kitchen"
LIVING_WORDS = ("SALON", "OTURMA", "YASAM")


@dataclass
class Rooms:
    labels: np.ndarray            # (h, w) int32, 0 = none
    origin: tuple[float, float]   # metres of pixel (0, 0) lower-left
    rooms: list[dict]
    doors: list[dict] | None = None   # doorways: {"rooms": [a, b], "xy": [x, y], "width_m"}; b = 0 is outside
    _nearest: tuple | None = None

    def at(self, xy: np.ndarray, snap_m: float = 0.0) -> np.ndarray:
        """Room id (0 = none) at points in metres. With `snap_m`, a point on a wall or just outside
        takes the nearest room within that distance (devices are drawn on the wall line)."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        ij = np.floor((xy - self.origin) / CELL_M).astype(int)
        h, w = self.labels.shape
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        out = np.zeros(len(ij), dtype=np.int32)
        if snap_m <= 0:
            out[ok] = self.labels[ij[ok, 1], ij[ok, 0]]
            return out
        if self._nearest is None:
            dist, (ii, jj) = ndimage.distance_transform_edt(self.labels == 0, return_indices=True)
            self._nearest = (dist * CELL_M, self.labels[ii, jj])
        dist, lab = self._nearest
        r, c = ij[ok, 1], ij[ok, 0]
        out[ok] = np.where(dist[r, c] <= snap_m, lab[r, c], 0)
        return out


def room_type(text: str) -> str | None:
    f = fold(text)
    if "MUTFAK" in f and any(w in f for w in LIVING_WORDS):
        return OPEN_KITCHEN
    for word, t in ROOM_TYPES:
        if word in f:
            return t
    return None


def stamps(scene: Scene) -> list[dict]:
    """Area stamps of the raw drawing: value in m² and position."""
    bb = scene.bboxes()
    out = []
    for i in np.flatnonzero(scene.text != ""):
        m = AREA_RE.search(str(scene.text[i]).replace(" ", " "))
        if m and np.isfinite(bb[i, 0]):
            value = float(m.group(1).replace(",", "."))
            if 0.5 <= value <= 2000:
                out.append({"m2": value, "x": float((bb[i, 0] + bb[i, 2]) / 2), "y": float((bb[i, 1] + bb[i, 3]) / 2)})
    return out


# ------------------------------------------------------------------------------- raster
def _raster(segs: np.ndarray, x0: float, y0: float, w: int, h: int) -> np.ndarray:
    """Lines (metres) on the grid; row = y upwards, column = x."""
    img = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(img)
    px = (segs - (x0, y0, x0, y0)) / CELL_M
    inside = ((np.maximum(px[:, 0], px[:, 2]) >= 0) & (np.minimum(px[:, 0], px[:, 2]) < w)
              & (np.maximum(px[:, 1], px[:, 3]) >= 0) & (np.minimum(px[:, 1], px[:, 3]) < h))
    for a, b, c, d in px[inside].tolist():
        draw.line((a, b, c, d), fill=1, width=1)
    return np.array(img, dtype=bool)


def door_lines(swings: np.ndarray, walls: np.ndarray, origin) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per swing (hinge, end, end) the chord hinge -> closed end: the end with a jamb beyond it and wall
    behind the hinge the other way (the open leaf's end usually stands free in the room)."""
    h, w = walls.shape
    steps = int(round(JAMB_M / CELL_M))

    def wall_near(p, u) -> int:
        for t in range(1, steps + 1):
            q = p + u * t * CELL_M
            j, i = int((q[0] - origin[0]) / CELL_M), int((q[1] - origin[1]) / CELL_M)
            if 0 <= i < h and 0 <= j < w and walls[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].any():
                return 1
        return 0
    out = []
    for hinge, e1, e2 in swings:
        best, score = None, 0
        for e in (e1, e2):
            v = e - hinge
            length = float(np.hypot(*v))
            if not 0.4 <= length <= 1.4:
                continue
            u = v / length
            s = wall_near(e, u) + wall_near(hinge, -u)
            if s > score:
                best, score = e, s
        if best is not None:
            out.append((hinge, best))
    return out


def _seeds(scene: Scene, region, parts: np.ndarray, blocked: np.ndarray, origin) -> tuple[np.ndarray, np.ndarray]:
    """Points of room names and area stamps in the region, and a group id per point (a name and its
    stamp in the same part, close, with a clear line between, are one group)."""
    bb = scene.bboxes()
    x0, y0, x1, y1 = region
    idx = np.flatnonzero((scene.text != "") & np.isfinite(bb[:, 0]))
    cx, cy = (bb[idx, 0] + bb[idx, 2]) / 2, (bb[idx, 1] + bb[idx, 3]) / 2
    sel = (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
    pts = [(x, y) for i, x, y in zip(idx[sel], cx[sel], cy[sel])
           if AREA_RE.search(str(scene.text[i]).replace(" ", " ")) or room_type(str(scene.text[i]))]
    pts = np.array(pts, dtype=float).reshape(-1, 2)
    if not len(pts):
        return pts, np.zeros(0, int)
    h, w = blocked.shape
    ij = np.floor((pts - origin) / CELL_M).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    part = np.where(ok, parts[np.clip(ij[:, 1], 0, h - 1), np.clip(ij[:, 0], 0, w - 1)], 0)
    parent = list(range(len(pts)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in cKDTree(pts).query_pairs(SEED_JOIN_M):
        if part[a] != part[b]:
            continue
        m = int(np.hypot(*(pts[a] - pts[b])) / CELL_M) + 2
        q = np.floor((pts[a] + np.linspace(0, 1, m)[:, None] * (pts[b] - pts[a]) - origin) / CELL_M).astype(int)
        if not blocked[np.clip(q[:, 1], 0, h - 1), np.clip(q[:, 0], 0, w - 1)].any():
            parent[find(a)] = find(b)
    return pts, np.array([find(a) for a in range(len(pts))])


def _grow(part: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Labels grow one pixel at a time inside the mask until it is full (never across a wall)."""
    cross = ndimage.generate_binary_structure(2, 1)
    for _ in range(1000):
        free = mask & (part == 0)
        if not free.any():
            break
        grown = ndimage.grey_dilation(part, footprint=cross)
        take = free & (grown > 0)
        if not take.any():
            break
        part = np.where(take, grown, part)
    return part


def _cores(mask: np.ndarray, r_m: float) -> tuple[np.ndarray, int]:
    core = ndimage.binary_erosion(mask, structure=_disk(int(round(r_m / CELL_M))), border_value=1)
    return ndimage.label(core)


def _split(mask: np.ndarray, seeds: list[tuple[int, int, int]]) -> np.ndarray | None:
    """Part labels 1..k, one per seed group: the smallest shrinking that puts every group in its own
    core, grown back. None when no shrinking separates them."""
    groups = sorted({g for g, _, _ in seeds})
    for r in SPLIT_R:
        cl, cn = _cores(mask, r)
        if cn == 0:
            return None
        _, (ci, cj) = ndimage.distance_transform_edt(cl == 0, return_indices=True)
        owner: dict[int, set] = {}
        for g, i, j in seeds:
            owner.setdefault(g, set()).add(int(cl[ci[i, j], cj[i, j]]))
        cores = [next(iter(c)) for c in owner.values()]
        if all(len(c) == 1 for c in owner.values()) and len(set(cores)) == len(cores):
            part = np.zeros(mask.shape, np.int32)
            for n, g in enumerate(groups, start=1):
                part[cl == next(iter(owner[g]))] = n
            return _grow(part, mask)
    return None


def _merge_pockets(labels: np.ndarray, parts: np.ndarray, dist: np.ndarray, edge: set,
                   unnamed: set[int], doors_px: np.ndarray) -> np.ndarray:
    """Closed shapes inside a room: free parts left without a room, and small unnamed rooms (`unnamed`),
    that touch exactly one room across a single line, nothing else free and no door, join that room
    (a WC has a door, a wardrobe has not)."""
    spaces = np.zeros(int(parts.max()) + 1, dtype=bool)      # parts wide enough to be rooms, not outside
    for k, sl in enumerate(ndimage.find_objects(parts), start=1):
        if sl is not None and k not in edge:
            spaces[k] = dist[sl][parts[sl] == k].max() * CELL_M >= THIN_M
    for _ in range(3):
        changed = False
        cands = [("part", k) for k in np.unique(parts[(labels == 0) & (parts > 0)]).tolist() if k not in edge]
        cands += [("room", k) for k in sorted(unnamed) if (labels == k).any()]
        for kind, k in cands:
            where = (parts == k) & (labels == 0) if kind == "part" else labels == k
            sl = ndimage.find_objects(where.astype(np.int8))[0]
            sl = tuple(slice(max(a.start - 3, 0), a.stop + 3) for a in sl)
            pocket = where[sl]
            if dist[sl][pocket].max() * CELL_M < POCKET_M or pocket.sum() * CELL_M ** 2 > POCKET_MAX_M2:
                continue
            ring = ndimage.binary_dilation(pocket, iterations=3) & ~pocket
            near = labels[sl][ring]
            rooms_near = set(np.unique(near[near > 0]).tolist()) - {k if kind == "room" else -1}
            if (len(rooms_near) == 1 and not (ring & spaces[parts[sl]] & (labels[sl] == 0)).any()
                    and not (ring & doors_px[sl]).any()):
                labels[sl][pocket] = rooms_near.pop()
                unnamed.discard(k)
                changed = True
        if not changed:
            break
    return labels


# ------------------------------------------------------------------------------- rooms
def read_rooms(scene: Scene, keep: np.ndarray, region: tuple[float, float, float, float],
               swings: np.ndarray | None = None) -> Rooms:
    """keep: bool per item, the architecture that survives the cleanup. region: metres.
    swings: door swings (n, 3, 2) of the raw drawing (elements.swing_points); found here when None."""
    x0, y0, x1, y1 = region
    origin = (x0, y0)
    w, h = int(np.ceil((x1 - x0) / CELL_M)) + 1, int(np.ceil((y1 - y0) / CELL_M)) + 1
    is_text = scene.text != ""
    segs = scene.segs[keep[scene.seg_item] & ~is_text[scene.seg_item] & np.isfinite(scene.segs).all(1)]
    walls = _raster(segs, x0, y0, w, h)
    walls |= ndimage.binary_closing(walls, structure=_disk(GAP_PX))
    if swings is None:
        swings = _swings(scene, region)
    lines = door_lines(swings, walls, origin)
    doors_px = np.zeros_like(walls)
    if lines:
        ends = []
        for a, b in lines:                    # 10 cm past both ends, so the line meets the jambs
            u = (b - a) / max(float(np.hypot(*(b - a))), 1e-9) * DOOR_PAST_M
            ends.append([*(a - u), *(b + u)])
        doors_px = _raster(np.array(ends), x0, y0, w, h)
        doors_px = ndimage.binary_dilation(doors_px) & ~walls
    blocked = walls | doors_px
    parts, _ = ndimage.label(~blocked)
    dist = ndimage.distance_transform_edt(~blocked)
    edge = set(np.unique(np.concatenate([parts[0], parts[-1], parts[:, 0], parts[:, -1]])).tolist()) - {0}
    pts, groups = _seeds(scene, region, parts, blocked, origin)
    seed_ij = np.floor((pts - origin) / CELL_M).astype(int)

    labels = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    unnamed: set[int] = set()                                 # whole parts without a name or stamp
    for k, sl in enumerate(ndimage.find_objects(parts), start=1):
        if sl is None or k in edge:
            continue                                          # outside
        mask = parts[sl] == k
        if mask.sum() * CELL_M ** 2 < MIN_ROOM_M2 * 0.5 or dist[sl][mask].max() * CELL_M < THIN_M:
            continue                                          # a gap, or the inside of a wall
        seeds = [(int(g), int(j) - sl[0].start, int(i) - sl[1].start) for (i, j), g in zip(seed_ij, groups)
                 if 0 <= j - sl[0].start < mask.shape[0] and 0 <= i - sl[1].start < mask.shape[1]
                 and mask[j - sl[0].start, i - sl[1].start]]
        if len({g for g, _, _ in seeds}) > 1:
            part = _split(mask, seeds)
        elif not seeds:
            cl, cn = _cores(mask, CORE_M)                     # unnamed: split at narrow openings
            big = [c for c in range(1, cn + 1) if (cl == c).sum() * CELL_M ** 2 >= 0.3]
            part = _grow(np.where(np.isin(cl, big), np.searchsorted(big, cl) + 1, 0).astype(np.int32), mask) \
                if len(big) > 1 else None
        else:
            part = None
        if part is None:
            part = mask.astype(np.int32)
            if not seeds:
                unnamed.add(next_id)
        part = np.where(mask & (part > 0), part + next_id - 1, 0)
        labels[sl] = np.where(part > 0, part, labels[sl])
        next_id = max(next_id, int(part.max()) + 1)
    labels = _merge_pockets(labels, parts, dist, edge, unnamed, doors_px)
    labels, n = _relabel(labels)
    # door-line pixels join the rooms beside them: the rooms keep the doorway's floor, and rooms meeting
    # at a door touch (doorways)
    if doors_px.any():
        dd, (ii, jj) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
        labels = np.where(doors_px & (labels == 0) & (dd <= 3), labels[ii, jj], labels)
    # a room's floor reaches the wall faces: half of the 5 cm line pixels around it count
    cross = ndimage.generate_binary_structure(2, 1)
    rim = np.zeros(n + 1)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is not None:
            sl = tuple(slice(max(a.start - 1, 0), a.stop + 1) for a in sl)
            m = labels[sl] == k
            rim[k] = (ndimage.binary_dilation(m, structure=cross) & ~m & walls[sl]).sum()

    bb = scene.bboxes()
    tx = np.flatnonzero(is_text & np.isfinite(bb[:, 0]))
    centres = np.column_stack([(bb[tx, 0] + bb[tx, 2]) / 2, (bb[tx, 1] + bb[tx, 3]) / 2])
    at = Rooms(labels, origin, []).at(centres, snap_m=0.3) if len(tx) else np.zeros(0, int)
    st = [s for s in stamps(scene) if x0 <= s["x"] <= x1 and y0 <= s["y"] <= y1]
    st_at = Rooms(labels, origin, []).at(np.array([[s["x"], s["y"]] for s in st]).reshape(-1, 2), snap_m=0.3)
    rooms = []
    for k in range(1, n + 1):
        mask = labels == k
        area = (mask.sum() + rim[k] / 2) * CELL_M ** 2
        if area < MIN_ROOM_M2:
            labels[mask] = 0
            continue
        rows, cols = np.nonzero(mask)
        names = [str(scene.text[tx[i]]) for i in np.flatnonzero(at == k)]
        types = [t for t in (room_type(s) for s in names) if t]
        if {"living", "kitchen"} <= set(types):                  # "SALON" and "MUTFAK" as two texts
            types.insert(0, OPEN_KITCHEN)
        in_room = [s["m2"] for s, r in zip(st, st_at) if r == k]
        rooms.append({
            "id": k, "area_m2": round(float(area), 2),
            "centre": [round(x0 + (cols.mean() + 0.5) * CELL_M, 3), round(y0 + (rows.mean() + 0.5) * CELL_M, 3)],
            "bbox": [round(x0 + cols.min() * CELL_M, 2), round(y0 + rows.min() * CELL_M, 2),
                     round(x0 + (cols.max() + 1) * CELL_M, 2), round(y0 + (rows.max() + 1) * CELL_M, 2)],
            "type": types[0] if types else None,
            "names": names[:4],
            "stamp_m2": in_room[0] if len(in_room) == 1 else None,   # several stamps: rooms read as one
            "stamps": len(in_room),
        })
    return Rooms(labels, origin, rooms, doorways(labels, walls, origin))


def _swings(scene: Scene, region) -> np.ndarray:
    from cadai.cleanup import elements, layers
    x0, y0, x1, y1 = region
    bb = scene.bboxes()
    where = (bb[:, 0] >= x0) & (bb[:, 2] <= x1) & (bb[:, 1] >= y0) & (bb[:, 3] <= y1)
    kind = layers.item_types(scene, layers.classify(scene)[0])
    types = np.array(scene.meta["types"])[scene.etype]
    return elements.swing_points(scene, where & np.isin(kind, ["door", "opening", "window", "wall", "unknown"]), types)


def area_check(rooms: list[dict]) -> dict:
    """The rooms against the architect's stamps: his convention (median of ours / his; above 1 he wrote
    less than the floor, below 1 more, e.g. with the walls) and the rooms off by more than 10 % after it."""
    pairs = [(r, r["area_m2"] / r["stamp_m2"]) for r in rooms if r.get("stamp_m2")]
    if not pairs:
        return {"stamped": 0, "ratio": None, "off": []}
    ratio = float(np.median([q for _, q in pairs]))
    off = [{"id": r["id"], "area_m2": r["area_m2"], "stamp_m2": r["stamp_m2"], "diff": round(q / ratio - 1, 3)}
           for r, q in pairs if abs(q / ratio - 1) > 0.10]
    return {"stamped": len(pairs), "ratio": round(ratio, 3), "off": off,
            "merged": [r["id"] for r in rooms if r.get("stamps", 0) > 1]}


def doorways(labels: np.ndarray, walls: np.ndarray, origin) -> list[dict]:
    """Where two rooms touch without a wall line between them (or a room touches free space
    outside every room) there is an opening: a door whatever way it was drawn (arc, polyline,
    block, or deleted by the cleanup). One entry per connected contact."""
    # -1 = free space outside every room, 0 = wall or nothing, k = room k
    lab = np.where((labels == 0) & ~walls, -1, labels)
    h, w = lab.shape
    contact = np.zeros((h, w), dtype=np.int64)          # pair key per contact pixel
    for dr, dc in ((0, 1), (1, 0)):                     # right and down neighbours
        a, b = lab[: h - dr, : w - dc], lab[dr:, dc:]
        touch = (a != 0) & (b != 0) & (a != b) & ((a > 0) | (b > 0))
        key = np.maximum(a, b) * 100_000 + np.maximum(np.minimum(a, b), 0)   # outside -> 0
        view = contact[: h - dr, : w - dc]
        new = touch & (view == 0)
        view[new] = key[new]
    out = []
    for key in np.unique(contact[contact > 0]):
        parts, n = ndimage.label(contact == key, structure=np.ones((3, 3)))
        for k in range(1, n + 1):
            rr, cc = np.nonzero(parts == k)
            width = (max(np.ptp(rr), np.ptp(cc)) + 1) * CELL_M
            if width < 0.5 or width > 3.0:                    # slivers and open-plan boundaries
                continue
            out.append({"rooms": [int(key // 100_000), int(key % 100_000)],
                        "xy": [round(origin[0] + (cc.mean() + 0.5) * CELL_M, 3), round(origin[1] + (rr.mean() + 0.5) * CELL_M, 3)],
                        "width_m": round(float(width), 2)})
    return out


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _relabel(labels: np.ndarray) -> tuple[np.ndarray, int]:
    ids = np.unique(labels[labels > 0])
    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[labels], len(ids)
