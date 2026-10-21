"""Candidate positions in a room, with the features the placement model scores them by (plan-placement M2.2-2.3).

Wall devices (sockets, switches, wall lights) sit on the room outline: candidates every 25 cm
along it. Ceiling lights sit inside: candidates on a 50 cm grid. Every candidate is described by
plain geometry, no rules:
- distance to the nearest doorway (rooms.doorways: where the room opens to another room or
  outside, however the door was drawn) and to the room centre;
- distance to the nearest piece of furniture, its size, and per confirmed furniture type the
  distance to the nearest one of that type (bed, counter, TV unit…);
- distance to the nearest corner of the room outline, distance to the walls (inside candidates);
- room type, area and proportions;
- where the candidate stands relative to its own room (ADR-026): its place in the room's box, distances
  divided by the room's size, and its rank among the room's candidates by distance to the door, the
  window, the centre and the door's latch side. "The wall point nearest the door" means the same in a
  6 m2 and a 30 m2 room; the plain distances don't;
- the rooms behind the room's doors (ADR-027): the type of the room behind the nearest door and the
  distance to the nearest door of a wet room, a kitchen, the outside, any room. A bathroom's switch is
  drawn in the hall, beside the bathroom door.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from cadai.placement import rooms as rooms_mod

WALL_STEP_M = 0.25
GRID_STEP_M = 0.5
FAR = 30.0
LATCH_ON_WALL_M = 0.1      # a swing's closing end sits on the jamb

ROOM_CODES = {t: i + 1 for i, t in enumerate(sorted({t for _, t in rooms_mod.ROOM_TYPES}))}
ROOM_CODES[rooms_mod.OPEN_KITCHEN] = len(ROOM_CODES) + 1      # appended: older models keep their codes
FURNITURE_GROUPS = ["bed_double", "bed_single", "wardrobe", "sofa", "table", "desk", "tv_unit", "kitchen_counter",
                    "sink", "stove", "fridge", "washing_machine", "wc", "washbasin", "shower", "bathtub", "boiler"]
WALL_CLASSES = ["socket", "socket_wet", "socket_power", "switch", "light_wall", "data", "tv", "intercom", "panel"]
CEILING_CLASSES = ["light_ceiling", "sensor", "light_emergency", "fire"]

# Furniture by size, while the catalogue has no confirmed types (ADR-022): short side x long side bins
# (a bed is about 1.6 x 2.0 m, a counter 0.6 x 2+ m, a WC 0.4 x 0.7 m).
SIZE_SHORT = (0.6, 1.2)
SIZE_LONG = (1.0, 2.2)
SIZE_BINS = [f"d_fsize_{i}{j}" for i in range(3) for j in range(3)]

FEATURES = ["room_type", "area", "width", "height", "aspect", "on_wall", "d_door", "d_centre", "d_corner", "d_wall",
            "d_furniture", "furniture_long", "furniture_short", "n_furniture", "n_doors"] + \
           [f"d_{g}" for g in FURNITURE_GROUPS] + ["d_window"] + SIZE_BINS + ["d_latch", "d_hinge"]
RANKED = ["d_door", "d_window", "d_centre", "d_latch"]
FEATURES += ["u_fold", "v_fold", "d_door_rel", "d_centre_rel", "d_window_rel"] + [f"rank_{f}" for f in RANKED]
NEIGHBOUR_FEATURES = ["nb_type", "d_wet_door", "d_kitchen_door", "d_out_door", "d_room_door",
                      "n_wet_doors", "n_neighbours", "n_out_doors"]
FEATURES += NEIGHBOUR_FEATURES
OUTSIDE = -1                   # nb_type of a door to no room (stair hall, outside); 0 = a room without a name
WET_TYPES = ("bathroom", "wc", "laundry")
KITCHEN_TYPES = ("kitchen", "living_kitchen")


def doors(rm: rooms_mod.Rooms, room_id: int) -> np.ndarray:
    """Doorway centres of a room (rooms.doorways: openings to another room or outside)."""
    pts = [d["xy"] for d in (rm.doors or []) if room_id in d["rooms"]]
    return np.array(pts, dtype=float).reshape(-1, 2)


def door_sides(swings: np.ndarray, walls: np.ndarray, rm: rooms_mod.Rooms | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(latch points, hinge points) of the plan's doors (ADR-023). A swing (elements.swing_points:
    hinge, two arc ends) closes onto the far jamb. Of its two arc ends the latch is the one whose
    line from the hinge crosses the doorway: the rooms on its two sides differ (`rm`), while the open
    leaf lies inside one room. When that does not decide, the end on the kept architecture (`walls`)."""
    if not len(swings) or not len(walls):
        return np.zeros((0, 2)), np.zeros((0, 2))
    tree = cKDTree(walls)
    da, _ = tree.query(swings[:, 1])
    db, _ = tree.query(swings[:, 2])
    pick_a = da <= db
    if rm is not None:
        cross_a, cross_b = _crosses(rm, swings[:, 0], swings[:, 1]), _crosses(rm, swings[:, 0], swings[:, 2])
        pick_a = np.where(cross_a != cross_b, cross_a, pick_a)
    near = np.minimum(da, db) <= LATCH_ON_WALL_M
    latch = np.where(pick_a[:, None], swings[:, 1], swings[:, 2])
    return latch[near], swings[near, 0]


def _crosses(rm: rooms_mod.Rooms, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Does the line a-b separate two different rooms (both sides of its middle, 30 cm out)?"""
    mid = (a + b) / 2
    d = b - a
    n = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(np.hypot(*d.T), 1e-6)[:, None]
    left, right = rm.at(mid + 0.3 * n), rm.at(mid - 0.3 * n)
    return (left > 0) & (right > 0) & (left != right)


def room_candidates(rm: rooms_mod.Rooms, room: dict, furn: list[dict],
                    furn_types: dict[str, str], windows: np.ndarray | None = None,
                    sides: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(points (n, 2) metres, features (n, len(FEATURES))) for one room. `windows`: points along the
    plan's window lines (metres); `sides`: door_sides() of the plan."""
    mask = rm.labels == room["id"]
    edge = mask & ~ndimage.binary_erosion(mask)
    wall_pts = _thin(np.argwhere(edge), WALL_STEP_M)
    inner = ndimage.distance_transform_edt(mask) * rooms_mod.CELL_M
    grid = np.argwhere(mask & (inner >= 0.3))
    grid_pts = _thin(grid, GRID_STEP_M)
    rc = np.vstack([wall_pts, grid_pts])
    on_wall = np.r_[np.ones(len(wall_pts)), np.zeros(len(grid_pts))]
    xy = np.column_stack([rm.origin[0] + (rc[:, 1] + 0.5) * rooms_mod.CELL_M,
                          rm.origin[1] + (rc[:, 0] + 0.5) * rooms_mod.CELL_M])
    if not len(xy):
        return np.zeros((0, 2)), np.zeros((0, len(FEATURES)))
    x0, y0, x1, y1 = room["bbox"]
    w, h = x1 - x0, y1 - y0
    corners = _corners(edge, rm)
    in_room_doors = doors(rm, room["id"])
    cols = {
        "room_type": np.full(len(xy), ROOM_CODES.get(room["type"], 0)),
        "area": np.full(len(xy), room["area_m2"]), "width": np.full(len(xy), min(w, h)),
        "height": np.full(len(xy), max(w, h)), "aspect": np.full(len(xy), max(w, h) / max(min(w, h), 0.1)),
        "on_wall": on_wall,
        "d_door": _nearest(xy, in_room_doors), "d_centre": np.hypot(*(xy - room["centre"]).T),
        "d_corner": _nearest(xy, corners),
        "d_wall": inner[rc[:, 0], rc[:, 1]],
        "n_doors": np.full(len(xy), len(in_room_doors)), "n_furniture": np.full(len(xy), len(furn)),
    }
    if furn:
        fxy = np.array([[f["x"], f["y"]] for f in furn])
        d, idx = cKDTree(fxy).query(xy)
        cols["d_furniture"] = d
        cols["furniture_long"] = np.array([furn[i]["long"] for i in idx])
        cols["furniture_short"] = np.array([furn[i]["short"] for i in idx])
    else:
        cols["d_furniture"] = np.full(len(xy), FAR)
        cols["furniture_long"] = cols["furniture_short"] = np.zeros(len(xy))
    for g in FURNITURE_GROUPS:
        pts = np.array([[f["x"], f["y"]] for f in furn if furn_types.get(f["sig"]) == g]).reshape(-1, 2)
        cols[f"d_{g}"] = _nearest(xy, pts)
    cols["d_window"] = _nearest(xy, windows if windows is not None else np.zeros((0, 2)))
    latch, hinge = sides if sides is not None else (np.zeros((0, 2)), np.zeros((0, 2)))
    cols["d_latch"], cols["d_hinge"] = _nearest(xy, latch), _nearest(xy, hinge)
    for i in range(3):
        for j in range(3):
            pts = np.array([[f["x"], f["y"]] for f in furn
                            if np.searchsorted(SIZE_SHORT, f["short"]) == i and np.searchsorted(SIZE_LONG, f["long"]) == j])
            cols[f"d_fsize_{i}{j}"] = _nearest(xy, pts.reshape(-1, 2))
    _relative(cols, xy, room, on_wall == 1)
    _neighbours(cols, xy, rm, room["id"])
    return xy, np.column_stack([np.asarray(cols[k], dtype=np.float32) for k in FEATURES])


def _relative(cols: dict, xy: np.ndarray, room: dict, on_wall: np.ndarray) -> None:
    """Features of a candidate relative to its room (ADR-026), added to `cols`."""
    x0, y0, x1, y1 = room["bbox"]
    u = (xy[:, 0] - x0) / max(x1 - x0, 0.1)
    v = (xy[:, 1] - y0) / max(y1 - y0, 0.1)
    cols["u_fold"], cols["v_fold"] = np.minimum(u, 1 - u), np.minimum(v, 1 - v)    # mirror-free
    size = np.sqrt(max(room["area_m2"], 0.5))
    for f in ("d_door", "d_centre", "d_window"):
        cols[f"{f}_rel"] = cols[f] / size
    for f in RANKED:
        rank = np.zeros(len(xy))
        for kind in (on_wall, ~on_wall):                        # wall and ceiling candidates apart
            d = cols[f][kind]
            rank[kind] = np.argsort(np.argsort(d, kind="stable")) / max(len(d) - 1, 1)
        cols[f"rank_{f}"] = rank


def _neighbours(cols: dict, xy: np.ndarray, rm: rooms_mod.Rooms, room_id: int) -> None:
    """Features of the rooms behind this room's doors (ADR-027), added to `cols`."""
    types = {r["id"]: r["type"] for r in rm.rooms}
    pts, behind = [], []                                   # behind: (type of the room behind, is it a room)
    for d in rm.doors or []:
        if room_id in d["rooms"]:
            other = d["rooms"][1] if d["rooms"][0] == room_id else d["rooms"][0]
            pts.append(d["xy"])
            behind.append((types.get(other), other > 0))
    n = len(xy)
    for f in ("d_wet_door", "d_kitchen_door", "d_out_door", "d_room_door"):
        cols[f] = np.full(n, FAR)
    cols["nb_type"] = np.full(n, float(OUTSIDE))
    wet = np.array([t in WET_TYPES for t, _ in behind], dtype=bool)
    kitchen = np.array([t in KITCHEN_TYPES for t, _ in behind], dtype=bool)
    is_room = np.array([r for _, r in behind], dtype=bool)
    cols["n_wet_doors"] = np.full(n, float(wet.sum()))
    cols["n_neighbours"] = np.full(n, float(is_room.sum()))
    cols["n_out_doors"] = np.full(n, float((~is_room).sum()))
    if not pts:
        return
    pts = np.array(pts, dtype=float)
    dist = np.hypot(xy[:, None, 0] - pts[None, :, 0], xy[:, None, 1] - pts[None, :, 1])
    codes = np.array([ROOM_CODES.get(t, 0) if r else OUTSIDE for t, r in behind], dtype=float)
    cols["nb_type"] = codes[dist.argmin(1)]
    for f, sel in (("d_wet_door", wet), ("d_kitchen_door", kitchen), ("d_out_door", ~is_room), ("d_room_door", is_room)):
        if sel.any():
            cols[f] = np.minimum(dist[:, sel].min(1), FAR)


def _thin(rc: np.ndarray, step_m: float) -> np.ndarray:
    """One pixel per step x step cell."""
    if not len(rc):
        return rc.reshape(0, 2)
    cell = max(1, int(round(step_m / rooms_mod.CELL_M)))
    _, first = np.unique(rc // cell, axis=0, return_index=True)
    return rc[np.sort(first)]


def _nearest(xy: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if not len(pts):
        return np.full(len(xy), FAR)
    d, _ = cKDTree(pts).query(xy)
    return np.minimum(d, FAR)


def _corners(edge: np.ndarray, rm: rooms_mod.Rooms) -> np.ndarray:
    """Outline pixels where the outline turns: few edge neighbours along a straight line."""
    horiz = ndimage.convolve(edge.astype(int), np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]), mode="constant")
    vert = ndimage.convolve(edge.astype(int), np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]), mode="constant")
    corner = edge & (horiz >= 1) & (vert >= 1)
    rc = np.argwhere(corner)
    return np.column_stack([rm.origin[0] + (rc[:, 1] + 0.5) * rooms_mod.CELL_M,
                            rm.origin[1] + (rc[:, 0] + 0.5) * rooms_mod.CELL_M])
