import numpy as np

from cadai.placement.rooms import read_rooms, room_type
from conftest import make_scene


def _box(x0, y0, x1, y1):
    return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


def test_two_rooms_with_a_door_gap():
    # a 8 x 4 m outline split by a wall at x = 4 with a 0.9 m door gap
    walls = _box(0, 0, 8, 4) + [(4, 0, 4, 1.5), (4, 2.4, 4, 4)]
    items = [(0, [s], None) for s in walls]
    items += [(1, [(1.5, 2, 2.5, 2.2)], "YATAK ODASI"), (1, [(5.5, 2, 6.5, 2.2)], "SALON")]
    scene = make_scene(items, layers=["DUVAR", "YAZI"])
    keep = np.array([True] * len(walls) + [False, False])
    rooms = read_rooms(scene, keep, (-1, -1, 9, 5))
    assert len(rooms.rooms) == 2
    by_type = {r["type"]: r for r in rooms.rooms}
    assert set(by_type) == {"bedroom", "living"}
    for r in rooms.rooms:
        assert abs(r["area_m2"] - 16.0) < 1.0                    # each half is 4 x 4 m
    assert rooms.at(np.array([[2.0, 2.0], [6.0, 2.0], [20.0, 20.0]])).tolist()[2] == 0
    on_wall = np.array([[2.0, 0.0], [2.0, -1.0]])                 # on the outer wall; 1 m outside
    assert rooms.at(on_wall).tolist() == [0, 0]
    snapped = rooms.at(on_wall, snap_m=0.4).tolist()
    assert snapped[0] == rooms.at(np.array([[2.0, 2.0]]))[0] and snapped[1] == 0


def test_room_type_names():
    assert room_type("YATAK ODASI 2") == "bedroom"
    assert room_type("Y.ODASI") == "bedroom"
    assert room_type("EBEVEYN BANYO") == "bathroom"
    assert room_type("Mutfak") == "kitchen"
    assert room_type("315") is None


def test_living_room_with_its_kitchen_is_an_open_kitchen():
    """Cevriye, Ali Kemal: "Salon & Mutfak" rooms were typed living (ADR-026)."""
    assert room_type("Salon & Mutfak") == "living_kitchen"
    assert room_type("SALON+MUTFAK") == "living_kitchen"
    assert room_type("SALON") == "living"
    walls = _box(0, 0, 6, 4)
    items = [(0, [s], None) for s in walls] + [(1, [(1, 2, 2, 2.2)], "SALON"), (1, [(4, 2, 5, 2.2)], "MUTFAK")]
    scene = make_scene(items, layers=["DUVAR", "YAZI"])
    rooms = read_rooms(scene, np.array([True] * 4 + [False, False]), (-1, -1, 7, 5))
    assert [r["type"] for r in rooms.rooms] == ["living_kitchen"]


def test_area_stamps_are_read():
    from cadai.check.cleanup_checks import stamps
    from conftest import make_scene
    scene = make_scene([(0, [(0, 0, 1, 0.2)], "SALON 18.03 m²"), (0, [(5, 5, 6, 5.2)], "12,5 M2"),
                        (0, [(9, 9, 9.5, 9.2)], "315")])
    assert [s["m2"] for s in stamps(scene)] == [18.03, 12.5]


def test_doorway_between_two_rooms():
    walls = _box(0, 0, 8, 4) + [(4, 0, 4, 1.5), (4, 2.4, 4, 4)]
    scene = make_scene([(0, [s], None) for s in walls], layers=["DUVAR"])
    rooms = read_rooms(scene, np.ones(len(walls), dtype=bool), (-1, -1, 9, 5))
    inner = [d for d in rooms.doors if 0 not in d["rooms"]]
    assert len(inner) == 1
    x, y = inner[0]["xy"]
    assert abs(x - 4.0) < 0.15 and 1.5 < y < 2.4 and 0.6 <= inner[0]["width_m"] <= 1.0


def _room_scene(walls, texts=()):
    items = [(0, [s], None) for s in walls] + [(1, [(x, y, x + 0.5, y + 0.2)], t) for x, y, t in texts]
    keep = np.array([True] * len(walls) + [False] * len(texts))
    return make_scene(items, layers=["DUVAR", "YAZI"]), keep


def test_narrow_wc_is_a_room():
    """The old reader grew the walls 0.55 m: rooms narrower than 1.1 m vanished (archive: WCs, showers)."""
    walls = [(0, 0, 4, 0), (4, 3, 0, 3), (0, 3, 0, 0), (4, 3, 4, 1.4)]     # a 4 x 3 m room ...
    walls += [(4, 0, 5, 0), (5, 0, 5, 1.4), (5, 1.4, 4, 1.4)]         # ... a 1 x 1.4 m WC beside it
    walls += [(4, 1.4, 4, 1.0), (4, 0.3, 4, 0)]                      # their wall with the WC's 0.7 m door
    scene, keep = _room_scene(walls)
    swing = np.array([[[4.0, 0.3], [4.0, 1.0], [4.7, 0.3]]])        # hinge, closed end, open end in the WC
    rooms = read_rooms(scene, keep, (-1, -1, 6, 4), swings=swing)
    areas = sorted(r["area_m2"] for r in rooms.rooms)
    assert len(areas) == 2 and 1.25 < areas[0] <= 1.45 and abs(areas[1] - 12.0) < 0.3     # pixel edges: within ~10 %


def test_door_swing_closes_a_doorway_too_wide_to_grow_shut():
    # 8 x 4 m, partition at x = 4 with a 1.2 m opening; the door hinges at (4, 1.5), closed leaf to (4, 2.7)
    walls = _box(0, 0, 8, 4) + [(4, 0, 4, 1.5), (4, 2.7, 4, 4)]
    scene, keep = _room_scene(walls)
    swing = np.array([[[4.0, 1.5], [4.0, 2.7], [2.8, 1.5]]])
    open_ = read_rooms(scene, keep, (-1, -1, 9, 5), swings=np.zeros((0, 3, 2)))
    assert len(open_.rooms) == 1
    rooms = read_rooms(scene, keep, (-1, -1, 9, 5), swings=swing)
    assert len(rooms.rooms) == 2 and all(abs(r["area_m2"] - 16) < 0.6 for r in rooms.rooms)
    inner = [d for d in rooms.doors if 0 not in d["rooms"]]
    assert len(inner) == 1 and abs(inner[0]["xy"][0] - 4.0) < 0.2           # the rooms still meet at the door


def test_named_rooms_part_at_an_opening_without_a_door():
    walls = _box(0, 0, 8, 4) + [(4, 0, 4, 1.2), (4, 2.8, 4, 4)]              # 1.6 m opening, no swing
    scene, keep = _room_scene(walls, [(1.5, 2.0, "YATAK ODASI"), (5.5, 2.0, "SALON")])
    rooms = read_rooms(scene, keep, (-1, -1, 9, 5), swings=np.zeros((0, 3, 2)))
    assert sorted(r["type"] for r in rooms.rooms) == ["bedroom", "living"]
    assert all(abs(r["area_m2"] - 16) < 1.5 for r in rooms.rooms)


def test_wardrobe_drawn_closed_is_the_rooms_floor():
    walls = _box(0, 0, 4, 4) + _box(-0.25, -0.25, 4.25, 4.25)                  # outer wall 25 cm thick
    walls += [(0.3, 4, 0.3, 3.4), (0.3, 3.4, 2.3, 3.4), (2.3, 3.4, 2.3, 4)]    # a 2 x 0.6 m wardrobe on it
    scene, keep = _room_scene(walls, [(1.5, 1.5, "YATAK ODASI 16.00 m²")])
    rooms = read_rooms(scene, keep, (-1, -1, 5, 5), swings=np.zeros((0, 3, 2)))
    assert len(rooms.rooms) == 1 and abs(rooms.rooms[0]["area_m2"] - 16) < 0.5
    assert rooms.rooms[0]["stamp_m2"] == 16.0


def test_area_check_divides_out_the_architects_convention():
    from cadai.placement.rooms import area_check
    rooms = [{"id": 1, "area_m2": 9.0, "stamp_m2": 10.0}, {"id": 2, "area_m2": 18.0, "stamp_m2": 20.0},
             {"id": 3, "area_m2": 6.0, "stamp_m2": 10.0}, {"id": 4, "area_m2": 30.0, "stamp_m2": None, "stamps": 2}]
    check = area_check(rooms)
    assert check["ratio"] == 0.9 and [o["id"] for o in check["off"]] == [3] and check["merged"] == [4]


def test_cleanup_check_reports_convention_and_merged_rooms():
    from cadai.check.cleanup_checks import check_cleanup
    walls = _box(0, 0, 4, 4) + _box(4, 0, 8, 4) + _box(0, 4, 8, 8)
    scene, keep = _room_scene(walls, [(1.5, 2, "ODA 17.60 m²"), (5.5, 2, "ODA 17.60 m²"),
                                      (1.5, 6, "SALON 15.00 m²"), (5.5, 6, "MUTFAK 15.00 m²")])
    codes = [i["code"] for i in check_cleanup(scene, keep)["issues"]]
    assert "check.clean.area_convention" in codes                  # 16 m² rooms written as 17.6
    assert codes.count("check.clean.rooms_merged") == 1           # the 8 x 4 m room holds two stamps
    assert "check.clean.area_mismatch" not in codes
