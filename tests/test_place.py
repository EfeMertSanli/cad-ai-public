"""Placing devices on a cleaned drawing (ADR-023): door sides, empty floors, plan regions, kept items."""
import numpy as np

from cadai.placement import rooms as rooms_mod


def _two_rooms():
    """Room 1 (x 0-4) and room 2 (x 4-8), y 0-4, on the 5 cm raster; the wall between is x = 4."""
    h = w = int(8 / rooms_mod.CELL_M)
    labels = np.zeros((h, w), dtype=np.int32)
    labels[:, : w // 2 - 1] = 1
    labels[:, w // 2 + 1:] = 2
    labels[int(4 / rooms_mod.CELL_M):] = 0
    return rooms_mod.Rooms(labels, (0.0, 0.0), [])


def _one_room(size: float):
    """A size x size m room with a door to the outside in the middle of its west wall."""
    n = int(size / rooms_mod.CELL_M)
    labels = np.zeros((n + 2, n + 2), dtype=np.int32)
    labels[1:-1, 1:-1] = 1
    room = {"id": 1, "area_m2": size * size, "bbox": [0.05, 0.05, size + 0.05, size + 0.05],
            "centre": [size / 2, size / 2], "type": "bedroom"}
    return rooms_mod.Rooms(labels, (0.0, 0.0), [room], [{"rooms": [1, 0], "xy": [0.05, size / 2], "width_m": 0.9}])


def test_latch_is_the_arc_end_across_the_doorway():
    from cadai.placement.candidates import door_sides
    rm = _two_rooms()
    # hinge on the wall line x = 4 at y = 1; the door closes onto y = 1.9 (across the opening,
    # both ends on the wall line) and stands open into room 2 (leaf end at x = 4.9, in the room)
    swings = np.array([[[4.0, 1.0], [4.9, 1.0], [4.0, 1.9]]])
    walls = np.array([[4.0, y] for y in np.arange(0, 4, 0.05)] + [[4.9, 1.0]])   # the leaf rests on a wall too
    latch, hinge = door_sides(swings, walls, rm)
    assert np.allclose(latch, [[4.0, 1.9]])
    assert np.allclose(hinge, [[4.0, 1.0]])


def test_floor_without_devices_is_left_out():
    from cadai.placement.model import drawn_views
    dev = [{"type": "socket", "x": 0, "y": 0}, {"type": "light_ceiling", "x": 1, "y": 1}]
    rows = [{"project": "p", "final": "f", "view": 0, "type": "bedroom", "devices": dev},
            {"project": "p", "final": "f", "view": 0, "type": "living", "devices": dev},
            {"project": "p", "final": "f", "view": 1, "type": "bedroom", "devices": []},      # typical floor drawn once
            {"project": "p", "final": "f", "view": 1, "type": "living", "devices": []}]
    assert drawn_views(rows).tolist() == [True, True, False, False]


def test_plan_regions_one_per_drawing_cluster():
    from conftest import make_scene
    from cadai.placement.place import plan_regions

    def box(x0, y0, s):
        return [(x0, y0, x0 + s, y0), (x0 + s, y0, x0 + s, y0 + s), (x0 + s, y0 + s, x0, y0 + s), (x0, y0 + s, x0, y0)]
    items = [(0, [seg], None) for seg in box(0, 0, 12) + box(3, 3, 4) + box(40, 0, 10) + box(80, 0, 2)]
    scene = make_scene(items)
    regions = plan_regions(scene, np.ones(scene.n, dtype=bool))
    assert len(regions) == 2                                   # the 2 m box is too small for a plan
    assert regions[0][0] < 0 < 12 < regions[0][2]


def test_kept_items_found_again_in_the_cleaned_drawing():
    from conftest import make_scene
    from cadai.placement.place import kept_by_match
    raw = make_scene([(0, [(0, 0, 5, 0)], None), (0, [(0, 1, 5, 1), (5, 1, 5, 3)], None),
                      (0, [(9, 9, 9, 12)], None), (0, [(2, 2, 2.1, 2.1)], "SALON")])
    clean = make_scene([(0, [(5, 0, 0, 0)], None), (0, [(0, 1, 5, 1), (5, 1, 5, 3)], None),
                        (0, [(2, 2, 2.1, 2.1)], "SALON")])                  # reversed segment still matches
    assert kept_by_match(raw, clean).tolist() == [True, True, False, True]


def test_placer_uses_the_models_own_features(tmp_path):
    """A model trained before features were added keeps working (columns by name)."""
    import lightgbm as lgb
    from cadai.placement.candidates import FEATURES
    from cadai.placement.model import ROOM_FEATURES
    from cadai.placement.place import Placer
    rng = np.random.default_rng(0)
    old = FEATURES[:-2]
    X = rng.uniform(0, 5, (400, len(old)))
    y = (X[:, old.index("d_door")] < 1).astype(int)
    lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y, feature_name=old), 20).save_model(
        str(tmp_path / "rank_socket.txt"))
    R = rng.uniform(1, 20, (100, len(ROOM_FEATURES)))
    lgb.train({"objective": "poisson", "verbose": -1}, lgb.Dataset(R, np.ones(100) * 2, feature_name=ROOM_FEATURES),
              20).save_model(str(tmp_path / "count_socket.txt"))
    placer = Placer(tmp_path)
    Xn = rng.uniform(0, 5, (50, len(FEATURES)))
    Xn[:, FEATURES.index("on_wall")] = 1
    xy = rng.uniform(0, 10, (50, 2))
    out = placer.room(xy, Xn)
    assert out["socket"]["n"] == 2 and len(out["socket"]["pick"]) == 2


def test_each_light_wired_to_nearest_switch_of_its_room():
    from cadai.placement.place import light_links
    devs = [{"type": "switch", "x": 0.0, "y": 0.0}, {"type": "switch", "x": 4.0, "y": 0.0},
            {"type": "light_ceiling", "x": 1.0, "y": 2.0}, {"type": "light_ceiling", "x": 3.5, "y": 2.0},
            {"type": "socket", "x": 2.0, "y": 0.0}]
    assert light_links(devs) == [[1.0, 2.0, 0.0, 0.0], [3.5, 2.0, 4.0, 0.0]]
    assert light_links([{"type": "light_ceiling", "x": 1.0, "y": 1.0}]) == []      # no switch: no wire


def test_unsure_wet_socket_counts_round_up_earlier():
    from cadai.placement.model import device_counts as counts
    assert counts([0.35, 0.45, 1.4], "socket_wet").tolist() == [0, 1, 2]
    assert counts([0.35, 0.45, 1.4], "socket").tolist() == [0, 0, 1]


def test_wall_device_faces_into_its_room():
    from cadai.placement.place import facing
    rm = _two_rooms()
    assert facing(rm, 1, np.array([3.95, 2.0])) == 180.0          # room 1 lies to the west of the x = 4 wall
    assert facing(rm, 2, np.array([4.05, 2.0])) == 0.0


def test_unnamed_room_gets_a_type_from_its_geometry(tmp_path):
    """Big rooms named "living", small ones "bathroom": an unnamed big room is guessed living."""
    from cadai.placement.candidates import FEATURES
    from cadai.placement.model import train_room_types
    from cadai.placement.place import Placer
    rng = np.random.default_rng(0)
    rows, Xs, room = [], [], []
    for r in range(200):
        big = r % 2 == 0
        x = rng.uniform(0, 3, (10, len(FEATURES)))
        x[:, FEATURES.index("area")] = rng.uniform(18, 30) if big else rng.uniform(3, 6)
        rows.append({"type": "living" if big else "bathroom"})
        Xs.append(x)
        room += [r] * 10
    split = np.array(["train" if r < 160 else "test" for r in range(200)])
    acc = train_room_types(rows, np.vstack(Xs), np.array(room), split, tmp_path)
    assert acc > 0.9
    placer = Placer.__new__(Placer)
    import lightgbm as lgb
    placer.room_type = lgb.Booster(model_file=str(tmp_path / "room_type.txt"))
    x = rng.uniform(0, 3, (10, len(FEATURES)))
    x[:, FEATURES.index("area")] = 25.0
    assert placer.guess_type(x) == "living"


def test_candidates_know_their_place_in_their_own_room():
    """The wall point nearest the door is rank 0 whatever the room's size; the box position is
    mirror-free (ADR-026)."""
    from cadai.placement.candidates import FEATURES, room_candidates
    for size in (3.0, 6.0):
        rm = _one_room(size)
        room = rm.rooms[0]
        xy, X = room_candidates(rm, room, [], {})
        wall = X[:, FEATURES.index("on_wall")] == 1
        rank = X[:, FEATURES.index("rank_d_door")]
        d = X[:, FEATURES.index("d_door")]
        assert rank[wall][np.argmin(d[wall])] == 0 and rank[wall].max() == 1
        assert X[:, FEATURES.index("u_fold")].max() <= 0.5 + 1e-6


def test_rank_groups_one_per_room_and_no_giant_areas():
    from cadai.placement.model import RANK_MAX_CANDIDATES, rank_groups
    room = np.r_[np.full(3, 2), np.full(2, 0), np.full(RANK_MAX_CANDIDATES + 1, 1)]
    idx, groups = rank_groups(room, np.ones(len(room), dtype=bool))
    assert groups.tolist() == [2, 3] and room[idx].tolist() == [0, 0, 2, 2, 2]


def test_hall_points_know_the_bathroom_behind_the_door():
    """A bathroom's switch is drawn in the hall beside the bathroom door (ADR-027)."""
    from cadai.placement.candidates import FEATURES, ROOM_CODES, room_candidates
    rm = _two_rooms()
    rm.rooms = [{"id": 1, "area_m2": 15.0, "bbox": [0, 0, 3.9, 4], "centre": [2, 2], "type": "hall"},
                {"id": 2, "area_m2": 15.0, "bbox": [4.1, 0, 8, 4], "centre": [6, 2], "type": "bathroom"}]
    rm.doors = [{"rooms": [2, 1], "xy": [4.0, 2.0], "width_m": 0.9}, {"rooms": [1, 0], "xy": [0.0, 2.0], "width_m": 0.9}]
    xy, X = room_candidates(rm, rm.rooms[0], [], {})
    f = {n: X[:, FEATURES.index(n)] for n in ("nb_type", "d_wet_door", "d_out_door", "n_wet_doors", "n_out_doors")}
    by_wet = np.argmin(f["d_wet_door"])
    assert np.hypot(*(xy[by_wet] - (4.0, 2.0))) < 0.3 and f["nb_type"][by_wet] == ROOM_CODES["bathroom"]
    by_out = np.argmin(f["d_out_door"])
    assert f["nb_type"][by_out] == -1
    assert f["n_wet_doors"][0] == 1 and f["n_out_doors"][0] == 1


def test_instruction_rules_are_checked_and_applied():
    from cadai.placement import steer
    from cadai.placement.candidates import FEATURES
    answer = ('Kurallar: [{"class": "socket", "room_types": ["living", "uzay"], "min": 4, "max": null, "near": null},'
              ' {"class": "switch", "room_types": [], "min": null, "max": null, "near": "latch"},'
              ' {"class": "kahve", "min": 2}, {"class": "socket", "room_types": [], "min": null, "max": null, "near": "tavan"}]')
    rules = steer.parse(answer, ["socket", "switch"])
    assert rules == [{"class": "socket", "room_types": ["living"], "min": 4, "max": None, "near": None},
                     {"class": "switch", "room_types": [], "min": None, "max": None, "near": "latch"}]
    X = np.full((3, len(FEATURES)), 5.0)
    X[2, FEATURES.index("d_latch")] = 0.2
    scores = np.array([0.9, 0.5, 0.1])
    assert steer.apply(rules, "socket", "living", 2, X, scores)[0] == 4
    assert steer.apply(rules, "socket", "bedroom", 2, X, scores)[0] == 2
    n, s = steer.apply(rules, "switch", "bedroom", 1, X, scores)
    assert n == 1 and np.argmax(s) == 2                     # the latch-side point first


def test_rule_book_keeps_only_valid_rules():
    from cadai.placement import steer
    assert steer.load_book(["socket"]) == []
    kept = steer.save_book([{"class": "socket", "room_types": ["living"], "min": 4, "max": None, "near": None},
                            {"class": "socket", "room_types": [], "min": -1, "max": None, "near": None},
                            {"class": "lamba", "min": 1}], ["socket"])
    assert kept == [{"class": "socket", "room_types": ["living"], "min": 4, "max": None, "near": None}]
    assert steer.load_book(["socket"]) == kept


def test_steering_prompt_keeps_each_sentence_scope():
    """Cevriye: Haiku carried "salonlara" from the socket sentence over to the switch sentence."""
    from cadai.placement import steer
    assert "başka cümlenin kuralına taşınmaz" in steer.PROMPT
