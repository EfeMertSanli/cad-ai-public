"""Furniture as loose lines and shape classes (ADR-025)."""
import json

import numpy as np

from cadai.placement import furniture, furniture_shapes


def _rect(x0, y0, w, h):
    return [(x0, y0, x0 + w, y0), (x0 + w, y0, x0 + w, y0 + h), (x0 + w, y0 + h, x0, y0 + h), (x0, y0 + h, x0, y0)]


def _bed(x0, y0, pillow_split=1):
    """A 1.6 x 2.0 m bed of loose lines: frame, a pillow line, pillows (split in `pillow_split` pieces)."""
    items = [(0, [s], None) for s in _rect(x0, y0, 1.6, 2.0)]
    items.append((0, [(x0, y0 + 1.6, x0 + 1.6, y0 + 1.6)], None))
    step = 1.6 / pillow_split
    items += [(0, [(x0 + k * step, y0 + 1.8, x0 + (k + 1) * step, y0 + 1.8)], None) for k in range(pillow_split)]
    return items


def test_loose_bed_is_one_piece_of_furniture():
    """Cevriye BERKOĞLU: the beds are polylines on an architectural layer, not blocks; they were
    invisible to the placement. Lines of another layer touching it and short scraps stay out."""
    from conftest import make_scene
    items = _bed(0, 0) + [(1, [(1.6, 0.5, 3.0, 0.5)], None)] + [(0, [(5, 5, 5.05, 5)], None)]
    scene = make_scene(items, layers=["MIMARI", "DUVAR"])
    found = furniture.loose_instances(scene, np.ones(scene.n, dtype=bool), with_shape=True)
    assert len(found) == 1
    f = found[0]
    assert f["sig"].startswith("~") and abs(f["short"] - 1.6) < 0.06 and abs(f["long"] - 2.0) < 0.06
    assert sorted(f["items"]) == list(range(6)) and f["shape"]["grid"].shape == (256,)


def test_door_leaf_with_its_swing_is_not_furniture():
    from conftest import make_scene
    scene = make_scene([(0, [s], None) for s in _rect(0, 0, 0.05, 0.9)] + [(0, [(0.05, 0.9, 0.9, 0.05)], None)])
    scene.etype[4] = len(scene.meta["types"])
    scene.meta["types"].append("ARC")
    scene.radius[4], scene.sweep[4] = 0.9, np.pi / 2
    assert furniture.loose_instances(scene, np.ones(scene.n, dtype=bool)) == []


def _pic(items):
    from conftest import make_scene
    scene = make_scene(items, layers=["MIMARI"])
    f = furniture.loose_instances(scene, np.ones(scene.n, dtype=bool), with_shape=True)[0]
    return f["sig"], f["shape"]


def test_the_same_bed_drawn_twice_is_one_class_a_wardrobe_another():
    beds = [_pic(_bed(0, 0, 1)), _pic(_bed(10, 0, 2))]                          # other segment count, other key
    wardrobe = _pic([(0, [s], None) for s in _rect(0, 0, 0.6, 2.0)] + [(0, [(0, 1.0, 0.6, 1.0)], None)])
    assert beds[0][0] != beds[1][0]
    pics = beds + [wardrobe]
    cls = furniture_shapes.group([p[0] for p in pics], np.array([p[1]["grid"] for p in pics]),
                                 np.array([p[1]["ratio"] for p in pics]), np.array([p[1]["size"] for p in pics]))
    assert cls[0] == cls[1] != cls[2]


def test_a_name_given_to_a_class_names_its_members_and_lookalikes(tmp_path, monkeypatch):
    monkeypatch.setattr(furniture, "TYPES", tmp_path / "types.json")
    monkeypatch.setattr(furniture, "FURNITURE_DIR", tmp_path)
    monkeypatch.setattr(furniture_shapes, "CLASSES", tmp_path / "classes.json")
    monkeypatch.setattr(furniture_shapes, "SHAPE_DIR", tmp_path)
    (a, sa), (b, sb) = _pic(_bed(0, 0, 1)), _pic(_bed(0, 0, 2))
    furniture_shapes.CLASSES.write_text(json.dumps([{"sig": a, "members": [a, b]}]), encoding="utf-8")
    furniture_shapes.save_pair("p__f", "p", "raw", [{"sig": a, "shape": sa, "x": 0, "y": 0, "short": 1.6, "long": 2,
                                                     "angle": 0}])
    furniture.set_type(a, "bed_double")
    assert furniture_shapes.resolved_types() == {a: "bed_double", b: "bed_double"}
    (c, sc) = _pic(_bed(0, 0, 3))                                               # a third drawing, never seen
    got = furniture_shapes.Matcher().types_for([{"sig": c, "shape": sc}])
    assert got == {c: "bed_double"}


def test_thumbnail_falls_back_to_another_drawing_when_one_cannot_be_loaded(tmp_path, monkeypatch):
    """One raw scene's texts are a fixed-width array that needed 9.4 GB: the class job died there."""
    from conftest import make_scene
    monkeypatch.setattr(furniture_shapes, "SHAPE_DIR", tmp_path)
    scene = make_scene(_bed(0, 0), layers=["MIMARI"])

    def load(raw):
        if raw == "huge":
            raise MemoryError
        return scene
    cls = {"id": "abc", "example": [("huge", 0.8, 1.0, 1.6, 2.0, 0.0), ("small", 0.8, 1.0, 1.6, 2.0, 0.0)]}
    furniture_shapes._thumbnails([cls], load=load)
    assert (tmp_path / "abc.png").exists()


def test_frame_around_a_room_name_is_kept_off_the_page():
    """The first classes of the page were room-name boxes: a rectangle with the name inside. Leaving
    them out of the data too also dropped bathroom fixtures (wet socket F1 0.450 -> 0.399), so they
    are only marked."""
    from conftest import make_scene
    items = [(0, [s], None) for s in _rect(0, 0, 1.2, 0.5)] + [(0, [(0.3, 0.2, 0.31, 0.21)], "SALON")]
    scene = make_scene(items, layers=["MIMARI"])
    found = furniture.loose_instances(scene, np.ones(scene.n, dtype=bool), with_shape=True)
    assert len(found) == 1 and found[0]["page"] == [False, True, False]
