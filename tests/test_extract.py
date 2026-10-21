import ezdxf
import numpy as np

from cadai.cad.scene import build
from cadai.cleanup import elements, extract, layers
from cadai.cleanup.apply import OUTLINE, write_clean


def test_layer_types_from_names():
    cases = {"Duvar": "wall", "Dış Duvar_Pen_No__21": "wall", "KOLON_Pen_No__101": "column", "A-WALL": "wall",
             "Kapi_Pencere": "opening", "Archicad Doors_Pen_No__3": "door", "A-GLAZ": "window",
             "Merdiven": "stair", "Korkuluk": "railing", "Kutuphane": "drop", "Olculendirme": "drop",
             "Tarama": "drop", "Archicad Door Markers_Pen_No__5": "drop", "0_Pen_No__91": "unknown",
             "sketch_Pen_No__101": "unknown", "döşeme_Pen_No__89": "unknown", "Mahal": "room_name",
             "S-STRS": "stair", "S-STRS-MBND": "stair"}       # AIA stairs (Ali Kemal ARSLAN: treads were judged as unknown)
    for name, kind in cases.items():
        assert layers.type_of(name) == kind, name


def _door_drawing(tmp_path):
    """A wall layer, an exploded door (arc as 5 cm pieces, leaf as two lines) and a window, in cm."""
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 5
    for name in ("Duvar", "Kapi_Pencere", "Kolon"):
        doc.layers.add(name)
    msp = doc.modelspace()
    msp.add_line((0, 0), (300, 0), dxfattribs={"layer": "Duvar"})
    msp.add_line((0, 20), (300, 20), dxfattribs={"layer": "Duvar"})
    angles = np.radians(np.linspace(0, 90, 30))
    pts = np.column_stack([100 + 90 * np.cos(angles), 20 + 90 * np.sin(angles)])     # hinge at (100, 20), r 90 cm
    for a, b in zip(pts[:-1], pts[1:]):
        msp.add_lwpolyline([a, b], dxfattribs={"layer": "Kapi_Pencere"})
    msp.add_line((100, 20), (100, 110), dxfattribs={"layer": "Kapi_Pencere"})         # open leaf
    msp.add_line((104, 20), (104, 110), dxfattribs={"layer": "Kapi_Pencere"})
    msp.add_line((220, 5), (280, 5), dxfattribs={"layer": "Kapi_Pencere"})            # window glass in the wall
    msp.add_line((220, 15), (280, 15), dxfattribs={"layer": "Kapi_Pencere"})
    for k in range(8):                                                                # exploded column hatching
        msp.add_line((10 + 4 * k, 0), (18 + 4 * k, 8), dxfattribs={"layer": "Kolon"})
    msp.add_lwpolyline([(10, 0), (50, 0), (50, 20), (10, 20)], close=True, dxfattribs={"layer": "Kolon"})
    doc.saveas(tmp_path / "door.dxf")
    return build(tmp_path / "door.dxf")


def test_exploded_door_swing_and_leaf_go_window_and_walls_stay(tmp_path):
    scene = _door_drawing(tmp_path)
    keep, stats = elements.select(scene, layers.classify(scene)[0], np.ones(scene.n, bool), np.zeros(scene.n, bool))
    names = np.array(scene.meta["layers"])[scene.layer]
    length = scene.lengths()
    door = names == "Kapi_Pencere"
    assert keep[names == "Duvar"].all()
    assert not keep[door & (length < 0.1)].any()                                     # swing pieces
    assert not keep[door & (length > 0.85)].any()                                    # leaf lines (90 cm)
    assert keep[door & (np.abs(length - 0.6) < 0.01)].all()                          # window glass (60 cm)
    kolon = names == "Kolon"
    assert keep[kolon & scene.closed].all() and not keep[kolon & ~scene.closed].any()  # outline stays, hatching goes
    assert stats["door_swings"] >= 30 and stats["hatch_lines"] == 8


def test_engineer_dictionary_wins_over_name_rules(tmp_path):
    scene = _door_drawing(tmp_path)
    kinds, sources = layers.classify(scene)
    assert kinds[scene.meta["layers"].index("Kolon")] == "column"
    layers.confirm({"Kolon_Pen_No__7": "drop"})                       # another pen of it: not this layer
    assert layers.classify(scene)[0][scene.meta["layers"].index("Kolon")] == "column"
    layers.confirm({"KOLON": "drop"})                                   # case and accents don't matter
    kinds, sources = layers.classify(scene)
    i = scene.meta["layers"].index("Kolon")
    assert kinds[i] == "drop" and sources[i] == "engineer"


def test_mode_by_layers_or_items(tmp_path):
    scene = _door_drawing(tmp_path)
    plan = np.ones(scene.n, bool)
    assert extract.choose_mode(scene, layers.classify(scene)[0], plan)[0] == "layers"
    one_layer = ["unknown"] * len(scene.meta["layers"])                 # everything on "0"
    assert extract.choose_mode(scene, one_layer, plan)[0] == "items"


def test_layer_mode_keeps_originals_plain_and_outlines_column_hatch(tmp_path):
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 5
    doc.layers.add("Duvar", color=3, lineweight=50)
    doc.layers.add("Kolon", color=1)
    msp = doc.modelspace()
    msp.add_line((0, 0), (500, 0), dxfattribs={"layer": "Duvar", "lineweight": 70})
    msp.add_arc((250, 0), 40, 0, 180, dxfattribs={"layer": "Duvar"})            # a curved wall stays an ARC
    h = msp.add_hatch(color=1, dxfattribs={"layer": "Kolon"})
    h.paths.add_polyline_path([(0, 0), (30, 0), (30, 30), (0, 30)])
    doc.saveas(tmp_path / "in.dxf")
    scene = build(tmp_path / "in.dxf")
    kinds = layers.classify(scene)[0]
    decision, stats = extract.decide(scene, kinds, np.ones(scene.n, bool), np.zeros(scene.n, np.int8), "layers")
    assert stats["hatches_outlined"] == 1 and (decision[np.array(scene.meta["types"])[scene.etype] == "HATCH"] == OUTLINE).all()
    write_clean(tmp_path / "in.dxf", scene, decision, tmp_path / "out.dxf", plain=True, to_cm=True)
    out = ezdxf.readfile(tmp_path / "out.dxf")
    kinds_out = sorted(e.dxftype() for e in out.modelspace())
    assert kinds_out == ["ARC", "LINE", "LWPOLYLINE"]                             # original entities, hatch -> outline
    assert out.header["$LWDISPLAY"] == 0
    assert all(e.dxf.lineweight == -1 for e in out.modelspace())
    assert all(lay.dxf.lineweight == -3 and lay.dxf.color == 7 for lay in out.layers)
