import ezdxf
import numpy as np

from cadai.cad.scene import build
from cadai.cleanup.apply import MTEXT_COLOUR, OUTLINE, write_clean
from cadai.cleanup.finish import finish, is_room_name


def test_room_names():
    for yes in ("Salon", "E.Yatak Odası", "Kat Holu", "Asansör", "Yatak Odası 2", "MUTFAK"):
        assert is_room_name(yes), yes
    for no in ("18.03 m²", "±0.00", "Seramik", "315", "A-A KESİTİ ÖLÇEK 1/50 ve çok uzun bir yazı"):
        assert not is_room_name(no), no


def _drawing(tmp_path):
    doc = ezdxf.new(setup=True)                       # defines the DASHED linetype
    for name, colour, lt in (("DUVAR", 5, "CONTINUOUS"), ("KIRIS", 3, "DASHED"), ("MERDIVEN", 2, "DASHED"),
                             ("YAZI", 1, "CONTINUOUS"), ("KOLON", 4, "CONTINUOUS"), ("TARAMA", 6, "CONTINUOUS"),
                             ("OLCULENDIRME", 3, "CONTINUOUS"), ("KUTUPHANE", 2, "CONTINUOUS")):
        doc.layers.add(name, color=colour, linetype=lt)
    msp = doc.modelspace()
    for a, b in (((0, 0), (500, 0)), ((500, 0), (500, 300)), ((500, 300), (0, 300)), ((0, 300), (0, 0))):
        msp.add_line(a, b, dxfattribs={"layer": "DUVAR", "color": 4})                # a 5 x 3 m room (cm)
    msp.add_line((0, 100), (500, 100), dxfattribs={"layer": "KIRIS"})                # dashed beam: goes
    msp.add_line((0, 200), (500, 200), dxfattribs={"layer": "MERDIVEN"})             # dashed stair: stays
    msp.add_line((-60, -40), (-60, -10), dxfattribs={"layer": "OLCULENDIRME"})       # dimension tick: goes
    msp.add_text("Salon", dxfattribs={"layer": "YAZI", "height": 20}).set_placement((100, 50))
    msp.add_text("Salon", dxfattribs={"layer": "YAZI", "height": 10}).set_placement((120, 150))  # repeat: goes
    msp.add_text("18.03 m²", dxfattribs={"layer": "YAZI", "height": 20}).set_placement((100, 20))
    msp.add_text("Kat Holu", dxfattribs={"layer": "YAZI", "height": 20}).set_placement((900, 900))  # in no room
    m = msp.add_mtext("\\C1;Mutfak", dxfattribs={"layer": "YAZI", "char_height": 20})
    m.set_location((300, 50))                                                          # 2nd name in the room: stays
    h = msp.add_hatch(color=1, dxfattribs={"layer": "KOLON"})
    h.paths.add_polyline_path([(460, 260), (490, 260), (490, 290), (460, 290)])         # column: -> outline
    h = msp.add_hatch(color=1, dxfattribs={"layer": "TARAMA"})
    h.paths.add_polyline_path([(100, 120), (400, 120), (400, 190), (100, 190)])         # floor tiles: go
    h = msp.add_hatch(color=1, dxfattribs={"layer": "TARAMA"})
    h.paths.add_polyline_path([(420, 120), (450, 120), (450, 150), (420, 150)])         # 30 cm shaft square: stays
    shaft = doc.blocks.new("SHAFT")                                                    # two shaft boxes with diagonals
    for x in (0, 30):
        shaft.add_lwpolyline([(x, 0), (x + 25, 0), (x + 25, 25), (x, 25)], close=True)
        shaft.add_line((x, 0), (x + 25, 25))
    msp.add_blockref("SHAFT", (20, 260), dxfattribs={"layer": "KUTUPHANE"})
    for x in (300, 325):                                                               # the same, exploded, side by side
        msp.add_lwpolyline([(x, 230), (x + 25, 230), (x + 25, 255), (x, 255)], close=True, dxfattribs={"layer": "KUTUPHANE"})
        msp.add_line((x, 230), (x + 25, 255), dxfattribs={"layer": "KUTUPHANE"})
    src = tmp_path / "in.dxf"
    doc.saveas(src)
    return src


def test_finish_rules(tmp_path):
    scene = build(_drawing(tmp_path))
    layers = np.array([scene.meta["layers"][i] for i in scene.layer])
    model = np.where(np.isin(layers, ["DUVAR", "KIRIS", "MERDIVEN", "OLCULENDIRME", "KOLON", "TARAMA"]), 1, 0).astype(np.int8)
    out, stats = finish(scene, model, np.ones(scene.n, dtype=bool))
    kept_names = sorted(t for t, o in zip(scene.text, out) if t and o == 1)
    assert kept_names == ["Mutfak", "Salon"]
    assert stats["dashed_deleted"] == 1 and stats["annotation_deleted"] == 1
    assert stats["hatches_deleted"] == 1 and stats["hatches_outlined"] == 1
    assert (out[layers == "KOLON"] == OUTLINE).all()
    assert (out[layers == "KUTUPHANE"] == 1).all()                                     # the shaft stays
    tarama = np.flatnonzero(layers == "TARAMA")
    assert sorted(out[tarama].tolist()) == [0, 1]                                      # tiles go, shaft square stays


def test_write_white_cm_and_outline(tmp_path):
    src = _drawing(tmp_path)
    scene = build(src, unit={"unit": "m", "scale": 1.0, "confidence": 1, "votes": {}})   # pretend metres
    layers = np.array([scene.meta["layers"][i] for i in scene.layer])
    decision = np.where(layers == "KOLON", OUTLINE, np.where(layers == "DUVAR", 1, 0)).astype(np.int8)
    stats = write_clean(src, scene, decision, tmp_path / "out.dxf", plain=True, to_cm=True)
    clean = ezdxf.readfile(tmp_path / "out.dxf")
    kept = list(clean.modelspace())
    assert stats["outlined"] == 1
    assert sorted(e.dxftype() for e in kept if e.dxf.layer == "KOLON") == ["LWPOLYLINE"]   # outline, no fill
    assert clean.header["$INSUNITS"] == 5
    wall = [e for e in kept if e.dxf.layer == "DUVAR"][0]
    assert abs(max(abs(wall.dxf.end.x), abs(wall.dxf.start.x)) - 50000) < 1e-6        # 500 m -> 50000 cm
    assert all(e.dxf.color == 256 for e in kept) and all(lay.dxf.color == 7 for lay in clean.layers)


def test_mtext_colour_codes():
    assert MTEXT_COLOUR.sub("", "\\C1;Mutfak\\c16711680; 2") == "Mutfak 2"


def test_window_tick_outside_the_wall_goes(tmp_path):
    from cadai.cleanup.finish import opening_ticks
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 5                                                        # centimetres
    doc.layers.add("DUVAR")
    doc.layers.add("PENCERE")
    msp = doc.modelspace()
    msp.add_line((0, 0), (500, 0), dxfattribs={"layer": "DUVAR"})                    # wall faces 25 cm apart
    msp.add_line((0, 25), (500, 25), dxfattribs={"layer": "DUVAR"})
    msp.add_line((200, 0), (200, -30), dxfattribs={"layer": "PENCERE"})              # sill tick sticking out: goes
    msp.add_line((150, 0), (150, 25), dxfattribs={"layer": "PENCERE"})               # frame across the wall: stays
    msp.add_line((100, 5), (300, 5), dxfattribs={"layer": "PENCERE"})                # glass line, long: stays
    doc.saveas(tmp_path / "w.dxf")
    scene = build(tmp_path / "w.dxf")
    layers = np.array([scene.meta["layers"][i] for i in scene.layer])
    ticks = opening_ticks(scene, np.ones(scene.n, dtype=np.int8), np.ones(scene.n, dtype=bool))
    assert ticks.tolist() == [False, False, True, False, False]
    assert (layers[ticks] == "PENCERE").all()
