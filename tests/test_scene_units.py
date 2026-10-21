import ezdxf
import numpy as np

from cadai.cad.scene import Scene, build
from cadai.cad.units import infer_unit


def test_door_arcs_beat_a_wrong_header():
    # 90-degree arcs of radius 90 in a file that claims millimetres: centimetres.
    res = infer_unit([(90.0, 90.0)] * 20, [], insunits=4)
    assert res["unit"] == "cm"


def test_no_evidence_falls_back_to_cm():
    assert infer_unit([], [], 0)["unit"] == "cm"


def test_build_explodes_blocks_and_inherits_layer(tmp_path):
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 5
    doc.layers.add("DUVAR")
    doc.layers.add("KAPI")
    blk = doc.blocks.new("DOOR")
    blk.add_line((0, 0), (90, 0), dxfattribs={"layer": "0"})
    blk.add_arc((0, 0), 90, 0, 90, dxfattribs={"layer": "0"})
    msp = doc.modelspace()
    msp.add_line((0, 0), (500, 0), dxfattribs={"layer": "DUVAR"})
    msp.add_blockref("DOOR", (100, 0), dxfattribs={"layer": "KAPI"})
    msp.add_text("SALON", dxfattribs={"layer": "DUVAR", "height": 20}).set_placement((200, 200))
    path = tmp_path / "t.dxf"
    doc.saveas(path)

    s = build(path)
    assert s.meta["unit"]["unit"] == "cm"
    assert s.n == 4                                   # line + 2 block children + text
    layers = [s.meta["layers"][i] for i in s.layer]
    assert layers.count("KAPI") == 2                  # layer "0" children take the INSERT's layer
    assert all(p.startswith(s.meta["top_handles"][1]) for p, t in zip(s.path, s.top) if t == 1)
    assert abs(s.lengths()[0] - 5.0) < 1e-6           # 500 cm line = 5 m
    assert "SALON" in list(s.text)

    s.save(tmp_path / "t.npz")
    again = Scene.load(tmp_path / "t.npz")
    assert again.n == s.n and np.allclose(again.segs, s.segs)
