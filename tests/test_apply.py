import ezdxf
import numpy as np

from cadai.cad.scene import build
from cadai.cleanup.apply import DELETE, KEEP, QUESTION, write_clean


def test_write_clean_deletes_and_explodes_mixed_blocks(tmp_path):
    doc = ezdxf.new()
    blk = doc.blocks.new("DOOR")
    blk.add_line((0, 0), (90, 0), dxfattribs={"layer": "FRAME"})     # frame: keep
    blk.add_arc((0, 0), 90, 0, 90, dxfattribs={"layer": "SWING"})      # swing: delete
    blk.add_attdef("NO", (0, -10))
    msp = doc.modelspace()
    wall = msp.add_line((0, 0), (500, 0), dxfattribs={"layer": "DUVAR"})
    msp.add_line((0, -50), (500, -50), dxfattribs={"layer": "OLCU"})   # dimension line: delete
    door = msp.add_blockref("DOOR", (100, 0))
    door.add_auto_attribs({"NO": "K1"})                                  # attribute: delete
    msp.add_line((0, 300), (10, 300), dxfattribs={"layer": "X"})         # question: kept
    src = tmp_path / "in.dxf"
    doc.saveas(src)

    scene = build(src)
    by_path = {"DUVAR": KEEP, "OLCU": DELETE, "FRAME": KEEP, "SWING": DELETE, "X": QUESTION}
    decision = []
    for path, layer in zip(scene.path, scene.layer):
        name = scene.meta["layers"][layer]
        decision.append(by_path.get(name, DELETE))                       # the attribute sits on "0" -> delete
    stats = write_clean(src, scene, np.array(decision, dtype=np.int8), tmp_path / "out.dxf")

    out = ezdxf.readfile(tmp_path / "out.dxf").modelspace()
    layers = sorted(e.dxf.layer for e in out)
    assert layers == ["DUVAR", "FRAME", "X"]
    assert stats["exploded"] == 1
    assert any(e.dxf.handle == wall.dxf.handle for e in out)             # untouched entities keep their handle
