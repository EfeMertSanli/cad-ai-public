"""The cleanup toolbox, the region finder and the Claude procedure (ADR-021), on small drawings."""
import json

import ezdxf
import numpy as np

from cadai.cad.scene import build
from cadai.cleanup import agent, regions, tools
from cadai.placement import furniture


def _flat(tmp_path):
    """Walls, a door swing, a bed block, a dimension text and a hatch-pattern layer, in cm."""
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 5
    for name in ("Duvar", "Kapi", "A$C2AC03", "Olcu", "Tarama"):
        doc.layers.add(name)
    bed = doc.blocks.new("GVXVVS")                                        # meaningless block name
    bed.add_lwpolyline([(0, 0), (160, 0), (160, 200), (0, 200)], close=True)
    bed.add_line((0, 40), (160, 40))
    msp = doc.modelspace()
    for y in (0, 20, 500, 520):
        msp.add_line((0, y), (600, y), dxfattribs={"layer": "Duvar"})
    msp.add_arc((100, 20), 90, 0, 90, dxfattribs={"layer": "Kapi"})
    msp.add_line((100, 20), (100, 110), dxfattribs={"layer": "Kapi"})
    msp.add_blockref("GVXVVS", (300, 100), dxfattribs={"layer": "A$C2AC03"})
    msp.add_text("3.40", dxfattribs={"layer": "Olcu", "height": 15}).set_placement((300, 30))
    for k in range(6):
        msp.add_line((400 + 10 * k, 100), (420 + 10 * k, 140), dxfattribs={"layer": "Tarama"})
    doc.saveas(tmp_path / "flat.dxf")
    return build(tmp_path / "flat.dxf")


def test_every_tool_finds_its_kind_and_walls_stay(tmp_path, monkeypatch):
    scene = _flat(tmp_path)
    plan = np.ones(scene.n, bool)
    sig = [i["sig"] for i in furniture.instances(scene, np.ones(scene.n, np.int8))]
    monkeypatch.setattr(furniture, "delete_signatures", lambda: set(sig))     # "engineers deleted this shape"
    r = tools.run(scene, plan, np.zeros(scene.n))
    names = np.array(scene.meta["layers"])[scene.layer]
    types = np.array(scene.meta["types"])[scene.etype]
    kept = r.decision != tools.DELETE
    assert kept[names == "Duvar"].all()
    assert r.found["door_swings"][(names == "Kapi") & (types == "ARC")].all()
    assert r.found["furniture"][scene.block_top >= 0].all()
    assert r.found["annotations"][scene.text == "3.40"].all()
    assert r.found["off_layers"][names == "Tarama"].all()
    assert not kept[names != "Duvar"].any()
    # switching a tool off brings its finds back (the layer itself is of unknown type here)
    r2 = tools.run(scene, plan, np.zeros(scene.n), disabled={"furniture"})
    assert (r2.decision[scene.block_top >= 0] != tools.DELETE).all()


def test_everything_outside_the_plans_goes(tmp_path):
    scene = _flat(tmp_path)
    plan = np.zeros(scene.n, bool)
    r = tools.run(scene, plan, np.zeros(scene.n))
    assert (r.decision == tools.DELETE).all() and (r.attribution == -2).all()


def _two_views(tmp_path):
    """A plan (room grid) and a section 40 m apart, a long frame line touching both, titles below."""
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 6                                            # metres
    msp = doc.modelspace()
    for x0, title in ((0, "ZEMIN KAT PLANI"), (40, "A-A KESITI")):
        for i in range(11):                                                # 1 m cells of short lines
            for j in range(10):
                msp.add_line((x0 + i, j), (x0 + i + 1, j))
                msp.add_line((x0 + i, j), (x0 + i, j + 1))
        msp.add_text(title, dxfattribs={"height": 0.5}).set_placement((x0 + 2, -1.5))
    msp.add_line((-2, -3), (60, -3))                                       # sheet frame
    doc.saveas(tmp_path / "views.dxf")
    return build(tmp_path / "views.dxf")


def _no_features(scene):
    names = list(regions.GLUE_FEATURES)
    return np.zeros((scene.n, len(names))), names


def test_regions_are_split_and_titled(tmp_path):
    scene = _two_views(tmp_path)
    item_region, regs = regions.find(scene, np.ones(scene.n), *_no_features(scene))
    assert len(regs) == 2
    assert sorted(r["titles"][0] for r in regs) == ["A-A KESITI", "ZEMIN KAT PLANI"]
    frame = scene.lengths() > 50
    assert (item_region[frame] >= -1).all()                                # the frame doesn't glue them


def test_claude_picks_the_plan_and_its_review_is_applied(tmp_path, monkeypatch):
    scene = _two_views(tmp_path)
    X, names = _no_features(scene)
    _, regs = regions.find(scene, np.ones(scene.n), X, names)
    plan_n = next(n for n, r in enumerate(regs) if r["titles"][0] == "ZEMIN KAT PLANI")
    answers = {"views": json.dumps([{"bölge": n, "plan": n == plan_n, "ne": "x"} for n in range(len(regs))]),
               "review": json.dumps({"disable": ["model"], "layer_types": {}, "add_regions": [], "drop_regions": [],
                                     "notes": "ok"})}
    monkeypatch.setattr(agent, "call", lambda prompt, workdir, tag: (answers[tag], {"step": tag, "cost_usd": 0.01,
                                                                                    "seconds": 1, "is_error": False}))
    result, in_plan, record = agent.clean(scene, np.ones(scene.n), np.ones(scene.n), X, names, tmp_path / "out")
    x = (scene.segs[:, 0] + scene.segs[:, 2]) / 2
    item_x = np.bincount(scene.seg_item, weights=x, minlength=scene.n) / np.maximum(np.bincount(scene.seg_item, minlength=scene.n), 1)
    lines = scene.text == ""
    assert in_plan[lines & (item_x < 12)].all() and not in_plan[lines & (item_x > 38) & (item_x < 52)].any()
    assert record["views"] == "claude" and record["fixes"]["disable"] == ["model"]
    assert not result.stats["tools"]["model"]["enabled"]
    assert (result.decision[lines & (item_x < 12)] != tools.DELETE).all()   # the model tool (p(delete)=1) is off
    assert (tmp_path / "out" / "views.png").exists() and (tmp_path / "out" / "tools.png").exists()


def test_without_claude_the_statistical_views_decide(tmp_path, monkeypatch):
    scene = _two_views(tmp_path)
    X, names = _no_features(scene)
    monkeypatch.setattr(agent, "call", lambda *a: ("", {"step": "views", "cost_usd": 0, "seconds": 0, "is_error": True}))
    p_view = np.full(scene.n, 0.7)
    _, in_plan, record = agent.clean(scene, p_view, np.zeros(scene.n), X, names, tmp_path / "out")
    assert record["views"] == {"tr": "istatistik model", "en": "statistical model"} and in_plan.all()


def test_dense_hatching_is_found_without_an_all_pairs_search():
    """60 000 short parallel lines on 10 m2 used to run out of memory (query_ball_point lists)."""
    from cadai.cleanup import elements
    from tests.conftest import make_scene
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 3, 60_000), rng.uniform(0, 3, 60_000)
    items = [(0, [(a, b, a + 0.2, b + 0.2)], None) for a, b in zip(x, y)]
    scene = make_scene(items)
    found = elements.hatch_lines(scene, np.ones(scene.n, bool))
    assert found.mean() > 0.99


def test_columns_drawn_as_one_fill_on_a_detail_layer_keep_their_outline(tmp_path):
    """Ali Kemal ARSLAN: every column and shear wall of a floor is one filled region on "A-DETL-GENF"
    (a dropped layer). The fill goes, the outlines stay; a room-sized floor pattern still goes."""
    doc = ezdxf.new()
    doc.header["$INSUNITS"] = 6                                            # metres
    doc.layers.add("A-DETL-GENF")
    doc.layers.add("A-FLOR-PATT")
    msp = doc.modelspace()
    cols = msp.add_hatch(dxfattribs={"layer": "A-DETL-GENF"})
    for x in (0, 5, 10):
        cols.paths.add_polyline_path([(x, 0), (x + 0.3, 0), (x + 0.3, 0.5), (x, 0.5)], is_closed=True)
    cols.paths.add_polyline_path([(0, 6), (4, 6), (4, 6.25), (0, 6.25)], is_closed=True)   # shear wall
    floor = msp.add_hatch(dxfattribs={"layer": "A-FLOR-PATT"})
    floor.paths.add_polyline_path([(1, 1), (4.5, 1), (4.5, 5), (1, 5)], is_closed=True)
    msp.add_line((0, 0.25), (12, 0.25))
    doc.saveas(tmp_path / "cols.dxf")
    scene = build(tmp_path / "cols.dxf")
    r = tools.run(scene, np.ones(scene.n, bool), np.zeros(scene.n))
    names = np.array(scene.meta["layers"])[scene.layer]
    assert (r.decision[names == "A-DETL-GENF"] == tools.OUTLINE).all()
    assert (r.decision[names == "A-FLOR-PATT"] == tools.DELETE).all()


def test_balcony_parapets_on_aia_handrail_layer_stay():
    from cadai.cleanup import layers
    assert layers.type_of("A-FLOR-HRAL") == "railing"               # Ali Kemal ARSLAN: balcony outer walls
    assert layers.type_of("A-FLOR-PATT") == "drop" and layers.type_of("A-FLOR-FNSH") == "drop"
