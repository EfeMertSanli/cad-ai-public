import numpy as np

from cadai.cleanup.align import Transform
from cadai.placement.devices import inverse


def test_inverse_undoes_every_transform():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-50, 50, size=(20, 2))
    for rot in range(4):
        for scale in (1.0, 10.0, 0.1):
            t = Transform(rot, scale, 12.5, -7.25)
            back = inverse({"rot": rot, "scale": scale, "tx": 12.5, "ty": -7.25}, t.apply(pts))
            assert np.allclose(back, pts)


def test_exploded_symbol_found_in_any_rotation():
    from cadai.placement.symbols import exploded_instances
    from conftest import make_scene

    def symbol(cx, cy, angle):
        # a 30 x 10 cm "switch": a box with a diagonal, rotated
        pts = np.array([(-0.15, -0.05), (0.15, -0.05), (0.15, 0.05), (-0.15, 0.05)])
        c, s = np.cos(angle), np.sin(angle)
        p = pts @ np.array([[c, s], [-s, c]]) + (cx, cy)
        return [(p[i][0], p[i][1], p[(i + 1) % 4][0], p[(i + 1) % 4][1]) for i in range(4)] + \
               [(p[0][0], p[0][1], p[2][0], p[2][1])]

    items = []
    for k, ang in enumerate((0.0, 0.7, np.pi / 2)):
        items += [(0, [seg], None) for seg in symbol(5.0 * k, 0.0, ang)]
    items.append((0, [(0, 3, 9, 3)], None))                          # a long wire: not a symbol part
    scene = make_scene(items, layers=["m-LİNYE"])
    found = exploded_instances(scene)
    assert len(found) == 3
    assert len({f["block"] for f in found}) == 1                      # one key for all rotations
    assert all(f["block"].startswith("~") for f in found)


def test_exploded_symbol_touching_another_layers_lines_stays_a_symbol():
    """An archive project: switches whose stem touched weak-current and emergency lines (other layers)
    merged with them into a 2 m blob and were dropped as too big; a symbol is drawn on one layer."""
    from cadai.placement.symbols import exploded_instances
    from conftest import make_scene
    switch = [(0, [(0.0, 0.0, 0.3, 0.0)], None), (0, [(0.0, 0.0, 0.0, 0.1)], None)]
    other = [(1, [(0.3 + 0.4 * k, 0.0, 0.7 + 0.4 * k, 0.0)], None) for k in range(5)]   # 2 m of short lines
    found = exploded_instances(make_scene(switch + other, layers=["m-LİNYE", "m-zayıfakım"]))
    assert any(f["items"] == [0, 1] for f in found)


def test_place_room_keeps_spacing_and_count():
    from cadai.placement.model import _match, place_room
    xy = np.array([[0, 0], [0.2, 0], [1.0, 0], [2.0, 0]], dtype=float)
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    pick = place_room(scores, xy, 2)
    assert pick.tolist() == [0, 2]                                   # 0.2 m from the best is too close
    assert _match(xy[pick], np.array([[0.1, 0.0], [5.0, 5.0]])) == 1


def test_device_takes_the_view_whose_lines_match_and_repeated_floors_count_once(monkeypatch):
    """Ali Kemal ARSLAN: one view's region covered two floors, so devices of floor B went through floor
    A's transform and landed between the plans; four alike floors put every device on the typical
    plan four times. Now the engineer's lines around a device decide the view."""
    from cadai.placement import devices, symbols
    from conftest import make_scene

    def room(x0, y0, w=6.0, h=4.0):                     # a room outline as short pieces
        segs = []
        for (a, b), (c, d) in (((0, 0), (w, 0)), ((w, 0), (w, h)), ((w, h), (0, h)), ((0, h), (0, 0))):
            n = 12
            for i in range(n):
                segs.append((x0 + a + (c - a) * i / n, y0 + b + (d - b) * i / n,
                             x0 + a + (c - a) * (i + 1) / n, y0 + b + (d - b) * (i + 1) / n))
        return segs

    # raw: plan A at x=0, plan B (other shape) at x=20
    raw = make_scene([(0, [s], None) for s in room(0, 0) + room(20, 0, 3.0, 5.0)])
    # final: the engineer's copies, A twice (two floors) shifted by 100 / 200, B shifted by 300
    final = make_scene([(0, [s], None) for s in room(100, 0) + room(200, 0) + room(320, 0, 3.0, 5.0)])
    transforms = [{"rot": 0, "scale": 1.0, "tx": 100, "ty": 0, "region": [-1, -1, 30, 6]},   # region covers A and B
                  {"rot": 0, "scale": 1.0, "tx": 200, "ty": 0, "region": [-1, -1, 7, 5]},
                  {"rot": 0, "scale": 1.0, "tx": 300, "ty": 0, "region": [19, -1, 24, 6]}]
    insts = [{"block": "S", "top": -1, "x": 103.0, "y": 2.0, "size": 0.2},    # floor 1 of A
             {"block": "S", "top": -1, "x": 203.0, "y": 2.0, "size": 0.2},    # floor 2 of A: same spot
             {"block": "S", "top": -1, "x": 321.5, "y": 2.5, "size": 0.2},    # B
             {"block": "S", "top": -1, "x": 500.0, "y": 50.0, "size": 0.2}]   # a riser diagram: no plan
    monkeypatch.setattr(symbols, "instances", lambda scene: insts)
    out = devices.locate(final, transforms, {"S": "socket"}, raw)
    got = sorted((round(d["x"], 1), round(d["y"], 1)) for d in out)
    assert got == [(3.0, 2.0), (21.5, 2.5)]
