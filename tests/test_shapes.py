"""Exploded symbols matched by shape (ADR-024)."""
import numpy as np

from cadai.placement import shapes


def _switch(n_circle=36, stem=0.12, turn=0.0, mirror=False):
    """Points of a lollipop switch: a 6 cm circle and a stem, every 2 cm, then turned / mirrored."""
    a = np.linspace(0, 2 * np.pi, n_circle)
    circle = 0.03 * np.column_stack([np.cos(a), np.sin(a)])
    t = np.linspace(0, 1, 12)[:, None]
    pts = np.vstack([circle, np.array([0.03, 0.0]) + t * np.array([stem, stem * 0.6])])
    if mirror:
        pts[:, 1] *= -1
    r = np.radians(turn)
    return pts @ np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]]) + (10.0, 5.0)


def _socket():
    """A half circle with a bar across: same size class, different drawing."""
    a = np.linspace(0, np.pi, 30)
    arc = 0.06 * np.column_stack([np.cos(a), np.sin(a)])
    bar = np.column_stack([np.linspace(-0.08, 0.08, 15), np.zeros(15)])
    return np.vstack([arc, bar])


def _dist(p, q, cp=1, cq=1):
    gp, rp, sp = shapes.picture(p)
    gq, rq, sq = shapes.picture(q)
    return float(shapes.distance(gp, rp, sp, cp, gq.reshape(1, -1), np.array([rq]), np.array([sq]), np.array([cq]))[0])


def test_same_symbol_turned_or_mirrored_is_the_same_shape():
    base = _switch()
    for turn, mirror in ((90, False), (37, False), (180, True), (215, True)):
        assert _dist(base, _switch(turn=turn, mirror=mirror)) < 0.15


def test_other_tessellation_is_close_other_symbol_is_far():
    assert _dist(_switch(36), _switch(44)) < shapes.MATCH_DIST
    assert _dist(_switch(), _socket(), 1, 0) > shapes.MATCH_DIST
    assert _dist(_switch(stem=0.12), _switch(stem=0.4)) > shapes.MATCH_DIST      # much longer stem: not the same


def test_unconfirmed_key_takes_the_nearest_confirmed_type():
    keys = ["~a-c1-h0", "~b-c0-h0", "~c-c1-h0", "~d-c0-h0"]
    pics = [shapes.picture(p) for p in (_switch(36), _socket(), _switch(44, turn=90), _switch(stem=0.5))]
    data = (np.array(keys), np.array([g.ravel() for g, _, _ in pics]), np.array([r for _, r, _ in pics]),
            np.array([s for _, _, s in pics]), np.array([shapes.circles(k) for k in keys]))
    got = shapes.nearest({"~a-c1-h0": "switch", "~b-c0-h0": "socket"}, data=data)
    assert got["~c-c1-h0"][0] == "switch" and got["~c-c1-h0"][1] < shapes.MATCH_DIST
    assert got["~d-c0-h0"][1] > shapes.MATCH_DIST                                  # too different: stays untyped


def test_switch_with_a_wire_stub_attached_is_peeled_out():
    """An archive project: a switch whose tap dot touched a 33 cm wire stub and the next tap dot became one
    unknown group; dropping parts until the rest looks like a confirmed switch finds it."""
    from conftest import make_scene

    def circle(cx, cy, r, n=16):
        a = np.linspace(0, 2 * np.pi, n + 1)
        return [(cx + r * np.cos(a[k]), cy + r * np.sin(a[k]), cx + r * np.cos(a[k + 1]), cy + r * np.sin(a[k + 1]))
                for k in range(n)]

    def switch(x, y):
        return [(0, circle(x, y, 0.069), None), (0, [(x - 0.02, y - 0.05, x + 0.21, y - 0.28)], None),
                (0, [(x - 0.02, y - 0.05, x + 0.07, y - 0.24)], None), (0, [(x + 0.07, y - 0.24, x + 0.02, y - 0.25)], None),
                (0, circle(x + 0.21, y - 0.31, 0.03, 10), None)]

    clean = make_scene(switch(5.0, 5.0), layers=["m-LİNYE"])            # the confirmed one, elsewhere
    g, r, sz = shapes.picture(np.vstack(list(shapes.part_points(clean, range(5)).values())))
    data = (np.array(["~sw-c0-h0"]), g.reshape(1, -1), np.array([r]), np.array([sz]), np.array([0]))
    refs = shapes.References({"~sw-c0-h0": "switch"}, data=data)
    stub = [(0, [(0.24, -0.31, 0.57, -0.31)], None), (0, circle(0.60, -0.31, 0.03, 10), None)]
    scene = make_scene(switch(0.0, 0.0) + stub, layers=["m-LİNYE"])
    found = shapes.peel(scene, list(range(7)), refs)
    assert found is not None and found[0] == "~sw-c0-h0"
    assert sorted(found[2]) == [0, 1, 2, 3, 4]
