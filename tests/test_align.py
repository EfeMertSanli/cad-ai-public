import numpy as np

from cadai.cleanup.align import DELETE, KEEP, OUT_OF_VIEW, align
from conftest import make_scene


def _plan(rng, x0, y0):
    """An irregular floor plan: wall pairs (kept) + dimensions and furniture (deleted)."""
    walls, junk = [], []
    x = x0
    for _ in range(30):
        w = rng.uniform(2.5, 6.0)
        h = rng.uniform(3.0, 7.0)
        for off in (0.0, 0.2):                                   # two faces of a wall
            walls.append([(x + off, y0, x + off, y0 + h)])
        walls.append([(x, y0 + h, x + w, y0 + h)])
        junk.append([(x + 0.5, y0 - 1.0, x + w - 0.5, y0 - 1.0)])  # a dimension line
        junk.append([(x + 1.0, y0 + 1.0, x + 1.7, y0 + 1.3)])      # furniture
        x += w
    return walls, junk


def test_recovers_translation_and_labels():
    rng = np.random.default_rng(1)
    walls, junk = _plan(rng, 100.0, 50.0)
    far_walls, far_junk = _plan(rng, 400.0, 50.0)                # another sheet the engineer didn't use
    items = [(0, w, None) for w in walls] + [(1, j, None) for j in junk] + \
            [(0, w, None) for w in far_walls] + [(1, j, None) for j in far_junk]
    raw = make_scene(items)
    shift = np.array([-87.3, 12.15, -87.3, 12.15])
    final_items = [(0, [tuple(np.array(s) + shift) for s in w], None) for w in walls]
    final_items += [(1, [(5.0, 5.0, 9.0, 5.0)], None)]           # an electrical line
    final = make_scene(final_items)

    res = align(raw, final)
    assert res["ok"]
    t = res["transforms"][0]
    assert (t["rot"], t["scale"]) == (0, 1.0)
    assert abs(t["tx"] + 87.3) < 0.03 and abs(t["ty"] - 12.15) < 0.03
    labels = res["labels"]
    nw, nj = len(walls), len(junk)
    assert (labels[:nw] == KEEP).all()
    junk_labels = labels[nw:nw + nj]
    assert (junk_labels[1::2] == DELETE).all()                       # furniture inside the plan
    assert np.isin(junk_labels[0::2], (DELETE, OUT_OF_VIEW)).all()    # dimensions 1 m outside: removed either way
    assert (labels[nw + nj:] == OUT_OF_VIEW).all()


def test_rotated_and_scaled_copy():
    rng = np.random.default_rng(2)
    walls, junk = _plan(rng, 10.0, 10.0)
    raw = make_scene([(0, w, None) for w in walls] + [(1, j, None) for j in junk])

    def move(seg):                                               # x10 scale, quarter turn, shift
        x1, y1, x2, y2 = (np.array(seg) * 10).tolist()
        return (-y1 + 3.0, x1 - 4.0, -y2 + 3.0, x2 - 4.0)

    final = make_scene([(0, [move(s) for s in w], None) for w in walls])
    res = align(raw, final)
    t = res["transforms"][0]
    assert (t["rot"], t["scale"]) == (1, 10.0)
    assert (res["labels"][:len(walls)] == KEEP).all()
    assert np.isin(res["labels"][len(walls):], (DELETE, OUT_OF_VIEW)).all()


def test_no_match_is_not_ok():
    rng = np.random.default_rng(3)
    walls, _ = _plan(rng, 0.0, 0.0)
    other, _ = _plan(np.random.default_rng(99), 0.0, 0.0)
    res = align(make_scene([(0, w, None) for w in walls]), make_scene([(0, w, None) for w in other]))
    assert not res["ok"] or res["counts"]["keep"] < len(walls) / 2


def test_one_scale_per_pair_and_weak_peaks_rejected():
    rng = np.random.default_rng(5)
    walls, junk = _plan(rng, 0.0, 0.0)
    raw = make_scene([(0, w, None) for w in walls] + [(1, j, None) for j in junk])
    final = make_scene([(0, w, None) for w in walls])               # same place, scale 1
    res = align(raw, final)
    assert res["ok"] and {t["scale"] for t in res["transforms"]} == {1.0}
    few = make_scene([(0, w, None) for w in walls[:6]])             # too little to trust
    assert not align(raw, few)["ok"]


def test_weak_view_does_not_swallow_neighbouring_sheets():
    from cadai.cleanup.align import _view_mask
    # a 6 x 6 m room (middle 3 m from the walls), a point just outside, another sheet
    matched = np.array([[x, y] for x in np.arange(0, 6, 0.1) for y in (0.0, 6.0)] +
                       [[x, y] for x in (0.0, 6.0) for y in np.arange(0, 6, 0.1)])
    centres = np.array([[3.0, 3.0], [8.5, 3.0], [40.0, 3.0]])
    assert _view_mask(matched, centres).tolist() == [True, True, False]
