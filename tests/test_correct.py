import json

import numpy as np
import pytest

from tests.conftest import make_scene
from cadai.i18n import say


def placed(created="20261014-100000"):
    """One plan, a 4 x 3 m room (walls on its sides) and a 2 x 3 m room right of it."""
    return {"model": "placement-x", "created": created, "plans": [{"region": [-1, -1, 7, 4], "rooms": [
        {"id": 1, "names": ["SALON"], "type": "living", "area_m2": 12, "bbox": [0, 0, 4, 3], "centre": [2, 1.5],
         "devices": [{"type": "socket", "x": 0.0, "y": 1.0, "angle": 0.0},
                     {"type": "light_ceiling", "x": 2.0, "y": 1.5}]},
        {"id": 2, "names": ["BANYO"], "type": "bathroom", "area_m2": 6, "bbox": [4, 0, 6, 3], "centre": [5, 1.5],
         "devices": [{"type": "socket_wet", "x": 6.0, "y": 1.0, "angle": 180.0}]}]}]}


WALLS = np.array([[0, 0, 0, 3], [0, 0, 6, 0], [0, 3, 6, 3], [4, 0, 4, 3], [6, 0, 6, 3]], dtype=float)


@pytest.fixture
def out(tmp_path):
    (tmp_path / "devices.json").write_text(json.dumps(placed()), encoding="utf-8")
    return tmp_path


def test_device_corrections_snap_log_and_count_as_learning_data(out):
    from cadai.correct import devices, store
    v = devices.view(devices.load(out))
    assert v["tally"] == {"open": 3} and v["rooms_total"] == 2
    # a socket dragged to 0.3 m right of the wall x=4, inside the bathroom: back on the wall, facing the bathroom
    e = devices.edit(out, "move", user="ali", segs=WALLS, id="d0", x=4.3, y=2.0)
    moved = e["changes"][0]["after"]
    assert (moved["x"], moved["y"], moved["angle"], moved["room"]) == (4.0, 2.0, 0.0, 2)
    assert e["user"] == "ali" and e["changes"][0]["before"]["x"] == 0.0
    # a ceiling light does not snap
    devices.edit(out, "add", segs=WALLS, type="light_ceiling", x=3.8, y=0.2)
    added = devices.load(out)["devices"][-1]
    assert (added["x"], added["y"], added["origin"], added["room"]) == (3.8, 0.2, "engineer", 1)
    devices.edit(out, "delete", id="d2")
    devices.edit(out, "room", plan=0, room=1, reviewed=True)
    st = {d["id"]: d["status"] for d in devices.view(devices.load(out))["devices"]}
    assert st == {"d0": "changed", "d1": "accepted", "d2": "deleted", "e1": "added"}
    # reasons given after the action are merged by seq
    devices.edit(out, "reason", target=0, reasons=["yanlış duvar"], note="banyo kapısının yanına")
    assert store.reasons(store.read(out))[0] == {"reasons": ["yanlış duvar"], "note": "banyo kapısının yanına"}
    # the corrected drawing leaves the deleted device out
    devs = [d for p in devices.result_for_dxf(devices.load(out))["plans"] for r in p["rooms"] for d in r["devices"]]
    assert sorted(d["type"] for d in devs) == ["light_ceiling", "light_ceiling", "socket"]


def test_undo_reverts_the_last_action_and_ids_are_not_reused(out):
    from cadai.correct import devices
    devices.edit(out, "add", type="socket", x=2.0, y=2.9, segs=WALLS)
    devices.edit(out, "undo")
    assert all(d["origin"] == "model" for d in devices.load(out)["devices"])
    devices.edit(out, "add", type="socket", x=2.0, y=2.9, segs=WALLS)
    assert devices.load(out)["devices"][-1]["id"] == "e2"
    devices.edit(out, "retype", id="d1", type="light_emergency")
    devices.edit(out, "undo")
    assert devices.load(out)["devices"][1]["type"] == "light_ceiling"
    devices.edit(out, "undo")                                    # the second add; the first was undone already
    with pytest.raises(ValueError):
        devices.edit(out, "undo")
    with pytest.raises(ValueError):
        devices.edit(out, "add", type="kahve", x=0, y=0)


def test_new_placement_starts_a_new_working_copy(out):
    from cadai.correct import devices
    devices.edit(out, "delete", id="d0")
    (out / "devices.json").write_text(json.dumps(placed("20261015-090000")), encoding="utf-8")
    assert not any(d["deleted"] for d in devices.load(out)["devices"])
    assert (out / "devices_corrected-20261014-100000.json").exists()


@pytest.fixture
def cleaned(tmp_path):
    """Wall (kept), a door arc removed by the door tool, a text outside the plans."""
    from cadai.cleanup import tools
    scene = make_scene([(0, [(0, 0, 4, 0)], None), (1, [(1, 0.05, 1.5, 0.5)], None), (2, [(10, 10, 11, 10)], "KESIT")],
                       layers=["DUVAR", "KAPI", "YAZI"])
    scene.save(tmp_path / "scene.npz")
    why = np.array([-1, tools.TOOLS.index("door_swings"), -2])
    np.savez_compressed(tmp_path / "decision.npz", kept=np.array([True, False, False]),
                        applied=np.array([1, 0, 0], dtype=np.int8), why=why)
    (tmp_path / "summary.json").write_text(json.dumps({"source": "x.dwg"}), encoding="utf-8")
    return tmp_path


def test_cleanup_pick_says_why_and_marks_are_kept_with_reasons(cleaned):
    from cadai.correct import cleanup
    L = cleanup.load(cleaned)
    assert cleanup.pick(L, 1.25, 0.3, 0.1) == [1]
    assert cleanup.pick(L, 3.0, 2.0, 0.1) == []
    d = cleanup.describe(L, [1], {})
    assert d["reasons"] == [{"text": say("silindi: Kapı yayı ve kanadı aracı", "deleted: Door swings and leaves tool"), "count": 1}]
    assert d["layers"][0]["name"] == "KAPI" and d["deleted"] == 1
    assert cleanup.reason(L, 2) == say("silindi: kat planı dışında", "deleted: outside the floor plans")
    assert cleanup.reason(L, 0).tr.startswith("kaldı: planda") and cleanup.reason(L, 0).en.startswith("kept: in the plan")
    assert cleanup.box(L, -1, -1, 5, 1) == [0, 1]
    cleanup.edit(cleaned, L, [1], True, ["kapı boşluğu"], "kapı yayı çizimde kalsın", user="ayşe")
    assert cleanup.overrides(cleaned) == {1: True}
    assert cleanup.codes(L, cleanup.overrides(cleaned)).tolist() == [cleanup.KEPT, cleanup.SAY_KEEP, cleanup.DELETED]
    m = cleanup.marks(cleaned)[0]
    assert m["keep"] is True and m["reasons"] == ["kapı boşluğu"] and m["system_reason"] == [{"tr": "silindi: Kapı yayı ve kanadı aracı", "en": "deleted: Door swings and leaves tool"}]
    assert m["bbox"] == [1.0, 0.05, 1.5, 0.5]
    cleanup.undo(cleaned)
    assert cleanup.overrides(cleaned) == {} and cleanup.marks(cleaned) == []


def test_cleanup_reason_without_tool_record_uses_layer_type(cleaned):
    from cadai.correct import cleanup
    np.savez_compressed(cleaned / "decision.npz", kept=np.array([True, False, False]))
    L = cleanup.load(cleaned)
    assert L.why is None and L.applied is None
    assert cleanup.reason(L, 0).tr.startswith("kaldı: katman tipi") and cleanup.reason(L, 0).en.startswith("kept: layer type")


def test_edit_page_routes_are_not_taken_by_output_files(cleaned, monkeypatch):
    from fastapi.testclient import TestClient
    from cadai.web import app as web
    monkeypatch.setattr(web, "OUTPUTS", cleaned.parent)
    web._LOADED.clear()
    name = cleaned.name
    c = TestClient(web.app)
    r = c.get(f"/api/outputs/{name}/view/clean.png", params={"x0": -1, "y0": -1, "x1": 5, "y1": 2, "w": 300})
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    picked = c.get(f"/api/outputs/{name}/cleanup/pick", params={"x": 1.25, "y": 0.3, "tol": 0.1}).json()
    assert picked["items"] == [1]
    r = c.post(f"/api/outputs/{name}/cleanup/edit", json={"items": [1], "keep": True, "reasons": ["kapı boşluğu"]},
               auth=("ayse", ""))
    assert r.status_code == 200 and r.json()["say_keep"] == 1
    assert c.get(f"/api/outputs/{name}/edits/log").json()[0]["user"] == "ayse"
    assert c.get(f"/api/outputs/{name}/devices/state").status_code == 404      # cleaned, not placed yet
    (cleaned / "devices.json").write_text(json.dumps(placed()), encoding="utf-8")
    r = c.post(f"/api/outputs/{name}/devices/edit", json={"action": "move", "id": "d0", "x": 0.2, "y": 0.1})
    assert r.status_code == 200 and r.json()["tally"]["changed"] == 1
    assert c.post(f"/api/outputs/{name}/devices/edit", json={"action": "fly"}).status_code == 400
    assert c.get("/edit").status_code == 200
