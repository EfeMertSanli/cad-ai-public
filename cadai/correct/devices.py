"""The engineer corrects the placed devices (Düzelt page, Cihazlar tab, ADR-029).

The working copy is devices_corrected.json next to devices.json: every device with an id, its room,
where the model put it (`orig`) and whether it is deleted; every room with `reviewed` ("oda tamam").
What a device counts as for learning comes from these, not from a stored status:

    added     the engineer put it there
    deleted   the model's device, removed
    changed   the model's device, moved or given another type
    accepted  the model's device, untouched, in a room the engineer marked as checked
    open      the model's device, untouched, room not checked yet (not evidence either way)

A new placement (devices.json with another `created`) starts a new working copy; the old one is
kept as devices_corrected-<created>.json. Wall devices snap to the nearest kept line within
SNAP_M and face the side the engineer clicked on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cadai.correct import store
from cadai.i18n import say
from cadai.placement.candidates import CEILING_CLASSES

STATE = "devices_corrected.json"
SNAP_M = 0.6
ON_LINE_M = 0.02
# the engineer's reasons; the Turkish text is what the log stores
REASONS = [say("kapı arkasında kalıyor", "behind the door"), say("kilit tarafına", "to the latch side"),
           say("mobilya önünde", "in front of furniture"), say("tezgah üstü", "above the counter"),
           say("yatak başı", "bedside"), say("TV / oturma köşesi", "TV / seating corner"),
           say("ıslak hacim", "wet area"), say("simetri / hiza", "symmetry / alignment"), say("fazla", "too many"),
           say("eksik", "missing"), say("yanlış duvar", "wrong wall"), say("yanlış tip", "wrong type"),
           say("ofis alışkanlığı", "office habit")]


def _fresh(placed: dict) -> dict:
    devices, rooms = [], []
    for p, plan in enumerate(placed["plans"]):
        for r in plan["rooms"]:
            rooms.append({"plan": p, "id": r["id"], "names": r.get("names", []), "type": r.get("type"),
                          "type_guessed": r.get("type_guessed", False), "area_m2": r.get("area_m2"),
                          "bbox": r["bbox"], "centre": r["centre"], "stamp_m2": r.get("stamp_m2"),
                          "stamps": r.get("stamps", 0), "reviewed": False})
            for d in r["devices"]:
                dev = {"id": f"d{len(devices)}", "type": d["type"], "x": d["x"], "y": d["y"],
                       "angle": d.get("angle", 0.0), "plan": p, "room": r["id"], "origin": "model", "deleted": False}
                dev["orig"] = {k: dev[k] for k in ("type", "x", "y", "angle")}
                devices.append(dev)
    return {"base": placed.get("created"), "model": placed.get("model"),
            "regions": [plan["region"] for plan in placed["plans"]], "rooms": rooms, "devices": devices}


def load(out_dir: Path) -> dict:
    placed_file = out_dir / "devices.json"
    if not placed_file.exists():
        raise FileNotFoundError(str(say("bu çıktıda yerleşim yok", "this output has no placement")))
    placed = json.loads(placed_file.read_text(encoding="utf-8"))
    f = out_dir / STATE
    if f.exists():
        state = json.loads(f.read_text(encoding="utf-8"))
        if state.get("base") == placed.get("created"):
            return state
        f.replace(out_dir / f"devices_corrected-{state.get('base')}.json")
    state = _fresh(placed)
    save(out_dir, state)
    return state


def save(out_dir: Path, state: dict) -> None:
    tmp = out_dir / (STATE + ".part")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_dir / STATE)


def status(dev: dict, reviewed: bool) -> str:
    if dev["origin"] == "engineer":
        return "deleted" if dev["deleted"] else "added"
    if dev["deleted"]:
        return "deleted"
    o = dev["orig"]
    if dev["type"] != o["type"] or abs(dev["x"] - o["x"]) > 1e-3 or abs(dev["y"] - o["y"]) > 1e-3:
        return "changed"
    return "accepted" if reviewed else "open"


def view(state: dict) -> dict:
    """What the page draws: devices with their status, rooms, progress."""
    reviewed = {(r["plan"], r["id"]) for r in state["rooms"] if r["reviewed"]}
    devs = [{**d, "status": status(d, (d["plan"], d["room"]) in reviewed)} for d in state["devices"]]
    tally: dict[str, int] = {}
    for d in devs:
        if d["origin"] == "engineer" and d["deleted"]:
            continue
        tally[d["status"]] = tally.get(d["status"], 0) + 1
    return {"model": state["model"], "regions": state["regions"], "rooms": state["rooms"], "devices": devs,
            "tally": tally, "rooms_reviewed": len(reviewed), "rooms_total": len(state["rooms"])}


# ------------------------------------------------------------------------------- geometry
def room_at(state: dict, x: float, y: float) -> tuple[int | None, int | None]:
    """(plan, room id) of a point: the smallest room box holding it, else the nearest room centre of the
    plan region holding it (room boxes are rough, ADR-029)."""
    inside = [r for r in state["rooms"] if r["bbox"][0] <= x <= r["bbox"][2] and r["bbox"][1] <= y <= r["bbox"][3]]
    if inside:
        r = min(inside, key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]))
        return r["plan"], r["id"]
    for p, (x0, y0, x1, y1) in enumerate(state["regions"]):
        if x0 <= x <= x1 and y0 <= y <= y1:
            rooms = [r for r in state["rooms"] if r["plan"] == p]
            if not rooms:
                return p, None
            r = min(rooms, key=lambda r: np.hypot(r["centre"][0] - x, r["centre"][1] - y))
            return p, r["id"]
    return None, None


def snap(segs: np.ndarray | None, x: float, y: float, towards: tuple[float, float] | None = None
         ) -> tuple[float, float, float | None]:
    """(x, y, angle) of a wall device put at (x, y): the nearest point of a kept line within SNAP_M, facing
    `towards` (where the engineer clicked, or the room centre when the click was on the line). Angle None
    when there is no line near."""
    if segs is None or not len(segs):
        return x, y, None
    a, b = segs[:, :2], segs[:, 2:]
    ab = b - a
    t = np.clip(((np.array([x, y]) - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    q = a + t[:, None] * ab
    dist = np.hypot(q[:, 0] - x, q[:, 1] - y)
    k = int(np.argmin(dist))
    if dist[k] > SNAP_M:
        return x, y, None
    qx, qy = float(q[k, 0]), float(q[k, 1])
    tx, ty = (x, y) if dist[k] > ON_LINE_M or towards is None else towards
    d = np.array([tx - qx, ty - qy])
    n = np.array([-ab[k, 1], ab[k, 0]])
    n = n / max(np.hypot(*n), 1e-12)
    if n @ d < 0:
        n = -n
    return round(qx, 3), round(qy, 3), round(float(np.degrees(np.arctan2(n[1], n[0]))) % 360, 1)


# ------------------------------------------------------------------------------- actions
def _find(state: dict, did: str) -> dict:
    for d in state["devices"]:
        if d["id"] == did:
            return d
    raise KeyError(str(say(f"cihaz yok: {did}", f"no such device: {did}")))


def _room(state: dict, plan, rid) -> dict:
    for r in state["rooms"]:
        if r["plan"] == plan and r["id"] == rid:
            return r
    raise KeyError(str(say(f"oda yok: {plan}/{rid}", f"no such room: {plan}/{rid}")))


def _put(state: dict, dev: dict, x: float, y: float, segs, towards=None) -> None:
    if dev["type"] not in CEILING_CLASSES:
        x, y, angle = snap(segs, x, y, towards)
        if angle is not None:
            dev["angle"] = angle
    dev["x"], dev["y"] = round(float(x), 3), round(float(y), 3)
    plan, room = room_at(state, dev["x"], dev["y"])
    if plan is not None:
        dev["plan"], dev["room"] = plan, room


def edit(out_dir: Path, action: str, user: str = "", segs: np.ndarray | None = None, **kw) -> dict:
    """Apply one action and log it. Returns the log entry (with the changed devices / rooms).
    move(id, x, y) · add(type, x, y) · delete(id) · retype(id, type) · room(plan, room, reviewed)
    · reason(target, reasons, note) · undo()"""
    state = load(out_dir)
    changes = []                                  # (kind, key, before, after)

    def device_change(dev_before, dev_after):
        changes.append({"kind": "device", "key": (dev_after or dev_before)["id"], "before": dev_before, "after": dev_after})

    if action == "move":
        dev = _find(state, kw["id"])
        before = dict(dev)
        _put(state, dev, kw["x"], kw["y"], segs)
        device_change(before, dict(dev))
    elif action == "add":
        if kw["type"] not in _classes():
            raise ValueError(str(say(f"bilinmeyen cihaz: {kw['type']}", f"unknown device: {kw['type']}")))
        state["added"] = state.get("added", 0) + 1                # ids are never reused, even after an undo
        dev = {"id": f"e{state['added']}", "type": kw["type"],
               "x": 0.0, "y": 0.0, "angle": 0.0, "plan": None, "room": None, "origin": "engineer", "deleted": False}
        _put(state, dev, kw["x"], kw["y"], segs)
        state["devices"].append(dev)
        device_change(None, dict(dev))
    elif action == "delete":
        dev = _find(state, kw["id"])
        before = dict(dev)
        dev["deleted"] = True
        device_change(before, dict(dev))
    elif action == "retype":
        if kw["type"] not in _classes():
            raise ValueError(str(say(f"bilinmeyen cihaz: {kw['type']}", f"unknown device: {kw['type']}")))
        dev = _find(state, kw["id"])
        before = dict(dev)
        dev["type"] = kw["type"]
        if dev["type"] not in CEILING_CLASSES and before["type"] in CEILING_CLASSES:
            _put(state, dev, dev["x"], dev["y"], segs)
        device_change(before, dict(dev))
    elif action == "room":
        r = _room(state, kw["plan"], kw["room"])
        before = dict(r)
        r["reviewed"] = bool(kw.get("reviewed", True))
        changes.append({"kind": "room", "key": [r["plan"], r["id"]], "before": before, "after": dict(r)})
    elif action == "reason":
        return store.append(out_dir, {"kind": "devices", "action": "reason", "user": user, "target": int(kw["target"]),
                                      "reasons": [str(x) for x in kw.get("reasons", [])][:20],
                                      "note": str(kw.get("note", ""))[:2000]})
    elif action == "undo":
        last = _last_undoable(out_dir)
        if last is None:
            raise ValueError(str(say("geri alınacak işlem yok", "nothing to undo")))
        for c in reversed(last["changes"]):
            if c["kind"] == "room":
                _room(state, *c["key"]).update(c["before"])
            elif c["before"] is None:
                state["devices"] = [d for d in state["devices"] if d["id"] != c["key"]]
            else:
                _find(state, c["key"]).update(c["before"])
        changes = [{**c, "before": c["after"], "after": c["before"]} for c in last["changes"]]
        save(out_dir, state)
        return store.append(out_dir, {"kind": "devices", "action": "undo", "user": user, "target": last["seq"],
                                      "changes": changes})
    else:
        raise ValueError(str(say(f"bilinmeyen işlem: {action}", f"unknown action: {action}")))
    save(out_dir, state)
    return store.append(out_dir, {"kind": "devices", "action": action, "user": user, "base": state["base"],
                                  "changes": changes})


def _last_undoable(out_dir: Path) -> dict | None:
    entries = [e for e in store.read(out_dir) if e.get("kind") == "devices"]
    undone = {e["target"] for e in entries if e["action"] == "undo"}
    undos = {e["seq"] for e in entries if e["action"] == "undo"}
    for e in reversed(entries):
        if e["action"] in ("reason",) or e["seq"] in undos or e["seq"] in undone:
            continue
        return e
    return None


def _classes() -> list[str]:
    from cadai.placement.place import PLACE_CLASSES
    return PLACE_CLASSES


def result_for_dxf(state: dict) -> dict:
    """place.write_dxf's input from the corrected devices (lights wired to their room's nearest switch)."""
    from cadai.placement.place import light_links
    plans = []
    for p, region in enumerate(state["regions"]):
        by_room: dict = {}
        for d in state["devices"]:
            if d["plan"] == p and not d["deleted"]:
                by_room.setdefault(d["room"], []).append({k: d[k] for k in ("type", "x", "y", "angle")})
        plans.append({"region": region, "rooms": [{"devices": devs, "links": light_links(devs)} for devs in by_room.values()]})
    return {"plans": plans}


def export_job(job, output: str) -> dict:
    """placed_corrected.dxf / .dwg: the cleaned drawing with the corrected devices."""
    from cadai.placement.place import OUTPUTS, write_dxf
    out_dir = (OUTPUTS / output).resolve()
    if OUTPUTS.resolve() not in out_dir.parents:
        raise RuntimeError(str(say(f"çıktı yok: {output}", f"no such output: {output}")))
    state = load(out_dir)
    stats = write_dxf(out_dir / "clean.dxf", result_for_dxf(state), out_dir / "placed_corrected.dxf")
    job.log(say(f"düzeltilmiş DXF: {stats['devices']} cihaz", f"corrected DXF: {stats['devices']} devices"))
    try:
        from cadai.cad import oda
        oda.to_dwg(out_dir / "placed_corrected.dxf", out_dir / "placed_corrected.dwg")
    except Exception as exc:  # noqa: BLE001 - the DXF is still usable
        job.log(say(f"DWG yazılamadı: {exc}", f"could not write the DWG: {exc}"))
    return {"output": output, **stats, "tally": view(state)["tally"]}
