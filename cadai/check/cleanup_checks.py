"""Checks on a cleaned drawing, before anyone looks at it (plan-check §2, M3a).

The architect wrote the area of most rooms in the raw drawing ("SALON 18.03 m²"). Those stamps
are deleted by the cleanup but are read first, so every cleaned plan can be measured for free:
rooms are read from what the cleanup kept (placement/rooms.py) and compared with the stamps.

Issues (code, severity, message, where):
- check.clean.area_mismatch   warning   room area differs > 10 % from the architect's stamp, after his
                                        convention (the median ratio; some write areas with the walls, ADR-030)
- check.clean.area_convention info      that convention, when it is more than 5 % off the floor
- check.clean.rooms_merged    warning   two stamps fall in one room: a wall or door between them is missing
- check.clean.stamp_no_room   warning   a stamp next to the plan lies in no room: the room is open
                                        (a wall was deleted or a doorway too wide to close)
Stamps farther than 1.5 m from every room (schedule tables, other sheets) are only counted.
"""
from __future__ import annotations

import numpy as np

from cadai.cad.scene import Scene
from cadai.i18n import say
from cadai.placement import rooms as rooms_mod
from cadai.placement.rooms import AREA_RE, stamps  # noqa: F401 - read here since ADR-030, kept as names of this module

MISMATCH = 0.10
NEAR_M = 1.5                  # stamps farther than this from every room are tables, not plans
CONVENTION = 0.05             # the architect's areas differ from the floor by more than this: say so


def plan_region(scene: Scene, keep: np.ndarray, margin: float = 2.0):
    bb = scene.bboxes()[keep]
    bb = bb[np.isfinite(bb[:, 0])]
    if not len(bb):
        return None
    return (float(np.percentile(bb[:, 0], 0.5)) - margin, float(np.percentile(bb[:, 1], 0.5)) - margin,
            float(np.percentile(bb[:, 2], 99.5)) + margin, float(np.percentile(bb[:, 3], 99.5)) + margin)


def check_cleanup(scene: Scene, keep: np.ndarray, stamp_scene: Scene | None = None) -> dict:
    """`stamp_scene`: read the area stamps from another scene in the same coordinates (the raw
    drawing, when `scene` is a cleaned copy that no longer has them)."""
    region = plan_region(scene, keep)
    if region is None:
        return {"rooms": 0, "stamps": 0, "issues": [{"code": "check.clean.empty", "severity": "error",
                                                     "message": say("Temizlikten sonra çizimde hiçbir şey kalmadı.",
                                                                    "Nothing was left in the drawing after cleanup.").both()}]}
    rm = rooms_mod.read_rooms(scene, keep, region)
    st = stamps(stamp_scene if stamp_scene is not None else scene)
    xy = np.array([[s["x"], s["y"]] for s in st]).reshape(-1, 2)
    inside = rm.at(xy, snap_m=0.3)
    near = rm.at(xy, snap_m=NEAR_M)
    by_id = {r["id"]: r for r in rm.rooms}
    diffs, issues, far, found = [], [], 0, []
    for s, rid, nid in zip(st, inside, near):
        if nid == 0:
            far += 1                                   # a schedule table or another sheet, not a plan
            continue
        if rid == 0:
            issues.append({"code": "check.clean.stamp_no_room", "severity": "warning",
                           "message": say(f"{s['m2']} m² yazan alan hiçbir odaya düşmüyor (oda açık ya da silinmiş).",
                                          f"The {s['m2']} m² area label falls in no room (room open or deleted).").both(),
                           "at": [round(s["x"], 2), round(s["y"], 2)]})
            continue
        found.append((by_id[int(rid)], s))
    counts: dict[int, int] = {}
    for room, _ in found:
        counts[room["id"]] = counts.get(room["id"], 0) + 1
    single = [(room, s) for room, s in found if counts[room["id"]] == 1]
    ratio = float(np.median([room["area_m2"] / s["m2"] for room, s in single])) if single else 1.0
    if single and abs(ratio - 1) > CONVENTION:
        off = f"{(1 / ratio - 1) * 100:+.0f}"
        issues.append({"code": "check.clean.area_convention", "severity": "info",
                       "message": say(f"Mimarın yazdığı alanlar duvar içinden ortalama %{off} farklı "
                                      "(alanları duvar payıyla ya da başka bir tanımla yazmış olabilir); oda kontrolü buna göre yapıldı.",
                                      f"The architect's areas differ from the floor by {off}% on average (they may include "
                                      "the walls or use another definition); the room check allows for this.").both()})
    for room, s in single:
        d = room["area_m2"] / (s["m2"] * ratio) - 1
        diffs.append(d)
        if abs(d) > MISMATCH:
            issues.append({"code": "check.clean.area_mismatch", "severity": "warning",
                           "message": say(f"Oda {room['area_m2']} m² okundu, mimar {s['m2']} m² yazmış (%{d * 100:+.0f}).",
                                          f"Room read as {room['area_m2']} m², the architect wrote {s['m2']} m² ({d * 100:+.0f}%).").both(),
                           "at": room["centre"], "room_type": room["type"]})
    for rid in sorted(k for k, n in counts.items() if n > 1):
        room = by_id[rid]
        written = [s["m2"] for r, s in found if r["id"] == rid]
        issues.append({"code": "check.clean.rooms_merged", "severity": "warning",
                       "message": say(f"Bir odada {len(written)} alan yazısı var ({', '.join(map(str, written))} m²): "
                                      "aralarındaki duvar ya da kapı eksik, odalar birleşmiş.",
                                      f"One room has {len(written)} area labels ({', '.join(map(str, written))} m²): "
                                      "the wall or door between them is missing, the rooms merged.").both(),
                       "at": room["centre"], "room_type": room["type"]})
    d = np.array(diffs)
    return {
        "rooms": len(rm.rooms),
        "stamps": len(st),
        "stamps_outside_plans": far,
        "stamps_in_rooms": int(len(d)),
        "area_ratio": round(ratio, 3) if single else None,
        "median_diff": round(float(np.median(d)), 3) if len(d) else None,
        "within_5": round(float(np.mean(np.abs(d) <= 0.05)), 3) if len(d) else None,
        "within_10": round(float(np.mean(np.abs(d) <= 0.10)), 3) if len(d) else None,
        "issues": issues,
    }
