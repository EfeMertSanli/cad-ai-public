"""Build the anonymised demo projects in `demo/` (ADR-032).

Per project: the architect's raw drawing, our cleaned drawing and our electrical placement as DWG (each through
`anonymise`, and only written when `leaks` finds nothing), pictures of the sheet and of every floor plan, and
the numbers (facts.json). The pictures are drawn from line geometry only: no text of the drawing is printed.

Run: python -m cadai.demo.build
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cadai import config
from cadai.cad import oda
from cadai.cad.render import render
from cadai.demo.anonymise import anonymise, leaks

ROOT = Path(__file__).resolve().parents[2] / "demo"
OUTPUTS = config.DATA / "outputs"
WORK = config.DATA / "tmp" / "demo"
# the cleanup outputs behind each demo project; their folder names carry plot numbers, so they stay on this computer
SOURCES = config.DATA / "demo" / "outputs.json"          # {"<key>": "<output folder>"}
PROJECTS = [{"key": "cevriye-berkoglu", "title": "Cevriye Berkoğlu"}, {"key": "ali-kemal-arslan", "title": "Ali Kemal Arslan"}]
NAMES = {"light_ceiling": "ceiling light", "socket": "socket", "socket_wet": "wet-area socket", "switch": "switch",
         "panel": "apartment panel", "light_emergency": "emergency light"}
COLOURS = {"light_ceiling": (235, 140, 0), "socket": (220, 0, 0), "socket_wet": (140, 0, 170), "switch": (0, 90, 230),
           "panel": (70, 70, 70), "light_emergency": (0, 150, 60)}
ROOMS = {"bedroom": "bedroom", "living": "living room", "living_kitchen": "living + kitchen", "kitchen": "kitchen",
         "bathroom": "bathroom", "wc": "WC", "hall": "hall", "balcony": "balcony", "study": "study", "laundry": "laundry",
         "dressing": "dressing", "storage": "storage", "stair": "stair", "lift": "lift", "shop": "shop", "office": "office",
         "shelter": "shelter", "plant": "plant room", "parking": "car park"}
PX_PER_M = 70


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _symbol(d: ImageDraw.ImageDraw, kind: str, X: float, Y: float, r: float, ring: bool = False):
    c = COLOURS[kind]
    if ring:                                       # ours on the comparison picture
        d.ellipse((X - r * 1.5, Y - r * 1.5, X + r * 1.5, Y + r * 1.5), outline=c, width=3)
    elif kind == "light_ceiling":
        d.ellipse((X - r, Y - r, X + r, Y + r), fill="white", outline=c, width=3)
        d.line((X - r * .7, Y - r * .7, X + r * .7, Y + r * .7), fill=c, width=2)
        d.line((X - r * .7, Y + r * .7, X + r * .7, Y - r * .7), fill=c, width=2)
    elif kind == "socket":
        d.ellipse((X - r, Y - r, X + r, Y + r), fill=c)
    elif kind == "socket_wet":
        d.rectangle((X - r, Y - r, X + r, Y + r), fill=c)
    elif kind == "panel":
        d.rectangle((X - r * .6, Y - r * 1.4, X + r * .6, Y + r * 1.4), fill="white", outline=c, width=3)
    elif kind == "light_emergency":
        d.ellipse((X - r, Y - r, X + r, Y + r), fill="white", outline=c, width=3)
        d.ellipse((X - r * .4, Y - r * .4, X + r * .4, Y + r * .4), fill=c)
    else:                                          # switch
        d.polygon([(X, Y - r - 2), (X - r, Y + r), (X + r, Y + r)], fill=c)


def _plan_pictures(data, plan: dict, truth: list[dict] | None, target: Path, compare: Path) -> None:
    x0, y0, x1, y1 = plan["region"]
    x0, y0, x1, y1 = x0 - 1, y0 - 1, x1 + 1, y1 + 1
    width = int((x1 - x0) * PX_PER_M)
    code = np.where(data.kept & (data.scene.text == ""), 1, -9)
    render(data.scene, target, code, bbox=(x0, y0, x1, y1), width=width, colours={1: (90, 90, 90)})
    base = Image.open(target).convert("RGB")
    P = lambda x, y: ((x - x0) * PX_PER_M, (y1 - y) * PX_PER_M)
    r, font = 7, _font(13)
    ours = [dev for room in plan["rooms"] for dev in room["devices"]]
    img = base.copy()
    d = ImageDraw.Draw(img)
    for room in plan["rooms"]:
        X, Y = P(*room["centre"])
        label = f"{ROOMS.get(room['type'], room['type'] or 'room')} {room['area_m2']:.1f} m²"
        d.text((X, Y + 14), label, fill=(120, 120, 140), font=font, anchor="mm")
    for dev in ours:
        _symbol(d, dev["type"], *P(dev["x"], dev["y"]), r)
    img.save(target)
    if truth is None:
        return
    img = base.copy()
    d = ImageDraw.Draw(img)
    for dev in truth:
        if dev["type"] in COLOURS and x0 <= dev["x"] <= x1 and y0 <= dev["y"] <= y1:
            X, Y = P(dev["x"], dev["y"])
            d.ellipse((X - r * .8, Y - r * .8, X + r * .8, Y + r * .8), fill=COLOURS[dev["type"]])
    for dev in ours:
        _symbol(d, dev["type"], *P(dev["x"], dev["y"]), r, ring=True)
    img.save(compare)


def _drawing(src_dxf: Path, work: Path, plans: list, margin: float, target: Path, log) -> dict:
    anon = work / (target.stem + ".dxf")
    report = anonymise(src_dxf, anon, plans, margin)
    found = leaks(anon)
    if found:
        raise RuntimeError(f"{target.name}: still looks personal: {found[:5]}")
    oda.to_dwg(anon, target)
    # the DWG writer stamps the logged-in Windows user as "last saved by" (the author's own name, allowed);
    # the computer name must not be there
    host = os.environ.get("COMPUTERNAME", "")
    raw = target.read_bytes()
    if host and (host.encode("utf-16-le") in raw or host.encode() in raw):
        raise RuntimeError(f"{target.name}: the computer name is in the file")
    log(f"{target.name}: {report}, {target.stat().st_size / 1e6:.1f} MB")
    return report


def build_project(p: dict, log=print) -> dict:
    from cadai.correct import cleanup
    from cadai.placement import place
    out_dir = OUTPUTS / json.loads(SOURCES.read_text(encoding="utf-8"))[p["key"]]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    placed = json.loads((out_dir / "devices.json").read_text(encoding="utf-8"))
    dest, work = ROOT / p["key"], WORK / p["key"]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    work.mkdir(parents=True, exist_ok=True)
    scale, to_cm = summary["unit"]["scale"], summary["write"]["scaled_to_cm"]
    regions = [plan["region"] for plan in placed["plans"]]
    # 1. drawings: raw (a copy of the read-only archive file), cleaned, placed
    raw_copy = work / "source.dwg"
    shutil.copyfile(summary["source"], raw_copy)
    raw_dxf = oda.to_dxf(raw_copy, work / "source.dxf")
    removed = _drawing(raw_dxf, work, [[v / scale for v in r] for r in regions], 1 / scale, dest / "1-architect-raw.dwg", log)
    out_plans = [[v / scale * to_cm for v in r] for r in regions]
    _drawing(out_dir / "clean.dxf", work, out_plans, to_cm / scale, dest / "2-cleaned.dwg", log)
    _drawing(out_dir / "placed.dxf", work, out_plans, to_cm / scale, dest / "3-electrical.dwg", log)
    # 2. pictures
    data = cleanup.load(out_dir)
    code = np.where(data.kept, 1, 0)
    render(data.scene, dest / "sheet.png", code, width=3200, colours={0: (240, 190, 190), 1: (30, 30, 30)})
    truth = place.archive_devices(summary["source"], data.scene)
    for k, plan in enumerate(placed["plans"], start=1):
        _plan_pictures(data, plan, truth, dest / f"plan-{k}.png", dest / f"plan-{k}-vs-engineer.png")
    # 3. numbers
    rooms = [room for plan in placed["plans"] for room in plan["rooms"]]
    stamped = [room for room in rooms if room.get("stamp_m2")]
    ratio = float(np.median([room["area_m2"] / room["stamp_m2"] for room in stamped])) if stamped else None
    within = sum(abs(room["area_m2"] / (room["stamp_m2"] * ratio) - 1) <= 0.10 for room in stamped) if stamped else 0
    score = placed.get("score") or {}
    hits = sum(v["hits_10"] for v in score.values())
    ours, eng = sum(v["ours"] for v in score.values()), sum(v["engineer"] for v in score.values())
    facts = {
        "title": p["title"], "key": p["key"], "model": placed["model"],
        "cleanup": {"items_kept": summary["counts"].get("keep"), "items_deleted": summary["counts"].get("delete"),
                    "mode": summary.get("mode")},
        "plans": len(regions), "rooms": len(rooms), "rooms_typed": sum(1 for room in rooms if room["type"]),
        "area_check": {"rooms_with_area_label": len(stamped), "within_10pct": int(within),
                       "architect_convention": round(ratio, 3) if ratio else None},
        "devices": placed["counts"], "names": NAMES,
        "vs_engineer": {k: v for k, v in score.items()},
        "f1_1m": round(2 * hits / (ours + eng), 3) if ours + eng else None,
        "privacy": removed,
        "files": sorted(f.name for f in dest.iterdir()),
    }
    (dest / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf-8")
    return facts


ORDER = ["light_ceiling", "socket", "socket_wet", "switch", "panel", "light_emergency"]


def readme(facts: list[dict]) -> str:
    """demo/README.md: the same story as the site's Demo page, for GitHub."""
    out = ["# Demo: from an architect's drawing to an electrical layout", "",
           "Two real projects from the office's archive, neither used for training. For each: the architect's raw "
           "drawing, our cleaned base plan and our electrical layout as DWG (they open in AutoCAD), pictures, and a "
           "comparison with the electrical engineer's own drawing of the same building.", "",
           "Devices: ceiling light (orange ring with a cross), socket (red dot), wet-area socket (purple square), "
           "switch (blue triangle), apartment panel (grey bar), emergency light (green ring with a dot). "
           "In the comparison pictures a filled dot is the engineer's device and a ring is ours.", ""]
    for f in facts:
        k, ac = f["key"], f["area_check"]
        total = sum(f["devices"].values())
        out += [f"## {f['title']}", "",
                f"- {f['plans']} floor plans found; sections, elevations, site plan and details removed",
                f"- {f['rooms']} rooms read; {ac['within_10pct']} of {ac['rooms_with_area_label']} within 10% of the area the architect wrote",
                f"- {total} devices placed; agreement with the engineer **F1 {f['f1_1m']:.2f}** (devices within 1 m)", "",
                f"Files: [1-architect-raw.dwg]({k}/1-architect-raw.dwg) · [2-cleaned.dwg]({k}/2-cleaned.dwg) · "
                f"[3-electrical.dwg]({k}/3-electrical.dwg)", "",
                f"The whole sheet: dark is what the cleanup kept, pink what it removed.", "",
                f"![sheet]({k}/sheet.png)", ""]
        plans = sorted((n for n in f["files"] if re.fullmatch(r"plan-\d+\.png", n)), key=lambda n: int(n[5:-4]))
        for n in plans:
            out += [f"![our layout]({k}/{n})", "", f"Ours vs the engineer's:", "", f"![ours vs the engineer]({k}/{n[:-4]}-vs-engineer.png)", ""]
        out += ["| Device | Ours | Engineer | Within 50 cm | Within 1 m |", "|---|---:|---:|---:|---:|"]
        out += [f"| {f['names'][c]} | {v['ours']} | {v['engineer']} | {v['hits_05']} | {v['hits_10']} |"
                for c in ORDER if (v := f["vs_engineer"].get(c))]
        out.append("")
    out += ["## Privacy", "",
            "The drawings are anonymised (`cadai/demo/anonymise.py`, ADR-032): every text outside the floor plans "
            "(title block, site plan, project tables) was removed, and inside the plans anything that looks like an "
            "address, phone number, e-mail or ID number; embedded files and pictures were removed too. A scan of every "
            "string in the written files, including extended data, finds nothing personal. Owners' names may appear. "
            "The office agreed to this demo. The pictures are drawn from line geometry only.", ""]
    return "\n".join(out)


def build(log=print) -> list[dict]:
    facts = [build_project(p, log) for p in PROJECTS]
    (ROOT / "projects.json").write_text(json.dumps([f["key"] for f in facts]), encoding="utf-8")
    (ROOT / "README.md").write_text(readme(facts), encoding="utf-8")
    return facts


if __name__ == "__main__":
    for f in build():
        print(f["title"], f["f1_1m"], f["devices"])
