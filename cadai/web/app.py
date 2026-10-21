"""Website: projects, pipeline status, models, and a debug panel (jobs, logs, failures, system).

    python -m cadai.web            (or start.bat) -> http://127.0.0.1:8800
"""
from __future__ import annotations

import base64
import json
import secrets
import platform
from datetime import datetime
import shutil
import subprocess
import threading
import sys
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from cadai import config, i18n, jobs, log
from cadai.i18n import say
from cadai.sysinfo import memory as _memory
from cadai.cleanup.train import split_of
from cadai.corpus import inventory, process
from cadai.placement import dataset as placement_dataset
from cadai.placement import devices, furniture, furniture_shapes, notes, place, shapes, symbols
from cadai.placement import model as placement_model

log.setup()
config.ensure_dirs()
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="cad-ai")
runner = jobs.JobRunner()


@app.middleware("http")
async def fresh_pages(request: Request, call_next):
    """Pages and their scripts are checked for a newer version on every visit: without this a browser kept
    yesterday's i18n.js and today's pages failed in it (no header, no EN/TR switch)."""
    response = await call_next(request)
    path = request.url.path
    if (path.startswith("/static/") or not path.startswith(("/api/", "/demo-files/"))) and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.middleware("http")
async def language(request: Request, call_next):
    """The EN/TR switch's choice (cookie) for this request: error texts and download names (ADR-033)."""
    token = i18n.use(request.cookies.get(i18n.COOKIE))
    try:
        return await call_next(request)
    finally:
        i18n.reset(token)


@app.middleware("http")
async def password(request: Request, call_next):
    """HTTP basic auth when CADAI_PASSWORD is set (any user name). Off on this computer only."""
    if config.PASSWORD:
        header = request.headers.get("authorization", "")
        ok = False
        if header.lower().startswith("basic "):
            try:
                _, _, given = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                ok = secrets.compare_digest(given, config.PASSWORD)
            except Exception:  # noqa: BLE001 - a malformed header is just a wrong password
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="cad-ai"'})
    return await call_next(request)


def _register_jobs() -> None:
    from cadai.cleanup import apply, legacy, train

    runner.register("inventory", say("Envanteri tara", "Scan the inventory"), lambda job: inventory.scan(log=job.log)["summary"])
    runner.register("convert", say("Çizimleri dönüştür (DWG → sahne)", "Convert drawings (DWG → scene)"), process.convert_all)
    runner.register("label", say("Temizlik etiketlerini çıkar (hizalama)", "Extract cleanup labels (alignment)"), process.label_all)
    runner.register("train_cleanup", say("Temizlik modelini eğit", "Train the cleanup model"), train.train_job)
    runner.register("clean", say("Çizimi temizle (model)", "Clean a drawing"), apply.clean_job)
    runner.register("clean_legacy", say("Çizimi temizle (eski sistem)", "Clean a drawing (old system)"), legacy.clean_legacy_job)
    runner.register("symbols", say("Sembol kataloğunu çıkar (son çizimler)", "Build the symbol catalogue (final drawings)"), symbols.harvest_job)
    runner.register("propose_symbols", say("Sembol önerileri (yanındaki yazılardan)", "Symbol proposals (from nearby text)"), symbols.propose_job)
    runner.register("symbol_shapes", say("Patlatılmış sembollerin şekilleri", "Shapes of exploded symbols"), shapes.collect_job)
    runner.register("devices", say("Cihazları ham çizime taşı", "Move devices onto the raw drawing"), devices.build_job)
    runner.register("furniture", say("Mobilya kataloğunu çıkar (ham çizimler)", "Build the furniture catalogue (raw drawings)"), furniture.harvest_job)
    runner.register("furniture_shapes", say("Mobilya şekil sınıfları (veri setindeki bütün mobilya)", "Furniture shape classes (all furniture in the dataset)"), furniture_shapes.cluster_job)
    runner.register("placement_data", say("Yerleştirme veri setini kur", "Build the placement dataset"), placement_dataset.build_job)
    runner.register("train_placement", say("Yerleştirme modelini eğit", "Train the placement model"), placement_model.train_job)
    runner.register("place", say("Cihazları yerleştir (temiz çizime)", "Place devices (on the clean drawing)"), place.place_job)
    from cadai.correct import cleanup as correct_cleanup, devices as correct_devices
    runner.register("export_devices", say("Düzeltilmiş yerleşimi yaz (DWG)", "Write the corrected placement (DWG)"), correct_devices.export_job)
    runner.register("export_clean", say("Düzeltilmiş temizliği yaz (DWG)", "Write the corrected cleanup (DWG)"), correct_cleanup.export_job)


_register_jobs()


# ------------------------------------------------------------------------------- pages
@app.get("/")
def page_index():
    return FileResponse(STATIC / "index.html")


@app.get("/project/{pid}")
def page_project(pid: str):
    return FileResponse(STATIC / "project.html")


@app.get("/clean")
def page_clean():
    return FileResponse(STATIC / "clean.html")


@app.get("/draw")
def page_draw():
    return FileResponse(STATIC / "draw.html")


@app.get("/edit")
def page_edit():
    return FileResponse(STATIC / "edit.html")


@app.get("/lexicon/{kind}")
def page_lexicon(kind: str):
    return FileResponse(STATIC / "lexicon.html")


@app.get("/debug")
def page_debug():
    return FileResponse(STATIC / "debug.html")


@app.get("/demo")
def page_demo():
    return FileResponse(STATIC / "demo.html")


DEMO = Path(__file__).resolve().parents[2] / "demo"


@app.get("/demo-files/{key}/{name}")
def demo_file(key: str, name: str):
    """The anonymised demo projects in the repository (ADR-032)."""
    path = (DEMO / key / name).resolve()
    if DEMO.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=f"{key}-{name}" if path.suffix == ".dwg" else None)


@app.get("/api/demo")
def demo_list():
    keys = json.loads((DEMO / "projects.json").read_text(encoding="utf-8")) if (DEMO / "projects.json").exists() else []
    return [json.loads((DEMO / k / "facts.json").read_text(encoding="utf-8")) for k in keys]


@app.get("/static/{name}")
def static_file(name: str):
    path = STATIC / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


# ------------------------------------------------------------------------------- data
def _project_status(p: dict) -> dict:
    shas = {f["sha"] for f in p["files"]}
    scenes = {s: process.scene_info(s) for s in shas}
    pairs = []
    for pair in p["pairs"]:
        lab = process.label_info(p["id"], pair["final"])
        pairs.append({**pair, "why": i18n.bi(pair.get("why", "")), "split": split_of(p["id"]),
                      "label": {k: lab.get(k) for k in ("ok", "counts", "error", "seconds")} if lab else None})
    return {
        "id": p["id"],
        "folder": p["folder"],
        "files": [{**f, "scene": scenes[f["sha"]] and {k: scenes[f["sha"]].get(k) for k in
                   ("ok", "items", "unit", "error", "seconds")}} for f in p["files"]],
        "pairs": pairs,
    }


@app.get("/api/overview")
def overview():
    inv = inventory.load()
    scene_infos = [json.loads(p.read_text(encoding="utf-8")) for p in config.SCENE_DIR.glob("*.json")]
    label_infos = [json.loads(p.read_text(encoding="utf-8")) for p in config.LABEL_DIR.glob("*.json")]
    pair_count = sum(1 for p in inv["projects"] for x in p["pairs"] if x["raw"])
    return {
        "inventory": inv.get("summary", {}),
        "root": inv.get("root"),
        "steps": [
            {"key": "inventory", "title": say("Envanter", "Inventory"), "done": len(inv["projects"]), "total": len(inv["projects"])},
            {"key": "convert", "title": say("Dönüştürme", "Conversion"), "done": sum(1 for s in scene_infos if s["ok"]),
             "failed": sum(1 for s in scene_infos if not s["ok"]), "total": 2 * pair_count},
            {"key": "label", "title": say("Etiketleme", "Labelling"), "done": sum(1 for s in label_infos if s.get("ok")),
             "failed": sum(1 for s in label_infos if not s.get("ok")), "total": pair_count},
        ],
        "models": _models(),
    }


@app.get("/api/projects")
def projects():
    return [_project_status(p) for p in inventory.load()["projects"]]


@app.get("/api/projects/{pid}")
def project(pid: str):
    for p in inventory.load()["projects"]:
        if p["id"] == pid:
            st = _project_status(p)
            for pair in st["pairs"]:
                lab = process.label_info(pid, pair["final"])
                pair["label_full"] = {**lab, "log": [i18n.bi(line) for line in lab.get("log") or []]} if lab else lab
            return st
    raise HTTPException(404)


@app.get("/api/projects/{pid}/labels/{final}.png")
def label_png(pid: str, final: str):
    path = config.LABEL_DIR / f"{pid}__{final[:16]}.png"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


def _models() -> list[dict]:
    out = []
    for d in sorted(config.MODEL_DIR.glob("*/report.json"), reverse=True):
        rep = json.loads(d.read_text(encoding="utf-8"))
        rep["name"] = d.parent.name
        out.append(rep)
    return out


@app.get("/api/models")
def models():
    return _models()


@app.get("/api/models/{name}/png/{file}")
def model_png(name: str, file: str):
    path = (config.MODEL_DIR / name / file).resolve()
    if config.MODEL_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


# ------------------------------------------------------------------------------- cleaning new drawings
OUTPUTS = config.DATA / "outputs"
UPLOADS = config.DATA / "uploads"


@app.post("/api/clean/upload")
async def clean_upload(file: UploadFile = File(...), keep_furniture: bool = Form(False), mode: str = Form("agent")):
    if mode not in CLEAN_MODES:
        raise HTTPException(400, str(say(f"bilinmeyen yöntem: {mode}", f"unknown method: {mode}")))
    name = Path(file.filename or "cizim.dwg").name
    if Path(name).suffix.lower() not in (".dwg", ".dxf"):
        raise HTTPException(400, str(say("Yalnızca DWG veya DXF.", "DWG or DXF only.")))
    UPLOADS.mkdir(parents=True, exist_ok=True)
    target = UPLOADS / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{name}"
    with open(target, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    return runner.submit("clean", path=str(target), keep_furniture=keep_furniture, mode=mode).summary()


# agent: toolbox + Claude (ADR-021); tools: toolbox only; auto / layers / items: layer method (ADR-020)
CLEAN_MODES = ("agent", "tools", "auto", "layers", "items")


@app.post("/api/clean/archive")
def clean_archive(sha: str = Body(...), engine: str = Body("agent")):
    info = inventory.files_by_sha(inventory.load()).get(sha)
    if not info:
        raise HTTPException(404, str(say("arşivde yok", "not in the archive")))
    path = str(config.PROJECTS / info["rel"])
    if engine == "legacy":
        return runner.submit("clean_legacy", path=path).summary()
    mode = "auto" if engine == "model" else engine
    if mode not in CLEAN_MODES:
        raise HTTPException(400, str(say(f"bilinmeyen yöntem: {engine}", f"unknown method: {engine}")))
    return runner.submit("clean", path=path, mode=mode).summary()


@app.get("/api/outputs")
def outputs():
    out = []
    # newest first by finish time: folder names start with the drawing name, not the date
    for d in sorted(OUTPUTS.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        row = json.loads(d.read_text(encoding="utf-8"))
        row["name"] = d.parent.name
        row["files"] = sorted(p.name for p in d.parent.iterdir())
        qf = d.parent / "questions.json"
        row["question_list"] = json.loads(qf.read_text(encoding="utf-8"))[:50] if qf.exists() else []
        cf = d.parent / "checks.json"
        row["issue_list"] = [{**i, "message": i18n.bi(i.get("message", ""))}
                             for i in json.loads(cf.read_text(encoding="utf-8"))["issues"][:100]] if cf.exists() else []
        af = d.parent / "agent.json"
        row["agent_record"] = _agent_record(json.loads(af.read_text(encoding="utf-8"))) if af.exists() else None
        pf = d.parent / "devices.json"
        row["placement"] = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else None
        lf = d.parent / "layers.json"
        row["layers"] = json.loads(lf.read_text(encoding="utf-8")) if lf.exists() else []
        out.append(row)
    return out


def _agent_record(rec: dict) -> dict:
    """Claude's notes in both languages; records written before ADR-033 are Turkish only."""
    rec["views"] = i18n.bi(rec.get("views", ""))
    for call in rec.get("calls", []):
        for r in call.get("regions") or []:
            r["what"] = i18n.bi(r.get("what", ""))
        if call.get("fixes"):
            call["fixes"]["notes"] = i18n.bi(call["fixes"].get("notes", ""))
    return rec


@app.post("/api/place")
def place_output(output: str = Body(...), instruction: str = Body(""), rules: list | None = Body(None)):
    """Devices on a cleaned drawing (Çizim page, ADR-023); `instruction`: the engineer's words (ADR-027),
    `rules`: the rules the engineer confirmed (ADR-028)."""
    if not (OUTPUTS / output / "clean.dxf").is_file():
        raise HTTPException(404, str(say("temizlik sonucu yok", "no cleanup result")))
    return runner.submit("place", output=output, instruction=instruction[:1000], rules=rules).summary()


@app.post("/api/steer")
def steer_rules(instruction: str = Body(..., embed=True)):
    """The instruction as rules, for the engineer to check before placing (ADR-028)."""
    from cadai.placement import steer
    try:
        rules, cost = steer.translate(instruction.strip()[:1000], place.PLACE_CLASSES)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"rules": rules, "cost": cost}


@app.get("/api/rulebook")
def rulebook():
    """The office rule book with what a rule can name (Çizim page)."""
    from cadai.placement import steer
    return {"rules": steer.load_book(place.PLACE_CLASSES), "classes": place.PLACE_CLASSES,
            "room_types": steer.ROOM_TYPES, "tags": list(steer.TAGS)}


@app.put("/api/rulebook")
def save_rulebook(rules: list = Body(..., embed=True)):
    """Rules applied to every placement until the engineer removes them (ADR-028)."""
    from cadai.placement import steer
    return steer.save_book(rules, place.PLACE_CLASSES)


@app.get("/api/tools")
def tool_legend():
    """The toolbox's tools with their picture colours (ADR-021)."""
    from cadai.cleanup import tools
    return [{"name": t, "label": tools.LABELS[t], "colour": "rgb(%d,%d,%d)" % tools.COLOURS[t]} for t in tools.TOOLS]


@app.get("/api/layer-types")
def layer_types():
    """Layer types the engineer can choose (ADR-020) and the dictionary confirmed so far."""
    from cadai.cleanup import layers
    return {"labels": layers.LABELS, "confirmed": layers.confirmed()}


@app.post("/api/outputs/{name}/relayer")
def output_relayer(name: str, types: dict = Body(...), mode: str = Body("auto")):
    """The engineer corrected layer types on a result: remember them (for every later drawing too)
    and clean the same drawing again."""
    from cadai.cleanup import layers
    summary = OUTPUTS / name / "summary.json"
    if not summary.exists():
        raise HTTPException(404)
    if mode not in CLEAN_MODES:
        raise HTTPException(400, str(say(f"bilinmeyen yöntem: {mode}", f"unknown method: {mode}")))
    info = json.loads(summary.read_text(encoding="utf-8"))
    try:
        layers.confirm(types)
    except ValueError as exc:
        raise HTTPException(400, str(say(f"bilinmeyen katman tipi: {exc}", f"unknown layer type: {exc}")))
    return runner.submit("clean", path=info["source"], keep_furniture=bool(info.get("keep_furniture")),
                         mode=mode).summary()


# ------------------------------------------------------------------------------- corrections (Düzelt, ADR-029)
_LOADED: dict = {}
_LOADED_LOCK = threading.Lock()


def _out_dir(name: str) -> Path:
    path = (OUTPUTS / name).resolve()
    if OUTPUTS.resolve() not in path.parents or not (path / "summary.json").exists():
        raise HTTPException(404, str(say("çıktı yok", "no such output")))
    return path


def _loaded(out_dir: Path):
    """The output's scene and decisions, kept in memory for the last two outputs (every pan reads them)."""
    from cadai.correct import cleanup
    stamp = (out_dir / "decision.npz").stat().st_mtime if (out_dir / "decision.npz").exists() else 0
    with _LOADED_LOCK:
        hit = _LOADED.get(out_dir)
        if hit and hit[0] == stamp:
            return hit[1]
    data = cleanup.load(out_dir)
    with _LOADED_LOCK:
        _LOADED.pop(out_dir, None)
        if len(_LOADED) >= 2:
            _LOADED.pop(next(iter(_LOADED)))
        _LOADED[out_dir] = ((out_dir / "decision.npz").stat().st_mtime, data)
    return data


def _user(request: Request) -> str:
    """The name typed at the password prompt (any name is accepted; the corrections log records it)."""
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return ""
    try:
        return base64.b64decode(header[6:]).decode("utf-8").partition(":")[0][:60]
    except Exception:  # noqa: BLE001 - no name is fine
        return ""


@app.get("/api/outputs/{name}/view/{mode}.png")
def output_view(name: str, mode: str, x0: float, y0: float, x1: float, y1: float, w: int = 1600):
    """The drawing inside a box, for the Düzelt page: `place` = the kept drawing, `clean` = kept dark,
    deleted pale, the engineer's keep green and delete orange."""
    import io
    from cadai.cad.render import render
    from cadai.correct import cleanup
    if mode not in ("place", "clean") or not (x1 > x0 and y1 > y0):
        raise HTTPException(400)
    out_dir = _out_dir(name)
    data = _loaded(out_dir)
    buf = io.BytesIO()
    w = max(100, min(int(w), 3200))
    if mode == "place":
        code = np.where(data.kept & (data.scene.text == ""), 1, -9)
        render(data.scene, buf, code, bbox=(x0, y0, x1, y1), width=w, colours={1: (70, 70, 70)})
    else:
        code = cleanup.codes(data, cleanup.overrides(out_dir))
        render(data.scene, buf, code, bbox=(x0, y0, x1, y1), width=w,
               colours={cleanup.DELETED: (240, 196, 196), cleanup.KEPT: (40, 40, 40),
                        cleanup.SAY_KEEP: (0, 160, 70), cleanup.SAY_DELETE: (240, 120, 0)},
               thick={cleanup.SAY_KEEP: 2, cleanup.SAY_DELETE: 2})
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/outputs/{name}/devices/state")
def devices_state(name: str):
    from cadai.correct import devices
    try:
        return {**devices.view(devices.load(_out_dir(name))), "reasons": devices.REASONS,
                "classes": place.PLACE_CLASSES, "names": place.NAMES}
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/outputs/{name}/devices/edit")
def devices_edit(name: str, request: Request, body: dict = Body(...)):
    from cadai.correct import devices
    out_dir = _out_dir(name)
    data = _loaded(out_dir)
    s = data.scene
    walls = s.segs[data.kept[s.seg_item] & (s.text == "")[s.seg_item] & np.isfinite(s.segs).all(1)]
    body = dict(body)
    action = body.pop("action", "")
    try:
        entry = devices.edit(out_dir, action, user=_user(request), segs=walls, **body)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc))
    return {"entry": entry, **devices.view(devices.load(out_dir))}


@app.get("/api/outputs/{name}/cleanup/state")
def cleanup_state(name: str):
    from cadai.correct import cleanup
    out_dir = _out_dir(name)
    data = _loaded(out_dir)
    bb = data.scene.bboxes()[data.kept]
    bb = bb[np.isfinite(bb).all(1)]
    box = ([float(np.percentile(bb[:, 0], 0.5)) - 2, float(np.percentile(bb[:, 1], 0.5)) - 2,
            float(np.percentile(bb[:, 2], 99.5)) + 2, float(np.percentile(bb[:, 3], 99.5)) + 2] if len(bb) else None)
    ov = cleanup.overrides(out_dir)
    return {"bbox": box, "reasons": cleanup.REASONS, "has_why": data.why is not None,
            "say_keep": sum(ov.values()), "say_delete": sum(not v for v in ov.values()),
            "marks": cleanup.marks(out_dir)}


@app.get("/api/outputs/{name}/cleanup/pick")
def cleanup_pick(name: str, x: float, y: float, tol: float = 0.2, leaf: bool = False,
                 x1: float | None = None, y1: float | None = None):
    """The entity at (x, y), or with x1 / y1 everything inside the box."""
    from cadai.correct import cleanup
    out_dir = _out_dir(name)
    data = _loaded(out_dir)
    items = (cleanup.box(data, min(x, x1), min(y, y1), max(x, x1), max(y, y1)) if x1 is not None and y1 is not None
             else cleanup.pick(data, x, y, max(0.01, min(tol, 5.0)), leaf))
    return cleanup.describe(data, items, cleanup.overrides(out_dir))


@app.post("/api/outputs/{name}/cleanup/edit")
def cleanup_edit(name: str, request: Request, body: dict = Body(...)):
    from cadai.correct import cleanup
    out_dir = _out_dir(name)
    data = _loaded(out_dir)
    try:
        if body.get("action") == "undo":
            entry = cleanup.undo(out_dir, user=_user(request))
        else:
            entry = cleanup.edit(out_dir, data, body.get("items") or [], body.get("keep"),
                                 body.get("reasons") or [], body.get("note") or "", user=_user(request))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"seq": entry["seq"], **cleanup_state(name)}


@app.get("/api/outputs/{name}/edits/log")
def edits_log(name: str):
    """Every correction of this output with the reasons given later merged in, newest first."""
    from cadai.correct import store
    entries = store.read(_out_dir(name))
    later = store.reasons(entries)
    undone = {e["target"] for e in entries if e.get("action") == "undo"}
    out = []
    for e in reversed(entries):
        if e.get("action") in ("reason", "undo") or e["seq"] in undone:
            continue
        e = {**e, **later.get(e["seq"], {})}
        e.pop("items", None)
        e.pop("before", None)
        e.pop("system_kept", None)
        out.append(e)
    return out


@app.post("/api/outputs/{name}/export/{what}")
def output_export(name: str, what: str):
    """Write the corrected drawing as a job (`devices` or `clean`)."""
    _out_dir(name)
    if what not in ("devices", "clean"):
        raise HTTPException(404)
    return runner.submit("export_devices" if what == "devices" else "export_clean", output=name).summary()


@app.get("/api/outputs/{name}/layers/{index}.png")
def output_layer_png(name: str, index: int):
    path = (OUTPUTS / name / "layers" / f"{index}.png").resolve()
    if OUTPUTS.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/outputs/{name}/{file}")
def output_file(name: str, file: str):
    path = (OUTPUTS / name / file).resolve()
    if OUTPUTS.resolve() not in path.parents or not path.is_file():
        raise HTTPException(404)
    if path.suffix not in (".dwg", ".dxf"):
        return FileResponse(path)
    return FileResponse(path, filename=download_name(path))


def download_name(path: Path) -> str:
    """"<drawing> - katman temizliği.dwg" (or the other method; English when the switch says so), not "clean.dwg"."""
    summary = path.parent / "summary.json"
    info = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else {}
    stem = Path(info.get("source", path.parent.name)).stem
    corrected = say(" (düzeltilmiş)", " (corrected)") if path.stem.endswith("_corrected") else ""
    if path.stem.startswith("placed"):
        return f"{stem} - {say('elektrik yerleşimi', 'electrical layout')}{corrected}{path.suffix}"
    method = (say("eski sistem temizliği", "old system cleanup") if info.get("engine") == "legacy" else
              say("Claude temizliği", "Claude cleanup") if info.get("mode") == "agent" else
              say("araç temizliği", "tools cleanup") if info.get("mode") == "tools" else
              say("katman temizliği", "layer cleanup") if info.get("mode") == "layers" else
              say("model temizliği", "model cleanup"))
    return f"{stem} - {method}{corrected}{path.suffix}"


# ------------------------------------------------------------------------------- lexicons
# One-time naming of what the data can't name itself: electrical symbols (by block name) and
# furniture (by geometry signature). An LLM proposes, the engineer confirms.
LEXICONS = {
    "symbols": {"module": symbols, "key": "block", "types": symbols.DEVICE_TYPES,
                "title": say("Elektrik sembolleri", "Electrical symbols"), "min_count": ("files", 2)},
    "furniture": {"module": furniture, "key": "sig", "types": furniture.FURNITURE_TYPES,
                  "title": say("Mobilya ve vitrifiye", "Furniture and sanitary ware"), "min_count": ("projects", 1)},
    # ADR-025: blocks and loose lines grouped by shape; one name covers the whole class
    "furniture_shapes": {"module": furniture_shapes, "key": "sig", "types": furniture.FURNITURE_TYPES,
                         "title": say("Mobilya şekilleri", "Furniture shapes"), "min_count": ("projects", furniture_shapes.MIN_PROJECTS)},
}


def _lexicon(kind: str) -> dict:
    if kind not in LEXICONS:
        raise HTTPException(404)
    return LEXICONS[kind]


def _lexicon_dir(kind: str):
    mod = _lexicon(kind)["module"]
    return getattr(mod, "LEXICON_DIR", None) or (mod.SYMBOL_DIR if kind == "symbols" else mod.FURNITURE_DIR)


@app.get("/api/lexicon/{kind}")
def lexicon(kind: str):
    lx = _lexicon(kind)
    mod = lx["module"]
    catalog = json.loads(mod.CATALOG.read_text(encoding="utf-8")) if mod.CATALOG.exists() else []
    field, minimum = lx["min_count"]
    catalog = [c for c in catalog if c.get("thumb") and c[field] >= minimum]
    catalog = sorted(catalog, key=lambda c: -c["instances"])[:300]      # most used first
    proposals = {k: {**v, "why": i18n.bi(v.get("why", "")), "confidence": i18n.bi(v.get("confidence", ""))}
                 for k, v in mod.load_proposals().items()}
    return {"title": lx["title"], "key": lx["key"], "catalog": catalog, "types": mod.load_types(),
            "proposals": proposals, "type_names": lx["types"], "notes": notes.load(_lexicon_dir(kind))}


@app.post("/api/lexicon/{kind}/type")
def lexicon_type(kind: str, key: str = Body(...), type: str = Body(...)):
    try:
        _lexicon(kind)["module"].set_type(key, type)
    except ValueError:
        raise HTTPException(400, str(say(f"bilinmeyen tip: {type}", f"unknown type: {type}")))
    return {"ok": True}


@app.post("/api/lexicon/{kind}/note")
def lexicon_note(kind: str, key: str = Body(...), text: str = Body(...)):
    """The engineer's own description of an entry, beyond the type list."""
    return notes.save(_lexicon_dir(kind), key, text)


@app.get("/api/lexicon/{kind}/{fid}.png")
def lexicon_png(kind: str, fid: str):
    path = _lexicon_dir(kind) / f"{fid}.png"
    if not fid.isalnum() or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


# ------------------------------------------------------------------------------- jobs
@app.get("/api/jobs")
def job_list():
    return runner.list()


@app.post("/api/jobs/{kind}")
def job_submit(kind: str, params: dict = Body(default={})):
    if kind not in runner.kinds:
        raise HTTPException(404, str(say(f"bilinmeyen iş: {kind}", f"unknown job: {kind}")))
    return runner.submit(kind, **params).summary()


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    runner.cancel(job_id)
    return {"ok": True}


# ------------------------------------------------------------------------------- debug
@app.get("/api/debug/system")
def debug_system():
    from cadai.cad import oda
    try:
        oda_path = str(oda.oda_binary())
    except Exception as exc:  # noqa: BLE001
        oda_path = str(say(f"YOK: {exc}", f"MISSING: {exc}"))
    disks = {}
    for root in ("C:\\", str(config.DATA.anchor)):
        try:
            u = shutil.disk_usage(root)
            disks[root] = {"free_gb": round(u.free / 2**30, 1), "total_gb": round(u.total / 2**30, 1)}
        except OSError:
            pass
    counts = {name: len(list(d.glob(pattern))) for name, d, pattern in (
        ("dxf", config.DXF_DIR, "*.dxf"), ("scenes", config.SCENE_DIR, "*.npz"),
        ("labels", config.LABEL_DIR, "*.npz"), ("models", config.MODEL_DIR, "*/report.json"))}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "paths": {"code": str(config.CODE), "data": str(config.DATA), "projects": str(config.PROJECTS)},
        "oda": oda_path,
        "disks": disks,
        "memory": _memory(),
        "cache_counts": counts,
        "workers": process.WORKERS,
    }


@app.get("/api/debug/logs")
def debug_logs(level: str = "INFO", limit: int = 300):
    order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    min_i = order.index(level) if level in order else 1
    rows = [r for r in log.RECENT if order.index(r["level"]) >= min_i]
    return rows[-limit:]


@app.get("/api/debug/failures")
def debug_failures():
    out = []
    for kind, d in ((say("dönüştürme", "conversion"), config.SCENE_DIR), (say("etiketleme", "labelling"), config.LABEL_DIR)):
        for p in d.glob("*.json"):
            info = json.loads(p.read_text(encoding="utf-8"))
            if not info.get("ok"):
                out.append({"step": kind, "file": info.get("src") or info.get("project"),
                            "error": i18n.bi(info.get("error") or say("hizalama bulunamadı", "no alignment found")),
                            "trace": info.get("trace"), "log": [i18n.bi(line) for line in info.get("log") or []]})
    return out


@app.exception_handler(Exception)
def _errors(request, exc):
    log.get("web").exception("request error %s", request.url.path)
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)
