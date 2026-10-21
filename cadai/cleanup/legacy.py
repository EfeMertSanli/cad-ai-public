"""The previous system's cleanup (cad-llm-bridge rule engine), run from this website (ADR-017).

    python -m corpus.pipeline clean <raw> --out <folder> --store <store>   (in the old repo, its own venv)

The old repo stays untouched apart from the starting cleanup profiles it writes for unknown
architect offices (git-ignored `auto_*` files); its training store goes to D: (C: has no room).
The output gets the same files and summary as the model's cleanup, so the Temizle page shows both
side by side, plus the engineer's own cleanup when the drawing comes from the archive.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from cadai import config
from cadai.i18n import say
from cadai.cad import oda
from cadai.cad.render import render
from cadai.cad.scene import build
from cadai.check.cleanup_checks import check_cleanup
from cadai.cleanup import compare

REPO = Path(os.environ.get("CADAI_LEGACY_REPO", r"C:\Users\aztek\cad-llm-bridge"))
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
STORE = config.DATA / "legacy" / "store"
OUTPUT_DIR = config.DATA / "outputs"


def plan_bbox(scene, keep):
    bb = scene.bboxes()[keep]
    bb = bb[np.isfinite(bb[:, 0])]
    return (np.percentile(bb[:, 0], 0.5) - 2, np.percentile(bb[:, 1], 0.5) - 2,
            np.percentile(bb[:, 2], 99.5) + 2, np.percentile(bb[:, 3], 99.5) + 2)


def clean_legacy_job(job, path: str) -> dict:
    if not PYTHON.exists():
        raise RuntimeError(str(say(f"Eski sistem bulunamadı: {PYTHON}", f"Old system not found: {PYTHON}")))
    src = Path(path)
    out = OUTPUT_DIR / f"{src.stem[:40]}-eski-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    work = out / "legacy"
    work.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CLB_LEARNSTORE": str(STORE), "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    job.progress(0, 4)
    proc = subprocess.Popen([str(PYTHON), "-m", "corpus.pipeline", "clean", str(src), "--out", str(work),
                             "--store", str(STORE)], cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for line in proc.stderr:                       # the old pipeline logs progress on stderr
        job.log(say("eski: " + line.rstrip()[:300], "old: " + line.rstrip()[:300]))
    stdout = proc.stdout.read()
    if proc.wait() != 0:
        raise RuntimeError(str(say(f"Eski sistem hata verdi (kod {proc.returncode}): {stdout[-600:]}",
                                   f"The old system failed (code {proc.returncode}): {stdout[-600:]}")))
    info = json.loads(stdout[stdout.index("{"):])
    job.log(say(f"eski sistem bitti: {time.time() - t0:.0f} s, profil {info.get('office')}",
                f"old system done: {time.time() - t0:.0f} s, profile {info.get('office')}"))
    job.progress(1)

    clean_dxf = out / "clean.dxf"
    clean_dxf.write_bytes(Path(info["clean"]).read_bytes())
    try:
        oda.to_dwg(clean_dxf, out / "clean.dwg")
    except Exception as exc:  # noqa: BLE001 - the DXF is still usable
        job.log(say(f"DWG yazılamadı: {exc}", f"could not write the DWG: {exc}"))
    job.progress(2)

    # pictures and the same room check as the model's output: rooms of the clean drawing against
    # the area stamps of the raw drawing
    raw_dxf = next(work.glob("work/*.dxf"), None)
    raw_scene = build(raw_dxf) if raw_dxf else None
    clean_scene = build(clean_dxf, unit=raw_scene.meta["unit"] if raw_scene else None)
    render(clean_scene, out / "clean.png", np.ones(clean_scene.n, dtype=np.int8),
           bbox=plan_bbox(clean_scene, np.ones(clean_scene.n, bool)), colours={1: (40, 40, 40)}, width=2400)
    check = {"rooms": None, "issues": []}
    if raw_scene is not None:
        check = check_cleanup(clean_scene, np.ones(clean_scene.n, bool), stamp_scene=raw_scene)
    (out / "checks.json").write_text(json.dumps(check, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "questions.json").write_text("[]", encoding="utf-8")
    job.progress(3)
    summary = {"source": str(src), "model": "eski sistem (kural motoru)", "engine": "legacy",
               "keep_furniture": False, "counts": {"delete": None, "keep": clean_scene.n, "question": 0},
               "questions": 0, "seconds": round(time.time() - t0, 1), "unit": clean_scene.meta["unit"],
               "legacy": {k: info.get(k) for k in ("office", "profile_created", "views", "engine")},
               "check": {k: v for k, v in check.items() if k != "issues"}, "issues": len(check["issues"])}
    if compare.engineer_png(src, out / "engineer.png"):
        job.log(say("mühendisin temizliği de çizildi (arşiv projesi)", "the engineer's cleanup was drawn too (archive project)"))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    job.progress(4)
    return {"output": out.name, **summary}
