"""Batch steps over the inventory, run as jobs from the website:

- `convert_all`: DWG -> DXF (ODA) -> scene (.npz) for every raw and final in a pair;
- `label_all`: raw/final alignment -> cleanup labels + a preview PNG per pair.

Each file is handled in a separate process (ezdxf on a 50 MB drawing holds a few GB), results are
cached by content hash, so re-running only does what is missing.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from cadai import config
from cadai import i18n
from cadai.i18n import say
from cadai.corpus import inventory
from cadai.sysinfo import memory

# upper bound; the real count follows free memory (see workers()). CADAI_WORKERS=1 when the machine is short.
WORKERS = int(os.environ.get("CADAI_WORKERS", "4"))
GB_PER_WORKER = 2.5         # a 50 MB raw drawing peaks around 2 GB in ezdxf


def workers(busy: int = 0) -> int:
    """Parallel processes for the memory right now: the `busy` workers already hold theirs, each
    further one needs GB_PER_WORKER free (other programs may hold most of the memory)."""
    free = memory().get("free_gb")
    if free is None:
        return 2
    return int(max(1, min(WORKERS, busy + free // GB_PER_WORKER)))


def key(sha: str) -> str:
    return sha[:16]


def scene_path(sha: str) -> Path:
    return config.SCENE_DIR / f"{key(sha)}.npz"


def scene_info(sha: str) -> dict | None:
    p = config.SCENE_DIR / f"{key(sha)}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def label_path(project: str, final_sha: str) -> Path:
    return config.LABEL_DIR / f"{project}__{key(final_sha)}.npz"


def label_info(project: str, final_sha: str) -> dict | None:
    p = label_path(project, final_sha).with_suffix(".json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ------------------------------------------------------------------------------- workers
def _convert_one(sha: str, src: str) -> dict:
    from cadai.cad import oda
    from cadai.cad.scene import build
    t0 = time.time()
    info = {"sha": sha, "src": src, "ok": False}
    try:
        dxf = config.DXF_DIR / f"{key(sha)}.dxf"
        if not dxf.exists():
            oda.to_dxf(Path(src), dxf)
        info["seconds_convert"] = round(time.time() - t0, 1)
        scene = build(dxf)
        scene.save(scene_path(sha))
        m = scene.meta
        info.update(ok=True, items=scene.n, segments=len(scene.segs), unit=m["unit"], layers=len(m["layers"]),
                    audit_fixes=m["audit_fixes"], failed_items=m["failed_items"], dxf_mb=round(dxf.stat().st_size / 1e6, 1))
    except Exception as exc:  # noqa: BLE001 - reported per file on the debug panel
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["trace"] = traceback.format_exc()[-3000:]
    info["seconds"] = round(time.time() - t0, 1)
    (config.SCENE_DIR / f"{key(sha)}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


def _label_one(project: str, raw_sha: str, final_sha: str) -> dict:
    from cadai.cad.render import render
    from cadai.cad.scene import Scene
    from cadai.cleanup.align import align
    t0 = time.time()
    out = label_path(project, final_sha)
    info = {"project": project, "raw": raw_sha, "final": final_sha, "ok": False}
    try:
        raw, final = Scene.load(scene_path(raw_sha)), Scene.load(scene_path(final_sha))
        lines: list = []                        # {"tr", "en"} per line (ADR-033)
        res = align(raw, final, log=lambda m: lines.append(m.both() if hasattr(m, "both") else str(m)))
        np.savez_compressed(out, labels=res["labels"], frac=res["frac"], shadow=res["shadow"])
        info.update(ok=res["ok"], counts=res["counts"], transforms=res["transforms"], log=lines)
        regions = [t["region"] for t in res["transforms"]]
        if regions:
            r = np.array(regions)
            bbox = (r[:, 0].min() - 5, r[:, 1].min() - 5, r[:, 2].max() + 5, r[:, 3].max() + 5)
            render(raw, out.with_suffix(".png"), res["labels"], bbox=bbox, width=2400)
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["trace"] = traceback.format_exc()[-3000:]
    info["seconds"] = round(time.time() - t0, 1)
    out.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return info


# ------------------------------------------------------------------------------- jobs
def _run_pool(job, tasks: list[tuple], fn, describe) -> list[dict]:
    """Run `fn(*task)` in worker processes. The number of tasks in flight follows the free memory,
    re-checked after every finished task, so closing a big program speeds the job up. A task that
    ran out of memory beside the others is run once more at the end, alone (4 of 247 dataset pairs
    were lost that way)."""
    results: list[dict] = []
    job.progress(0, len(tasks))
    pending = list(tasks)
    running: dict = {}
    retry: list[tuple] = []          # ran out of memory beside others: once more, alone, at the end
    alone = False
    limit = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        while pending or running or retry:
            if not pending and not running:
                pending, retry, alone = retry, [], True
                job.log(say(f"bellek yetmeyen {len(pending)} iş tek başına yeniden deneniyor",
                            f"retrying {len(pending)} jobs that ran out of memory, one at a time"))
            now = 1 if alone else workers(busy=len(running))
            if now != limit:
                limit = now
                job.log(say(f"{limit} paralel süreç (boş belleğe göre)", f"{limit} parallel processes (by free memory)"))
            while pending and len(running) < limit and not job.cancelled:
                task = pending.pop(0)
                running[pool.submit(fn, *task)] = task
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                task = running.pop(fut)
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001 - a crashed worker process
                    res = {"ok": False, "error": f"process crashed: {exc}"}
                if not res.get("ok") and str(res.get("error", "")).startswith("MemoryError") and not alone:
                    retry.append(task)
                    said = i18n.bi(describe(task, res))
                    job.log(say(said["tr"] + " · sonra tek başına denenecek", said["en"] + " · will retry alone later"))
                    continue
                results.append(res)
                job.progress(len(results))
                job.log(describe(task, res))
            if job.cancelled and (pending or retry):
                job.log(say(f"iptal edildi; {len(pending) + len(retry)} iş başlatılmadı",
                            f"cancelled; {len(pending) + len(retry)} jobs not started"))
                pending.clear()
                retry.clear()
    return results


def convert_all(job, limit: int | None = None, retry_failed: bool = False) -> dict:
    inv = inventory.load()
    by_sha = inventory.files_by_sha(inv)
    wanted = []
    for p in inv["projects"]:
        for pair in p["pairs"]:
            if pair["raw"]:
                wanted += [pair["raw"], pair["final"]]
    tasks = []
    for sha in dict.fromkeys(wanted):
        info = scene_info(sha)
        if info and (info["ok"] or not retry_failed):
            continue
        tasks.append((sha, str(config.PROJECTS / by_sha[sha]["rel"])))
    tasks = tasks[:limit] if limit else tasks
    ready = len(set(wanted)) - len(tasks)
    job.log(say(f"{len(tasks)} dosya işlenecek ({ready} zaten hazır)", f"{len(tasks)} files to process ({ready} already done)"))

    def describe(task, res):
        name = Path(task[1]).name
        if res.get("ok"):
            return say(f"✓ {name}: {res['items']} nesne, birim {res['unit']['unit']}, {res['seconds']} s",
                       f"✓ {name}: {res['items']} items, unit {res['unit']['unit']}, {res['seconds']} s")
        return say(f"✗ {name}: {res.get('error')}", f"✗ {name}: {res.get('error')}")

    results = _run_pool(job, tasks, _convert_one, describe)
    return {"processed": len(results), "failed": sum(1 for r in results if not r.get("ok"))}


def label_all(job, limit: int | None = None, force: bool = False) -> dict:
    inv = inventory.load()
    tasks = []
    for p in inv["projects"]:
        for pair in p["pairs"]:
            if not pair["raw"]:
                continue
            ready = all((scene_info(s) or {}).get("ok") for s in (pair["raw"], pair["final"]))
            if ready and (force or not label_info(p["id"], pair["final"])):
                tasks.append((p["id"], pair["raw"], pair["final"]))
    tasks = tasks[:limit] if limit else tasks
    job.log(say(f"{len(tasks)} çift etiketlenecek", f"{len(tasks)} pairs to label"))

    def describe(task, res):
        if res.get("ok"):
            c, n = res["counts"], len(res["transforms"])
            return say(f"✓ {task[0]}: {n} görünüş, kaldı {c['keep']}, silindi {c['delete']}, {res['seconds']} s",
                       f"✓ {task[0]}: {n} views, kept {c['keep']}, deleted {c['delete']}, {res['seconds']} s")
        return say(f"✗ {task[0]}: {res.get('error') or 'hizalanamadı'}", f"✗ {task[0]}: {res.get('error') or 'not aligned'}")

    results = _run_pool(job, tasks, _label_one, describe)
    return {"labelled": len(results), "aligned": sum(1 for r in results if r.get("ok"))}
