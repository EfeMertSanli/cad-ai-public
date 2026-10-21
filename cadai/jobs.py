"""Background jobs with progress, logs and errors, shown live on the debug panel.

A job is a Python function `fn(job, **params)`. It reports through `job.log()` (a `say(tr, en)` message, or
plain text that reads the same in both languages, ADR-033) and
`job.progress(done, total)` and may check `job.cancelled`. Jobs run one at a time in a worker
thread (heavy steps parallelise inside the job with processes). Finished jobs are saved to
DATA/jobs/<id>.json so the history survives a restart.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from queue import Queue
from typing import Callable

from cadai import config, i18n, log
from cadai.i18n import Msg

_log = log.get("jobs")


@dataclass
class Job:
    id: str
    kind: str
    title: dict | str                # {"tr", "en"}; plain text in history saved before ADR-033
    params: dict
    status: str = "queued"           # queued | running | done | failed | cancelled
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started: float | None = None
    finished: float | None = None
    done: int = 0
    total: int = 0
    lines: list = field(default_factory=list)     # {"t", "tr", "en"}; plain text before ADR-033
    error: str | None = None
    result: dict | None = None
    cancel_requested: bool = False
    lang: str = "tr"                 # of whoever started it: exception texts and the server log use it

    # ----------------------------------------------------------- called from inside the job
    def log(self, msg: Msg | str) -> None:
        both = msg.both() if isinstance(msg, Msg) else {"tr": str(msg), "en": str(msg)}
        self.lines.append({"t": datetime.now().strftime("%H:%M:%S"), **both})
        del self.lines[:-500]                       # keep the tail only
        _log.info("[%s] %s", self.kind, both["tr"], extra={"en": f"[{self.kind}] {both['en']}"})

    def progress(self, done: int, total: int | None = None) -> None:
        self.done = done
        if total is not None:
            self.total = total

    @property
    def cancelled(self) -> bool:
        return self.cancel_requested

    # ----------------------------------------------------------- for the API
    def summary(self) -> dict:
        d = asdict(self)
        d["title"] = i18n.bi(self.title)
        d["lines"] = [line if isinstance(line, dict) else {"t": line[:8], **i18n.bi(line[9:])}
                      for line in self.lines[-200:]]
        end = self.finished or time.time()
        d["seconds"] = round(end - self.started, 1) if self.started else None
        return d


class JobRunner:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.kinds: dict[str, tuple[str, Callable]] = {}
        self.queue: Queue[str] = Queue()
        self._load_history()
        threading.Thread(target=self._worker, daemon=True, name="jobs").start()

    def register(self, kind: str, title: Msg, fn: Callable) -> None:
        self.kinds[kind] = (title, fn)

    def submit(self, kind: str, **params) -> Job:
        title, _ = self.kinds[kind]
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, title=i18n.bi(title), params=params, lang=i18n.lang())
        self.jobs[job.id] = job
        self.queue.put(job.id)
        _log.info("queued %s %s %s", job.id, kind, params)
        return job

    def cancel(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"

    def list(self) -> list[dict]:
        out = []
        for j in sorted(self.jobs.values(), key=lambda j: j.created, reverse=True):
            d = j.summary()
            if isinstance(j.title, str) and j.kind in self.kinds:      # saved before ADR-033: today's title
                d["title"] = i18n.bi(self.kinds[j.kind][0])
            out.append(d)
        return out

    def _worker(self) -> None:
        while True:
            job = self.jobs[self.queue.get()]
            if job.status == "cancelled":
                continue
            _, fn = self.kinds[job.kind]
            job.status, job.started = "running", time.time()
            token = i18n.use(job.lang)
            try:
                job.result = fn(job, **job.params) or {}
                job.status = "cancelled" if job.cancel_requested else "done"
            except Exception:  # noqa: BLE001 - every failure is shown on the debug panel
                job.error = traceback.format_exc()
                job.status = "failed"
                _log.error("job %s failed\n%s", job.id, job.error)
            finally:
                i18n.reset(token)
            job.finished = time.time()
            self._save(job)

    def _save(self, job: Job) -> None:
        config.ensure_dirs()
        d = job.summary()
        d["lines"] = job.lines[-200:]                     # both languages, as logged
        (config.JOB_DIR / f"{job.id}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1, default=i18n.bi), encoding="utf-8")

    def _load_history(self) -> None:
        if not config.JOB_DIR.exists():
            return
        for p in sorted(config.JOB_DIR.glob("*.json"))[-200:]:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                d.pop("seconds", None)
                self.jobs[d["id"]] = Job(**d)
            except Exception:  # noqa: BLE001 - a broken history file is skipped
                continue
