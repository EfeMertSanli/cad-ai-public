"""Logging: one JSON line per record in DATA/logs/app.jsonl, plus the latest records in memory
for the debug panel."""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from datetime import datetime

from cadai import config

RECENT: deque[dict] = deque(maxlen=2000)
_lock = threading.Lock()
_setup_done = False


class _Handler(logging.Handler):
    def __init__(self):
        super().__init__()
        config.ensure_dirs()
        self.path = config.LOG_DIR / "app.jsonl"

    def emit(self, record: logging.LogRecord) -> None:
        row = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if getattr(record, "en", None):                 # the English of a two-language message (ADR-033)
            row["msg_en"] = record.en
        if record.exc_info:
            row["exc"] = logging.Formatter().formatException(record.exc_info)
        with _lock:
            RECENT.append(row)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def setup() -> None:
    global _setup_done
    if _setup_done:
        return
    _setup_done = True
    root = logging.getLogger("cadai")
    root.setLevel(logging.INFO)
    root.addHandler(_Handler())
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(console)


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"cadai.{name}")
