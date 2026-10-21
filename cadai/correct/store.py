"""Append-only log of the engineer's corrections, one JSON line per action, in the output folder.

The log is the training and evaluation data (ADR-029): each line keeps what the system proposed
(`before`) and what the engineer made of it (`after`), with the reasons picked and a free note.
Reasons can be added after the action (the engineer drags a device, then says why): a "reason"
line points to the action by its `seq`.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

LOG = "edits.jsonl"


def read(out_dir: Path) -> list[dict]:
    f = out_dir / LOG
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(out_dir: Path, entry: dict) -> dict:
    entry = {"seq": len(read(out_dir)), "t": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **entry}
    with (out_dir / LOG).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def reasons(entries: list[dict]) -> dict[int, dict]:
    """seq -> the latest {reasons, note} given for that action."""
    out = {}
    for e in entries:
        if e.get("action") == "reason":
            out[e["target"]] = {"reasons": e.get("reasons", []), "note": e.get("note", "")}
    return out
