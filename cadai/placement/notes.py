"""The engineer's free-text comments on lexicon entries (symbols, furniture).

A type picked from a list can't say everything ("this is a socket, but only above the counter",
"two of these are one wardrobe"). The comment is kept next to the confirmed types, with a log of
every change, so it can later be read by people and given to an LLM as context.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

MAX_CHARS = 2000


def load(folder: Path) -> dict[str, dict]:
    path = folder / "notes.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save(folder: Path, key: str, text: str) -> dict:
    """Store (or, with empty text, remove) the comment on `key`."""
    text = text.strip()[:MAX_CHARS]
    folder.mkdir(parents=True, exist_ok=True)
    notes = load(folder)
    entry = {"text": text, "at": datetime.now().isoformat(timespec="seconds")}
    if text:
        notes[key] = entry
    else:
        notes.pop(key, None)
    (folder / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(folder / "notes_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, **entry}, ensure_ascii=False) + "\n")
    return entry
