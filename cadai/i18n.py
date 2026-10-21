"""Turkish and English for everything a person reads (ADR-033).

Every user-facing string is written in both languages where it is made: `say("Kural yok", "No rules")`.
A `Msg` is sent to the browser as {"tr": ..., "en": ...} and the page shows the language its EN/TR switch
selects; stored messages (job logs, check warnings, Claude's notes) keep both, so switching later works.
`str(msg)` and f-strings give the current language: the browser's for a request (cookie `cadai-lang`), the
language of whoever started a job while it runs (exception texts, the server log).

Text stored before ADR-033 is Turkish only; `bi()` adds its English from the translation memory
(DATA/translations.json, filled once by `python -m cadai.translate_stored`).
"""
from __future__ import annotations

import contextvars
import json
from dataclasses import dataclass

from cadai import config

LANGS = ("tr", "en")
COOKIE = "cadai-lang"
_current: contextvars.ContextVar[str] = contextvars.ContextVar("cadai_lang", default="tr")


def lang() -> str:
    return _current.get()


def use(code: str | None) -> contextvars.Token:
    return _current.set(code if code in LANGS else "tr")


def reset(token: contextvars.Token) -> None:
    _current.reset(token)


@dataclass(frozen=True)
class Msg:
    tr: str
    en: str

    def __str__(self) -> str:
        return self.en if lang() == "en" else self.tr

    def __format__(self, spec: str) -> str:
        return format(str(self), spec)

    def both(self) -> dict:
        return {"tr": self.tr, "en": self.en}


def say(tr: str, en: str) -> Msg:
    return Msg(tr, en)


def pick(pairs: dict) -> dict:
    """{key: Msg} -> {key: text in the current language}."""
    return {k: str(v) for k, v in pairs.items()}


_MEMORY: dict = {"mtime": None, "data": {}}
MEMORY_FILE = config.DATA / "translations.json"


def memory() -> dict[str, str]:
    """Turkish -> English for text stored before ADR-033."""
    try:
        mtime = MEMORY_FILE.stat().st_mtime
    except OSError:
        return {}
    if _MEMORY["mtime"] != mtime:
        _MEMORY["data"] = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        _MEMORY["mtime"] = mtime
    return _MEMORY["data"]


def bi(value):
    """A message as {"tr", "en"}: a Msg, an already bilingual dict, or older Turkish-only text."""
    if isinstance(value, Msg):
        return value.both()
    if isinstance(value, dict) and "tr" in value:
        return value
    if isinstance(value, str):
        return {"tr": value, "en": memory().get(value.strip(), value)}
    return value
