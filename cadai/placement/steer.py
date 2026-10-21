"""The engineer steers the placement in words (ADR-027).

Haiku turns an instruction ("salonlara en az 4 priz koy", "anahtarları kapının kilit tarafına koy")
into rules; code checks them and applies them on top of the placement model:
- count rules raise or cap the model's count for a class (in some room types);
- position rules put the points with a tag (door latch side, window, door, corner) first, the model's
  order kept among them, when the room has such points.

Measured on 60 test rooms (1 m F1 against the engineer): letting Sonnet edit the model's draft follows
the instruction too (14/14 living rooms with 4 sockets) but costs ~$1 per 60 rooms and loses accuracy
(socket 0.58 -> 0.50); rules applied by code keep the model's positions (socket 0.57) for ~$0.003.

The engineer sees the rules before placing and can change them (Haiku once carried "salonlara" from one
sentence over to the next); rules kept for every project live in the office rule book (ADR-028).
"""
from __future__ import annotations

import json
import re
import subprocess

import numpy as np

from cadai import config
from cadai.i18n import say
from cadai.placement import rooms as rooms_mod
from cadai.placement.candidates import FEATURES

CLAUDE = r"C:\Users\aztek\.local\bin\claude.exe"
TAGS = {"latch": ("d_latch", 0.5), "window": ("d_window", 0.3), "door": ("d_door", 0.6), "corner": ("d_corner", 0.3)}
ROOM_TYPES = sorted({t for _, t in rooms_mod.ROOM_TYPES} | {rooms_mod.OPEN_KITCHEN})
TAG_FIRST = 100.0              # added to the scores of tagged points: they come first, model order inside

PROMPT = """Bir elektrik mühendisinin cihaz yerleştirme talimatını kurallara çevir.
Cihaz sınıfları: {classes}.
Oda tipleri: {types}. "Salon" hem living hem living_kitchen (açık mutfaklı salon) demektir; "yatak odası"
bedroom; "banyo" bathroom; "mutfak" kitchen ve living_kitchen.
Konum etiketleri: latch (kapının kilit tarafı), window (pencere), door (kapı yanı), corner (köşe).
Her cümleyi ayrı oku: bir cümlede geçen oda tipi başka cümlenin kuralına taşınmaz; cümle oda söylemiyorsa
room_types boş listedir.
Talimat: "{text}"
Cevabın yalnızca bir JSON listesi olsun; her kural:
{{"class": "...", "room_types": [...] (boş liste = bütün odalar), "min": sayı ya da null, "max": sayı ya da null,
  "near": etiket ya da null}}"""


def parse(text: str, classes: list[str]) -> list[dict]:
    """Rules from the LLM's answer; anything outside the schema is dropped."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return check(raw, classes)


def check(raw, classes: list[str]) -> list[dict]:
    """Rules in the schema; anything else (unknown class, room type or tag, negative count) is dropped."""
    rules = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict) or r.get("class") not in classes:
            continue
        types = [t for t in r.get("room_types") or [] if t in ROOM_TYPES]
        lo, hi = r.get("min"), r.get("max")
        rule = {"class": r["class"], "room_types": types,
                "min": int(lo) if isinstance(lo, (int, float)) and lo >= 0 else None,
                "max": int(hi) if isinstance(hi, (int, float)) and hi >= 0 else None,
                "near": r.get("near") if r.get("near") in TAGS else None}
        if rule["min"] is not None or rule["max"] is not None or rule["near"]:
            rules.append(rule)
    return rules


def book_path():
    """The office rule book, applied to every placement."""
    return config.DATA / "placement_rules.json"


def load_book(classes: list[str]) -> list[dict]:
    try:
        return check(json.loads(book_path().read_text(encoding="utf-8")), classes)
    except (OSError, json.JSONDecodeError):
        return []


def save_book(rules, classes: list[str]) -> list[dict]:
    rules = check(rules, classes)
    book_path().parent.mkdir(parents=True, exist_ok=True)
    book_path().write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
    return rules


def translate(text: str, classes: list[str], model: str = "haiku") -> tuple[list[dict], float]:
    """(rules, cost in USD) for an instruction."""
    prompt = PROMPT.format(classes=", ".join(classes), types=", ".join(ROOM_TYPES), text=text.replace('"', "'"))
    proc = subprocess.run([CLAUDE, "-p", "--model", model, "--output-format", "json"], input=prompt,
                          capture_output=True, text=True, encoding="utf-8", timeout=300)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(str(say(f"Claude cevap vermedi: {proc.stderr[-300:]}", f"Claude did not answer: {proc.stderr[-300:]}"))) from exc
    return parse(out.get("result") or "", classes), float(out.get("total_cost_usd") or 0.0)


def apply(rules: list[dict], cls: str, room_type: str | None, n: int, X: np.ndarray,
          scores: np.ndarray) -> tuple[int, np.ndarray]:
    """(count, scores) of one class in one room after the rules. X, scores: the class's candidates."""
    for r in rules:
        if r["class"] != cls or (r["room_types"] and room_type not in r["room_types"]):
            continue
        if r["min"] is not None:
            n = max(n, r["min"])
        if r["max"] is not None:
            n = min(n, r["max"])
        if r["near"]:
            f, limit = TAGS[r["near"]]
            ok = X[:, FEATURES.index(f)] < limit
            if ok.any():
                scores = np.where(ok, scores + TAG_FIRST, scores)
    return n, scores
