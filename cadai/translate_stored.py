"""Give text stored before ADR-033 (Turkish only) its English, once, into the translation memory.

What it reads: Claude's region and review notes (outputs/*/agent.json), check warnings (outputs/*/checks.json),
the symbol and furniture proposals, why a raw drawing was paired (inventory), the alignment logs and errors of
the labelling step, and the titles of saved jobs. Each distinct text is translated once by Claude Haiku in
batches; the result goes to DATA/translations.json, which `i18n.bi()` reads. Job log lines saved before
ADR-033 are left as they are (thousands of lines with numbers, only seen on the Debug page).

    python -m cadai.translate_stored            (--dry-run: count only)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

from cadai import config, i18n
from cadai.placement import furniture, furniture_shapes, symbols
from cadai.placement.steer import CLAUDE

TURKISH = re.compile(r"[çğıöşüÇĞİÖŞÜ]|\b(ve|bir|için|yok|oda|kat|plan[ıi]|kaldı|silindi|görünüş|kesit|yazı)\b", re.I)
BATCH = 60
PROMPT = """Translate each Turkish text below to English. They come from software that cleans architectural CAD drawings
and places electrical devices (kat planı = floor plan, kesit = section, görünüş = elevation/view, vaziyet = site plan,
antet = title block, pafta = drawing sheet, tarama = hatching, altlık = base plan, mahal = room, kot = level).
Keep numbers, units, layer names, block names and quoted drawing text exactly as they are. Keep it short and plain.
Answer ONLY with a JSON array of the translations, in the same order, same length.

{items}"""


# messages the code wrote from fixed templates (now written in both languages where they are made)
TEMPLATES = [
    (r"görünüş kabul: dönüş (\S+)°, ölçek (\S+), öteleme \((.+)\) m, eşleşen (\d+) m çizgi",
     r"view accepted: rotation \1°, scale \2, shift (\3) m, \4 m of lines matched"),
    (r"(\d+) aday dönüşüm, en güçlü oy (\d+), ölçek (\S+)", r"\1 candidate transforms, strongest vote \2, scale \3"),
    (r"en güçlü oy (\d+) < (\d+): güvenilir hizalama yok", r"strongest vote \1 < \2: no reliable alignment"),
    (r"yanındaki yazılar: (.*)", r"nearby text: \1"),
    (r"Oda (\S+) m² okundu, mimar (\S+) m² yazmış \(%([+-]?\d+)\)\.", r"Room read as \1 m², the architect wrote \2 m² (\3%)."),
    (r"(\S+) m² yazan alan hiçbir odaya düşmüyor \(oda açık ya da silinmiş\)\.",
     r"The \1 m² area label falls in no room (room open or deleted)."),
    (r"Mimarın yazdığı alanlar duvar içinden ortalama %([+-]?\d+) farklı \(alanları duvar payıyla ya da başka bir tanımla "
     r"yazmış olabilir\); oda kontrolü buna göre yapıldı\.",
     r"The architect's areas differ from the floor by \1% on average (they may include the walls or use another "
     r"definition); the room check allows for this."),
    (r"Bir odada (\d+) alan yazısı var \((.*) m²\): aralarındaki duvar ya da kapı eksik, odalar birleşmiş\.",
     r"One room has \1 area labels (\2 m²): the wall or door between them is missing, the rooms merged."),
    (r"Temizlikten sonra çizimde hiçbir şey kalmadı\.", r"Nothing was left in the drawing after cleanup."),
    (r"(\d+) ham çizim arasından seçildi \(parsel no\)", r"chosen among \1 raw drawings (parcel number)"),
    (r"(\d+) ham çizim arasından seçildi \(adında mimari/uygulama\)", r"chosen among \1 raw drawings (name says architectural/application)"),
    (r"(\d+) ham çizim arasından seçildi \(en büyük dosya\)", r"chosen among \1 raw drawings (largest file)"),
    (r"tek ham çizim", r"single raw drawing"), (r"ham çizim yok", r"no raw drawing"),
    (r"hizalama bulunamadı", r"no alignment found"), (r"hizalanamadı", r"not aligned"),
    (r"yüksek", r"high"), (r"orta", r"medium"), (r"düşük", r"low"), (r"istatistik model", r"statistical model"),
]


def from_template(text: str) -> str | None:
    for pattern, english in TEMPLATES:
        m = re.fullmatch(pattern, text, re.S)
        if m:
            return m.expand(english)
    return None


def _texts(value) -> list[str]:
    return [value] if isinstance(value, str) and value.strip() and TURKISH.search(value) else []


def collect() -> list[str]:
    found: list[str] = []
    for f in (config.DATA / "outputs").glob("*/agent.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        found += _texts(rec.get("views"))
        for call in rec.get("calls", []):
            for r in call.get("regions") or []:
                found += _texts(r.get("what"))
            found += _texts((call.get("fixes") or {}).get("notes"))
    for f in (config.DATA / "outputs").glob("*/checks.json"):
        for issue in json.loads(f.read_text(encoding="utf-8")).get("issues", []):
            found += _texts(issue.get("message"))
    for mod in (symbols, furniture, furniture_shapes):
        for p in mod.load_proposals().values():
            found += _texts(p.get("why")) + _texts(p.get("confidence"))
    if config.INVENTORY.exists():
        for project in json.loads(config.INVENTORY.read_text(encoding="utf-8"))["projects"]:
            for pair in project["pairs"]:
                found += _texts(pair.get("why"))
    for f in config.LABEL_DIR.glob("*.json"):
        info = json.loads(f.read_text(encoding="utf-8"))
        found += _texts(info.get("error"))
        for line in info.get("log") or []:
            found += _texts(line)
    for f in config.JOB_DIR.glob("*.json"):
        found += _texts(json.loads(f.read_text(encoding="utf-8")).get("title"))
    known = i18n.memory()
    return sorted({t.strip() for t in found} - set(known))


def translate(texts: list[str]) -> tuple[list[str], float]:
    prompt = PROMPT.format(items=json.dumps(texts, ensure_ascii=False, indent=0))
    proc = subprocess.run([CLAUDE, "-p", "--model", "haiku", "--output-format", "json"], input=prompt,
                          capture_output=True, text=True, encoding="utf-8", timeout=600)
    out = json.loads(proc.stdout)
    text = out.get("result") or ""
    m = re.search(r"\[.*\]", text, re.S)
    answer = json.loads(m.group(0)) if m else []
    if len(answer) != len(texts):
        raise RuntimeError(f"expected {len(texts)} translations, got {len(answer)}")
    return [str(a) for a in answer], float(out.get("total_cost_usd") or 0.0)


def main(dry_run: bool = False) -> None:
    texts = collect()
    fixed = {t: e for t in texts if (e := from_template(t)) is not None}
    texts = [t for t in texts if t not in fixed]
    print(f"{len(fixed)} texts from templates; {len(texts)} for Haiku, {sum(len(t) for t in texts)} characters")
    if dry_run:
        return
    memory, cost = {**i18n.memory(), **fixed}, 0.0
    i18n.MEMORY_FILE.write_text(json.dumps(memory, ensure_ascii=False, indent=0), encoding="utf-8")
    for start in range(0, len(texts), BATCH):
        part = texts[start:start + BATCH]
        try:
            english, c = translate(part)
        except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            print(f"batch {start // BATCH + 1}: skipped ({exc})")
            continue
        cost += c
        memory.update(zip(part, english))
        i18n.MEMORY_FILE.write_text(json.dumps(memory, ensure_ascii=False, indent=0), encoding="utf-8")
        print(f"batch {start // BATCH + 1}/{(len(texts) + BATCH - 1) // BATCH}: {len(part)} texts, ${cost:.3f} so far")
    print(f"done: {len(memory)} texts in the memory, ${cost:.3f}")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
