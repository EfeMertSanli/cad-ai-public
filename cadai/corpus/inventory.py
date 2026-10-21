"""Project inventory: which drawings each project folder has, and which pair to learn from.

Folder layout (the office's archive): one folder per customer, inside it the architect's file(s)
and the engineer's final electrical drawing, named "<Owner>(<parcel>).dwg", e.g.
"Örnek SAHİBİ(100-1).dwg". Everything else (.bak, .dwl, Word, Excel) is ignored.

Roles:
- final: "<name>(<digits-digits>)" at the start of the file name;
- scratch: "Drawing1", "DrawingX..." (the engineer's working copies);
- raw: every other DWG. When there are several, the one naming "mimari"/"uygulama" wins, then the
  one sharing the final's parcel number, then the largest.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from cadai import config
from cadai.i18n import say

FINAL_RE = re.compile(r"^(?P<owner>.+?)\s*\(\s*(?P<parcel>\d+\s*-\s*\d+)\s*\)", re.IGNORECASE)
SCRATCH_RE = re.compile(r"^drawing", re.IGNORECASE)
RAW_HINT_RE = re.compile(r"mimari|mimar|uygulama|ruhsat|proje", re.IGNORECASE)


def fold(text: str) -> str:
    """ASCII, upper case: 'Gülşen ÇAĞLAR' -> 'GULSEN CAGLAR'."""
    text = text.replace("ı", "i").replace("İ", "I")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()


def project_id(folder: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "-", fold(folder)).strip("-").lower()
    return f"{slug[:40]}-{hashlib.sha1(folder.encode()).hexdigest()[:4]}"


def file_hash(path: Path, cache: dict) -> str:
    """sha256 of the content, reused while path, size and mtime are unchanged."""
    st = path.stat()
    key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    cache[key] = h.hexdigest()
    return cache[key]


def classify(name: str) -> str:
    stem = Path(name).stem
    if FINAL_RE.match(stem):
        return "final"
    if SCRATCH_RE.match(stem):
        return "scratch"
    return "raw"


def _parcel(text: str) -> str | None:
    m = re.search(r"(\d+)\s*[-_ ]\s*(\d+)", text)
    return f"{int(m.group(1))}-{int(m.group(2))}" if m else None


def choose_raw(raws: list[dict], final: dict) -> tuple[dict | None, dict]:
    """The raw drawing for a final one, and why ({"tr", "en"}, ADR-033)."""
    if not raws:
        return None, say("ham çizim yok", "no raw drawing").both()
    if len(raws) == 1:
        return raws[0], say("tek ham çizim", "single raw drawing").both()
    parcel = _parcel(final["name"])
    with_parcel = [r for r in raws if parcel and _parcel(r["name"]) == parcel]
    pool = with_parcel or raws
    hinted = [r for r in pool if RAW_HINT_RE.search(fold(r["name"]).lower())]
    pool = hinted or pool
    best = max(pool, key=lambda r: r["size"])
    why = (say("parsel no", "parcel number") if with_parcel else
           say("adında mimari/uygulama", "name says architectural/application") if hinted else say("en büyük dosya", "largest file"))
    return best, say(f"{len(raws)} ham çizim arasından seçildi ({why.tr})", f"chosen among {len(raws)} raw drawings ({why.en})").both()


def scan(root: Path = config.PROJECTS, log=lambda m: None) -> dict:
    cache_file = config.DATA / "hash_cache.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    projects = []
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    for i, folder in enumerate(folders, 1):
        files = []
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() not in (".dwg", ".dxf") or not f.is_file():
                continue
            files.append({
                "name": f.name,
                "rel": str(f.relative_to(root)),
                "size": f.stat().st_size,
                "role": classify(f.name),
                "sha": file_hash(f, cache),
            })
        finals = [f for f in files if f["role"] == "final"]
        raws = [f for f in files if f["role"] == "raw"]
        pairs = []
        for fin in finals:
            raw, why = choose_raw(raws, fin)
            pairs.append({"final": fin["sha"], "raw": raw["sha"] if raw else None, "why": why})
        projects.append({"id": project_id(folder.name), "folder": folder.name, "files": files, "pairs": pairs})
        if i % 20 == 0:
            log(say(f"{i}/{len(folders)} klasör tarandı", f"{i}/{len(folders)} folders scanned"))
    cache_file.write_text(json.dumps(cache), encoding="utf-8")
    inv = {
        "root": str(root),
        "projects": projects,
        "summary": {
            "projects": len(projects),
            "files": sum(len(p["files"]) for p in projects),
            "with_pair": sum(1 for p in projects if any(x["raw"] for x in p["pairs"])),
            "without_final": sum(1 for p in projects if not p["pairs"]),
        },
    }
    config.ensure_dirs()
    config.INVENTORY.write_text(json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")
    return inv


def load() -> dict:
    if not config.INVENTORY.exists():
        return {"projects": [], "summary": {}}
    return json.loads(config.INVENTORY.read_text(encoding="utf-8"))


def files_by_sha(inv: dict) -> dict[str, dict]:
    out = {}
    for p in inv["projects"]:
        for f in p["files"]:
            out.setdefault(f["sha"], {**f, "project": p["id"]})
    return out
