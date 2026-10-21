"""Remove personal details from a drawing before it leaves the office (ADR-032).

Names may stay; everything else personal goes: addresses, streets, districts, plot / parcel / sheet numbers,
phone numbers, e-mail and web addresses, ID, tax and registration numbers, embedded files and pictures.
These live in the title block, the site plan and the project tables, all outside the floor plans, so:

- outside the floor plans every text goes (labels and their values are separate texts, so matching the
  label alone would leave the value);
- inside the floor plans texts stay (room names, levels, areas) unless they look personal (PRIVATE);
- embedded objects (OLE, raster images, underlays) go everywhere;
- `leaks()` then reads every string in the written DXF, including extended data, and reports anything that
  still looks personal. The demo is only built when it reports nothing.
"""
from __future__ import annotations

import re
from pathlib import Path

import ezdxf

from cadai.corpus.inventory import fold

# on fold()ed text (ASCII upper case)
PRIVATE = re.compile(
    r"[\w.-]+@[\w-]+\.|WWW\.|HTTPS?:|\.COM\b|\.NET\b|\.ORG\b"
    r"|\bTEL\b|\bGSM\b|\bFAKS?\b|\bFAX\b|\bCEP\b"
    r"|(?<![\d-])0\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b"         # phone numbers: 0 + area code
    r"|ADRES|\bMAH\b|\bMAH\.|MAHALLE|\bCAD\b|\bCAD\.|CADDE|\bSOK\b|\bSOK\.|SOKA[GK]|\bSK\.|BULVAR|\bBLV\b"
    r"|\bNO\s?[:.]\s?\d|\bADA\b|PARSEL|PAFTA|\bKOY\b|MEVKI|ILCE|PLAZA|BELEDIYE"
    r"|SICIL|TES\.\s?NO|KIMLIK|\bT\.?C\.?\s?NO|VERGI|\bV\.?D\.")
SOFTWARE = re.compile(r"AUTODESK|EZDXF")      # vendor stamps the CAD programs write themselves
TEXTS = {"TEXT", "MTEXT", "MULTILEADER", "ACAD_TABLE", "ATTDEF"}
EMBEDDED = {"OLE2FRAME", "IMAGE", "PDFUNDERLAY", "DWFUNDERLAY", "DGNUNDERLAY"}
HEADER = ("$PROJECTNAME", "$HYPERLINKBASE", "$LASTSAVEDBY")


def is_private(text: str) -> bool:
    return bool(PRIVATE.search(fold(text or "")))


def _text(e) -> str:
    t = e.dxftype()
    if t in ("TEXT", "ATTDEF", "ATTRIB"):
        return e.dxf.text
    if t == "MTEXT":
        return e.text
    if t == "MULTILEADER":
        try:
            return e.context.mtext.default_content
        except AttributeError:
            return ""
    return ""


def _where(e):
    for attr in ("insert", "text_midpoint", "defpoint"):
        if e.dxf.hasattr(attr):
            p = e.dxf.get(attr)
            return float(p[0]), float(p[1])
    if e.dxftype() == "MULTILEADER":
        try:
            p = e.context.mtext.insert
            return float(p[0]), float(p[1])
        except AttributeError:
            pass
    return None


def _inside(p, boxes) -> bool:
    return p is not None and any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for x0, y0, x1, y1 in boxes)


def _blocks_used(doc, names: set[str]) -> set[str]:
    """`names` and every block nested in them."""
    seen, todo = set(), list(names)
    while todo:
        n = todo.pop()
        if n in seen or n not in doc.blocks:
            continue
        seen.add(n)
        todo += [e.dxf.name for e in doc.blocks[n] if e.dxftype() == "INSERT"]
    return seen


def anonymise(src: Path, target: Path, plans: list[tuple[float, float, float, float]], margin: float) -> dict:
    """Write `src` without personal details to `target`. `plans` are the floor-plan boxes in drawing units,
    widened by `margin` (drawing units). Returns counts of what was removed."""
    doc = ezdxf.readfile(src)
    boxes = [(x0 - margin, y0 - margin, x1 + margin, y1 + margin) for x0, y0, x1, y1 in plans]
    report = {"outside": 0, "private": 0, "attribs": 0, "embedded": 0, "block_texts": 0, "paper": 0}
    msp = doc.modelspace()
    inside_blocks, outside_blocks = set(), set()
    for e in list(msp):
        t = e.dxftype()
        if t in EMBEDDED:
            msp.delete_entity(e)
            report["embedded"] += 1
        elif t in TEXTS:
            if not _inside(_where(e), boxes):
                msp.delete_entity(e)
                report["outside"] += 1
            elif is_private(_text(e)) or t == "ACAD_TABLE":
                msp.delete_entity(e)
                report["private"] += 1
        elif t == "DIMENSION":
            override = e.dxf.get("text", "")
            if override not in ("", "<>") and is_private(override):
                e.dxf.text = "<>"
                report["private"] += 1
        elif t == "INSERT":
            inside = _inside(_where(e), boxes)
            (inside_blocks if inside else outside_blocks).add(e.dxf.name)
            for a in list(e.attribs):
                if not inside or is_private(a.dxf.text):
                    e.attribs.remove(a)
                    a.destroy()
                    report["attribs"] += 1
    # blocks drawn only outside the plans (title blocks, stamps): no text survives; elsewhere only private text goes
    only_out = _blocks_used(doc, outside_blocks) - _blocks_used(doc, inside_blocks)
    for block in doc.blocks:
        if block.name.lower().startswith(("*model_space", "*paper_space")):
            continue
        for e in list(block):
            t = e.dxftype()
            if t in EMBEDDED or (t in TEXTS and (block.name in only_out or is_private(_text(e)))):
                block.delete_entity(e)
                report["block_texts"] += 1
    for layout in doc.layouts:
        if layout.name != "Model":
            for e in list(layout):
                if e.dxftype() in TEXTS | EMBEDDED | {"INSERT"}:
                    layout.delete_entity(e)
                    report["paper"] += 1
    for obj in list(doc.objects.query("IMAGEDEF")):
        doc.objects.delete_entity(obj)
    for var in HEADER:
        if var in doc.header:
            doc.header[var] = ""
    target.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(target)
    return report


STRING_CODES = {1, 3, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 1000}    # text values, not layer or block names


def leaks(dxf: Path) -> list[str]:
    """Every string value in the DXF file (entities, tables, objects, extended data) that looks personal."""
    found = []
    with open(dxf, encoding="utf-8", errors="replace") as f:
        while True:
            code, value = f.readline(), f.readline()
            if not value:
                break
            try:
                c = int(code)
            except ValueError:
                continue
            if c in STRING_CODES and is_private(value.strip()) and not SOFTWARE.search(fold(value)):
                found.append(value.strip())
    return found
