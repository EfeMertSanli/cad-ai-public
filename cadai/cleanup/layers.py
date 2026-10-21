"""What each architect layer holds, as an element type of the engineer's base drawing (ADR-019, ADR-020).

The engineer never edits the architect's drawing: they switch layers off (and line weights), and
what stays is the architect's own geometry. Measured on the archive (Cevriye BERKOĞLU and two other
projects): once a layer is seen as an element type, layers are mostly
all-or-nothing (walls, columns, windows, stairs stay; hatches, furniture, dimensions, axes go); the
rest is a few parts inside a kept layer (column hatching, door swings, labels) that `elements.py`
removes by geometry.

A layer's type comes from, in order: the engineer's confirmed dictionary (by the full layer name,
pen number included: ArchiCAD puts walls on "0_Pen_No__91" and stairs on "0_Pen_No__11"; shared by
every project, written from the Temizle page), the name rules below (Turkish and English,
ArchiCAD's "<layer>_Pen_No__<n>" suffix stripped), else `unknown`: names that say nothing ("0",
"sketch", "Standart") or mean different things per office ("Döşeme": slab edge in one, floor
tiles in another; "Nokta"). Unknown layers are judged item by item by the cleanup model.
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np

from cadai import config
from cadai.cad.scene import Scene
from cadai.corpus.inventory import fold
from cadai.i18n import say

# Order matters: the first type whose word is in the name wins ("KAPI_PENCERE" is an opening,
# "DIS DUVAR" a wall, "KOLON TARAMA" a hatch).
TYPES: list[tuple[str, tuple[str, ...]]] = [
    # AIA "A-FLOR-HRAL": handrails and balcony parapets; "FLOR" alone (floor patterns) is dropped below.
    # Ali Kemal ARSLAN: the balconies' outer walls are on it (ADR-021).
    ("railing", ("HRAL",)),
    ("drop", ("TARAMA", "HATCH", "DESEN", "PATT", "FILL", "FLOR", "TEFRIS", "FURN", "MOBILYA",
              "KUTUPHANE", "OBJE", "VITRIFIYE", "SIHHI", "SANR", "FIXT", "EQPM", "OLCU", "DIM", "ANNO", "AKS",
              "GRID", "KOT", "LEVEL", "YAZI", "METIN", "TEXT", "CATI", "ROOF", "KIRIS", "BEAM", "IMAR",
              "VAZIYET", "KADASTRAL", "PARSEL", "KESIT", "CEPHE", "GORUNUS", "PEYZAJ", "LANDSCAPE", "AGAC",
              "TREE", "ARAC", "CAR", "MARKER", "CEMBER", "DETL", "DETAY", "ROTA", "DEFPOINTS")),
    ("opening", ("KAPI_PENCERE", "KAPI PENCERE", "KAPI-PENCERE", "DOGRAMA")),
    ("door", ("KAPI", "DOOR")),
    ("window", ("PENCERE", "WINDOW", "GLAZ", "CAM")),
    ("stair", ("MERDIVEN", "STAIR", "BASAMAK", "RAMPA", "STRS")),          # STRS: AIA "S-STRS"
    ("railing", ("KORKULUK", "RAIL", "PARAPET")),
    ("shaft", ("SAFT", "SHAFT", "BACA", "HAVALANDIRMA", "ASANSOR", "LIFT", "ELEV")),
    ("column", ("KOLON", "COLUMN", "COLS")),
    ("wall", ("DUVAR", "WALL", "PERDE", "BETON", "PANEL", "TUGLA", "BRIKET", "GAZBETON", "SIVA")),
    ("room_name", ("MAHAL", "ROOM", "AREA-IDEN")),
]
KEEP_TYPES = {"opening", "door", "window", "stair", "railing", "shaft", "column", "wall"}
LABELS = {"wall": say("Duvar", "Wall"), "column": say("Kolon / perde", "Column / shear wall"),
          "window": say("Pencere", "Window"), "door": say("Kapı", "Door"),
          "opening": say("Kapı + pencere", "Door + window"), "stair": say("Merdiven", "Stair"),
          "railing": say("Korkuluk", "Railing"), "shaft": say("Şaft / asansör", "Shaft / lift"),
          "room_name": say("Oda adı (yalnız adlar kalır)", "Room name (only the names stay)"),
          "drop": say("Kapat", "Switch off"), "unknown": say("Belirsiz (model karar verir)", "Ambiguous (the model decides)")}
PEN = re.compile(r"_PEN_NO__\d+$")
DICTIONARY = "types.json"


def base_name(name: str) -> str:
    """'Dış Duvar_Pen_No__21' -> 'DIS DUVAR'."""
    return " ".join(PEN.sub("", fold(name)).split())


def type_of(name: str) -> str:
    n = base_name(name)
    for kind, words in TYPES:
        if any(w in n for w in words):
            return kind
    return "unknown"


def _dir():
    return config.DATA / "layers"


def confirmed() -> dict[str, str]:
    """The engineer's layer dictionary: {base name: type}."""
    path = _dir() / DICTIONARY
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def key(name: str) -> str:
    """Dictionary key: the folded full name ('Dış Duvar_Pen_No__21' -> 'DIS DUVAR_PEN_NO__21')."""
    return " ".join(fold(name).split())


def confirm(types: dict[str, str]) -> dict[str, str]:
    """Store the engineer's layer types, by full layer name."""
    bad = [t for t in types.values() if t not in LABELS]
    if bad:
        raise ValueError(bad)
    folder = _dir()
    folder.mkdir(parents=True, exist_ok=True)
    known = confirmed()
    with open(folder / "types_log.jsonl", "a", encoding="utf-8") as log:
        for name, kind in types.items():
            known[key(name)] = kind
            log.write(json.dumps({"layer": name, "key": key(name), "type": kind,
                                  "at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False) + "\n")
    (folder / DICTIONARY).write_text(json.dumps(known, ensure_ascii=False, indent=1), encoding="utf-8")
    return known


def classify(scene: Scene) -> tuple[list[str], list[str]]:
    """Element type per layer index of `scene.meta['layers']`, and where it came from
    ('engineer' dictionary or 'name' rules)."""
    known = confirmed()
    kinds, sources = [], []
    for name in scene.meta["layers"]:
        k = key(name)
        if k in known:
            kinds.append(known[k]); sources.append("engineer")
        else:
            kinds.append(type_of(name)); sources.append("name")
    return kinds, sources


def item_types(scene: Scene, layer_types: list[str] | dict[int, str]) -> np.ndarray:
    lt = layer_types if isinstance(layer_types, list) else [layer_types.get(i, "unknown")
                                                             for i in range(len(scene.meta["layers"]))]
    return np.array(lt, dtype=object)[scene.layer]
