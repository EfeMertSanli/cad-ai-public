"""The cleanup toolbox: fixed programs that each find one kind of thing the engineer always deletes
(ADR-021). Claude does not judge items; it reads what the tools found (counts, pictures) and
decides per tool and per layer (agent.py).

Each tool returns the items it would delete inside the chosen floor plans. The decision is
everything in the plans minus the union of the enabled tools, with room names put back and
structural hatches kept as outlines. `attribution` says which tool removed an item first (for the
coloured picture and for measuring every tool against the archive).

    off_layers   layers whose type says "not in the base" (layers.py: names, engineer dictionary)
    annotations  texts (room names come back at the end), dimensions, leaders, tables
    furniture    block shapes engineers delete (furniture.delete_signatures)
    door_swings  door arcs (also exploded or drawn on the wall layer), leaves, frame pieces
    hatching     fills that are patterns, and exploded hatch lines
    dashed       dashed lines except stair flights
    model        items on layers of unknown type the cleanup model would delete
    ticks        short marks of openings sticking out of the walls

Protected, whatever the tools say: hatches that are columns / shear walls by shape
(elements.structural_fills, any layer); they stay as their outline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cadai.cad.scene import ATOMIC, TEXT, Scene
from cadai.i18n import say
from cadai.cleanup import elements, layers
from cadai.cleanup.finish import opening_ticks, room_names
from cadai.placement import furniture

DELETE, KEEP, OUTLINE = 0, 1, 2           # apply.py's codes

TOOLS = ["off_layers", "annotations", "furniture", "door_swings", "hatching", "dashed", "model", "ticks"]
LABELS = {"off_view": say("Kat planı dışı (kesit, görünüş, vaziyet…)", "Outside floor plans (sections, elevations, site…)"),
          "off_layers": say("Kapatılan katmanlar", "Switched-off layers"),
          "annotations": say("Yazı ve ölçü", "Text and dimensions"),
          "furniture": say("Mobilya ve vitrifiye", "Furniture and sanitary ware"),
          "door_swings": say("Kapı yayı ve kanadı", "Door swings and leaves"),
          "hatching": say("Tarama", "Hatching"), "dashed": say("Kesikli çizgi", "Dashed lines"),
          "model": say("Belirsiz katmanlarda model", "Model on ambiguous layers"),
          "ticks": say("Duvardan taşan çentikler", "Ticks sticking out of walls")}
COLOURS = {"off_layers": (120, 120, 255), "annotations": (0, 170, 170), "furniture": (230, 120, 0),
           "door_swings": (220, 0, 160), "hatching": (150, 90, 40), "dashed": (120, 180, 0),
           "model": (255, 60, 60), "ticks": (100, 0, 200)}
STRUCTURE = {"wall", "column", "shaft"}


@dataclass
class Result:
    decision: np.ndarray                       # DELETE / KEEP / OUTLINE per item
    found: dict[str, np.ndarray]               # tool -> items it would delete (in the plans)
    attribution: np.ndarray                    # first tool (index into TOOLS) that removed the item, -1 kept, -2 off view
    stats: dict = field(default_factory=dict)


def run(scene: Scene, in_plan: np.ndarray, p_delete: np.ndarray, kinds: list[str] | None = None,
        disabled: set[str] = frozenset(), layer_types: dict[str, str] | None = None) -> Result:
    """`kinds`: type per layer (layers.classify); `layer_types`: overrides by layer name (the
    engineer's or Claude's corrections for this drawing)."""
    kinds = list(kinds if kinds is not None else layers.classify(scene)[0])
    for i, name in enumerate(scene.meta["layers"]):
        if layer_types and name in layer_types:
            kinds[i] = layer_types[name]
    kind = layers.item_types(scene, kinds)
    types = np.array(scene.meta["types"])[scene.etype]
    is_text = np.isin(types, list(TEXT))
    plan = in_plan.copy()
    geometry = plan & ~is_text
    keep_layer = np.isin(kind, list(layers.KEEP_TYPES))
    unknown = kind == "unknown"
    doorish = np.isin(kind, ["door", "opening", "window"])

    # columns and shear walls drawn as fills (any layer): no tool may delete them, only their inside goes
    structure_fill = elements.structural_fills(scene, geometry)
    found: dict[str, np.ndarray] = {}
    found["off_layers"] = plan & ~keep_layer & ~unknown & ~is_text
    found["annotations"] = plan & (is_text | np.isin(types, list(ATOMIC)))
    found["furniture"] = furniture.detect(scene, geometry)
    found["door_swings"] = elements.door_swings(scene, geometry & (doorish | np.isin(kind, ["wall", "unknown"])),
                                                types, frame=doorish)
    pattern_fill = (types == "HATCH") & ~np.isin(kind, list(STRUCTURE))
    found["hatching"] = geometry & (pattern_fill | elements.hatch_lines(scene, geometry & np.isin(kind, ["wall", "column", "unknown"])))
    dashed = scene.dashed if scene.dashed is not None else np.zeros(scene.n, bool)
    found["dashed"] = geometry & dashed & (kind != "stair")
    found["model"] = geometry & unknown & (p_delete >= 0.5)

    for t in found:
        found[t] &= ~structure_fill
    enabled = [t for t in TOOLS if t != "ticks" and t not in disabled]
    removed = np.zeros(scene.n, dtype=bool)
    attribution = np.where(plan, -1, -2)
    for t in enabled:
        new = found[t] & ~removed
        attribution[new] = TOOLS.index(t)
        removed |= found[t]
    keep = plan & ~removed
    # sill / jamb marks are judged against the walls that stay
    found["ticks"] = opening_ticks(scene, np.where(keep, KEEP, DELETE).astype(np.int8), plan)
    if "ticks" not in disabled:
        attribution[found["ticks"] & keep] = TOOLS.index("ticks")
        keep &= ~found["ticks"]
    names = room_names(scene, np.where(keep, KEEP, DELETE).astype(np.int8), plan)
    keep[names] = True
    attribution[names] = -1
    decision = np.where(keep, KEEP, DELETE).astype(np.int8)
    decision[keep & (types == "HATCH") & (np.isin(kind, list(STRUCTURE)) | structure_fill)] = OUTLINE

    length = scene.lengths()
    stats = {"plan_items": int(plan.sum()), "kept": int(keep.sum()), "room_names": int(len(names)),
             "structural_fills": int(structure_fill.sum()),
             "tools": {t: {"items": int(found[t].sum()), "m": round(float(length[found[t]].sum()), 1),
                           "first": int((attribution == TOOLS.index(t)).sum()), "enabled": t not in disabled}
                       for t in TOOLS}}
    return Result(decision, found, attribution, stats)
