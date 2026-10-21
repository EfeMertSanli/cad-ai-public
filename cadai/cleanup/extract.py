"""Choose how a drawing is cleaned and decide every item (ADR-020).

The engineer's way first: by layers. When the architect's layers say what they hold, the base
drawing is the kept layers minus a few parts inside them (`elements.select`). Some architects draw
everything on one or two layers, or name layers by pen number only; then layers can't separate
anything and the drawing is cleaned item by item (cleanup model + finishing rules, `finish.py`).

    layers  : at least half of the plan geometry is on layers of a known type, and walls are among them
    items   : otherwise

Decisions use apply.py's codes: DELETE, KEEP, OUTLINE (a hatch on a wall / column layer is the
element itself, and so is a shaft fill: only its boundary stays, ADR-018).
"""
from __future__ import annotations

import numpy as np

from cadai.cad.scene import Scene
from cadai.cleanup import elements, finish, layers

DELETE, KEEP, OUTLINE = 0, 1, 2
MIN_KNOWN_SHARE = 0.5


def layer_table(scene: Scene, kinds: list[str], sources: list[str], in_plan: np.ndarray,
                decision: np.ndarray | None = None) -> list[dict]:
    """One row per layer with geometry in the plans: what it is, how much, how much stayed."""
    length = scene.lengths()
    n = len(scene.meta["layers"])
    plan_m = np.bincount(scene.layer, weights=np.where(in_plan, length, 0), minlength=n)
    kept_m = (np.bincount(scene.layer, weights=np.where(in_plan & (decision != DELETE), length, 0), minlength=n)
              if decision is not None else np.zeros(n))
    items = np.bincount(scene.layer, weights=in_plan, minlength=n)
    rows = []
    for i in np.argsort(-plan_m):
        if items[i] == 0:
            continue
        rows.append({"index": int(i), "layer": scene.meta["layers"][i], "base": layers.base_name(scene.meta["layers"][i]),
                     "type": kinds[i], "source": sources[i], "items": int(items[i]),
                     "plan_m": round(float(plan_m[i]), 1), "kept_m": round(float(kept_m[i]), 1)})
    return rows


def choose_mode(scene: Scene, kinds: list[str], in_plan: np.ndarray) -> tuple[str, dict]:
    kind = layers.item_types(scene, kinds)
    length = np.where(in_plan, scene.lengths(), 0)
    total = length.sum()
    known = length[kind != "unknown"].sum() / total if total else 0.0
    walls = length[np.isin(kind, ["wall", "column"])].sum() / total if total else 0.0
    mode = "layers" if known >= MIN_KNOWN_SHARE and walls > 0 else "items"
    return mode, {"known_share": round(float(known), 3), "wall_share": round(float(walls), 3)}


def decide(scene: Scene, kinds: list[str], in_plan: np.ndarray, model_applied: np.ndarray,
           mode: str) -> tuple[np.ndarray, dict]:
    """Decision per item for the chosen mode. `model_applied`: the cleanup model's best guess."""
    if mode == "items":
        return finish.finish(scene, model_applied, in_plan)
    keep, stats = elements.select(scene, kinds, in_plan, model_applied == KEEP)
    out = np.where(keep, KEEP, DELETE).astype(np.int8)
    hatch = keep & (np.array(scene.meta["types"])[scene.etype] == "HATCH")
    structural = hatch & np.isin(layers.item_types(scene, kinds), ["wall", "column", "shaft"])
    out[structural] = OUTLINE
    out[hatch & ~structural] = DELETE                 # fills on window / stair / unknown layers are patterns
    stats.update(hatches_outlined=int(structural.sum()), hatches_deleted=int((hatch & ~structural).sum()))
    return out, stats
