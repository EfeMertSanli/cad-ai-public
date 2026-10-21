"""Drawing unit from evidence. $INSUNITS is wrong in most real files, so it only gets a small vote.

Evidence per candidate unit (mm, cm, dm, m):
- a 90-degree arc with a door-sized radius (0.6-1.2 m) is a door swing: strong vote;
- text heights of 8-60 cm in the plan are typical: weak vote;
- $INSUNITS: one weak vote.
"""
from __future__ import annotations

import math
from collections import Counter

UNITS = {"mm": 0.001, "cm": 0.01, "dm": 0.1, "m": 1.0}
INSUNITS = {4: "mm", 5: "cm", 14: "dm", 6: "m"}


def infer_unit(arcs: list[tuple[float, float]], text_heights: list[float], insunits: int) -> dict:
    """arcs: (radius, sweep in degrees). Returns {unit, scale, confidence, votes}."""
    votes: Counter[str] = Counter()
    for radius, sweep in arcs:
        if 80 <= sweep <= 100:
            for name, s in UNITS.items():
                if 0.6 <= radius * s <= 1.2:
                    votes[name] += 3
    for h in text_heights:
        for name, s in UNITS.items():
            if 0.08 <= h * s <= 0.6:
                votes[name] += 1
    header = INSUNITS.get(insunits)
    if header:
        votes[header] += 5
    if not votes:
        return {"unit": "cm", "scale": UNITS["cm"], "confidence": 0.0, "votes": {}}
    best, score = votes.most_common(1)[0]
    total = sum(votes.values())
    conf = round(score / total * min(1.0, math.log10(score + 1) / 2), 2)
    return {"unit": best, "scale": UNITS[best], "confidence": conf, "votes": dict(votes)}
