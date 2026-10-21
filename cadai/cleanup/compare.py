"""The engineer's own cleanup of an archive drawing, as a picture next to a cleanup output.

When the cleaned drawing comes from the archive, the alignment labels say which raw items the
engineer kept in the final electrical drawing; drawing only those gives "what the engineer did"
in the same coordinates and frame as the cleanup output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cadai import config
from cadai.cad.render import render
from cadai.cad.scene import Scene
from cadai.cleanup.align import KEEP
from cadai.corpus import inventory, process


def archive_pair(src: Path) -> dict | None:
    """The aligned pair whose raw drawing is `src`, if any."""
    try:
        rel = str(Path(src).resolve().relative_to(config.PROJECTS.resolve()))
    except ValueError:
        return None
    for p in inventory.load()["projects"]:
        for f in p["files"]:
            if f["rel"] == rel:
                for pair in p["pairs"]:
                    info = process.label_info(p["id"], pair["final"]) if pair["raw"] == f["sha"] else None
                    if info and info.get("ok"):
                        return {"project": p["id"], "raw": pair["raw"], "final": pair["final"]}
    return None


def engineer_png(src: Path, target: Path) -> bool:
    pair = archive_pair(src)
    if not pair:
        return False
    scene = Scene.load(process.scene_path(pair["raw"]))
    with np.load(process.label_path(pair["project"], pair["final"])) as z:
        keep = z["labels"] == KEEP
    bb = scene.bboxes()[keep]
    bb = bb[np.isfinite(bb[:, 0])]
    bbox = (np.percentile(bb[:, 0], 0.5) - 2, np.percentile(bb[:, 1], 0.5) - 2,
            np.percentile(bb[:, 2], 99.5) + 2, np.percentile(bb[:, 3], 99.5) + 2)
    render(scene, target, np.where(keep, 1, -9), bbox=bbox, colours={1: (40, 40, 40)}, width=2400)
    return True
