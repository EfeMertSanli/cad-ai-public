import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """Every test writes into its own data folder, never into D:\\cad-ai-data."""
    from cadai import config
    for name in ("DATA", "DXF_DIR", "SCENE_DIR", "LABEL_DIR", "FEATURE_DIR", "MODEL_DIR", "JOB_DIR", "LOG_DIR"):
        sub = "" if name == "DATA" else name.replace("_DIR", "").lower()
        monkeypatch.setattr(config, name, tmp_path / "data" / sub)
    monkeypatch.setattr(config, "INVENTORY", tmp_path / "data" / "inventory.json")
    from cadai.placement import symbols
    monkeypatch.setattr(symbols, "SYMBOL_DIR", tmp_path / "data" / "symbols")
    for name in ("CATALOG", "TYPES", "PROPOSALS"):
        monkeypatch.setattr(symbols, name, tmp_path / "data" / "symbols" / getattr(symbols, name).name)
    from cadai.placement import furniture
    monkeypatch.setattr(furniture, "FURNITURE_DIR", tmp_path / "data" / "furniture")
    for name in ("CATALOG", "TYPES", "PROPOSALS", "DELETE_SIGNATURES"):
        monkeypatch.setattr(furniture, name, tmp_path / "data" / "furniture" / getattr(furniture, name).name)
    from cadai.cleanup import agent
    monkeypatch.setattr(agent, "VIEW_RULEBOOK", tmp_path / "data" / "agent" / "view_rulebook.md")
    os.environ.pop("CADAI_DATA", None)
    yield


def make_scene(items, layers=None):
    """items: list of (layer index, [segments (x1, y1, x2, y2)], text). Metres."""
    from cadai.cad.scene import Scene
    n = len(items)
    seg_item, segs = [], []
    for i, (_, s, _) in enumerate(items):
        for seg in s:
            seg_item.append(i)
            segs.append(seg)
    layers = layers or [f"L{i}" for i in range(max(it[0] for it in items) + 1)]
    return Scene(
        meta={"layers": layers, "types": ["LINE", "TEXT"], "blocks": [], "unit": {"unit": "m", "scale": 1.0}},
        path=np.array([str(i) for i in range(n)]), top=np.arange(n, dtype=np.int32),
        layer=np.array([it[0] for it in items], dtype=np.int32),
        etype=np.array([1 if it[2] else 0 for it in items], dtype=np.int16),
        block=np.full(n, -1, dtype=np.int32), block_top=np.full(n, -1, dtype=np.int32),
        depth=np.zeros(n, dtype=np.int8), color=np.full(n, 256, dtype=np.int16),
        text=np.array([it[2] or "" for it in items]), text_h=np.zeros(n, dtype=np.float32),
        radius=np.zeros(n, dtype=np.float32), sweep=np.zeros(n, dtype=np.float32),
        closed=np.zeros(n, dtype=bool), filled=np.zeros(n, dtype=bool),
        seg_item=np.array(seg_item, dtype=np.int32), segs=np.array(segs, dtype=float).reshape(-1, 4),
    )
