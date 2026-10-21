"""Paths and settings. Everything heavy lives under DATA (default D:\\cad-ai-data), never in the code folder."""
from __future__ import annotations

import os
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("CADAI_DATA", r"D:\cad-ai-data"))
PROJECTS = Path(os.environ.get("CADAI_PROJECTS", Path.home() / "Desktop" / "proje" / "Proje"))
PORT = int(os.environ.get("CADAI_PORT", "8800"))
# 127.0.0.1 = this computer only. 0.0.0.0 = reachable from the network (Hamachi, Tailscale, LAN):
# then a password is required (CADAI_PASSWORD), see start_remote.bat.
HOST = os.environ.get("CADAI_HOST", "127.0.0.1")
PASSWORD = os.environ.get("CADAI_PASSWORD", "")

# Sub-folders of DATA. Created on first use.
DXF_DIR = DATA / "dxf"              # DWG converted by ODA, one file per content hash
SCENE_DIR = DATA / "scenes"         # compact exploded geometry (npz), one per content hash
LABEL_DIR = DATA / "labels"         # cleanup labels from raw -> final alignment, one per project
FEATURE_DIR = DATA / "features"     # cleanup features per raw scene (cache)
MODEL_DIR = DATA / "models"         # trained models with their reports
JOB_DIR = DATA / "jobs"             # finished job records for the debug panel
LOG_DIR = DATA / "logs"
INVENTORY = DATA / "inventory.json"

# Bump when the scene format changes, so old cached scenes are rebuilt.
SCENE_VERSION = 1


def ensure_dirs() -> None:
    for d in (DATA, DXF_DIR, SCENE_DIR, LABEL_DIR, FEATURE_DIR, MODEL_DIR, JOB_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
