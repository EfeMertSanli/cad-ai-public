"""DWG -> DXF with the ODA File Converter.

ODA converts whole folders, so every file gets its own temporary input folder under an ASCII
name (Turkish characters in paths have broken ODA runs before). The original is only read.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from cadai import config
from cadai.i18n import say


class ConversionError(RuntimeError):
    pass


def oda_binary() -> Path:
    explicit = os.environ.get("ODA_PATH", "").strip().strip('"')
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    found = sorted(Path(r"C:\Program Files\ODA").glob("ODAFileConverter*/ODAFileConverter.exe"))
    if not found:
        raise ConversionError(str(say("ODA File Converter bulunamadı (ODA_PATH ayarlayın).",
                                      "ODA File Converter not found (set ODA_PATH).")))
    return found[-1]                                    # newest version folder


def to_dxf(src: Path, target: Path, timeout: int = 600) -> Path:
    """Convert `src` (DWG or DXF) into the DXF file `target`. Returns `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".dxf":
        shutil.copyfile(src, target)
        return target
    work = Path(tempfile.mkdtemp(prefix="oda_", dir=config.DATA))
    try:
        inp, out = work / "in", work / "out"
        inp.mkdir()
        out.mkdir()
        shutil.copyfile(src, inp / "drawing.dwg")
        cmd = [str(oda_binary()), str(inp), str(out), "ACAD2018", "DXF", "0", "1", "*.DWG"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        produced = out / "drawing.dxf"
        if not produced.exists() or produced.stat().st_size == 0:
            raise ConversionError(str(say(f"ODA DXF üretmedi (kod {proc.returncode}): {(proc.stderr or proc.stdout)[-400:]}",
                                          f"ODA produced no DXF (code {proc.returncode}): {(proc.stderr or proc.stdout)[-400:]}")))
        tmp = target.with_suffix(".part")
        shutil.move(str(produced), tmp)
        tmp.replace(target)
        return target
    finally:
        shutil.rmtree(work, ignore_errors=True)


def to_dwg(src: Path, target: Path, timeout: int = 600) -> Path:
    """Convert the DXF `src` into the DWG file `target` (AutoCAD 2018)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="oda_", dir=config.DATA))
    try:
        inp, out = work / "in", work / "out"
        inp.mkdir()
        out.mkdir()
        shutil.copyfile(src, inp / "drawing.dxf")
        cmd = [str(oda_binary()), str(inp), str(out), "ACAD2018", "DWG", "0", "1", "*.DXF"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        produced = out / "drawing.dwg"
        if not produced.exists() or produced.stat().st_size == 0:
            raise ConversionError(str(say(f"ODA DWG üretmedi (kod {proc.returncode}): {(proc.stderr or proc.stdout)[-400:]}",
                                          f"ODA produced no DWG (code {proc.returncode}): {(proc.stderr or proc.stdout)[-400:]}")))
        shutil.move(str(produced), target)
        return target
    finally:
        shutil.rmtree(work, ignore_errors=True)
