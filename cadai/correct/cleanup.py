"""The engineer corrects a cleanup (Düzelt page, Temizlik tab, ADR-029).

He clicks an entity (or drags a box) and sees why the system kept or deleted it: the toolbox tool
that removed it (decision.npz `why`, outputs cleaned since ADR-029), "outside the floor plans", or
its layer type for older outputs and the layer / item modes. He marks it "kalmalı" (keep) or
"silinmeli" (delete) with reasons and a note. Overrides are per scene item (cleanup_corrected.json);
a click selects the whole top-level entity, an alt-click only the leaf.

The corrected drawing is written from the source DWG with the same decision plus the overrides
(clean_corrected.dxf / .dwg). Outputs cleaned before ADR-029 lack the outline decision of hatches;
there a kept hatch stays whole.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cadai.cad.scene import Scene
from cadai.cleanup import layers, tools
from cadai.correct import store
from cadai.i18n import Msg, say

STATE = "cleanup_corrected.json"
MAX_SELECT = 20000
MAX_HIGHLIGHT_SEGS = 4000
# the engineer's reasons; the Turkish text is what the log stores
REASONS = [say("duvar", "wall"), say("kolon / perde", "column / shear wall"), say("pencere", "window"),
           say("kapı boşluğu", "door opening"), say("merdiven", "stair"), say("asansör / şaft", "lift / shaft"),
           say("korkuluk", "railing"), say("oda adı", "room name"), say("kot / alan yazısı", "level / area label"),
           say("mobilya", "furniture"), say("vitrifiye", "sanitary ware"), say("tarama", "hatching"),
           say("ölçü / yazı", "dimension / text"), say("kat planı dışı", "outside the floor plans"),
           say("mimari detay", "architectural detail"), say("ofis alışkanlığı", "office habit")]
DELETED, KEPT, SAY_KEEP, SAY_DELETE = 0, 1, 2, 3          # render codes


@dataclass
class Loaded:
    scene: Scene
    kept: np.ndarray
    applied: np.ndarray | None          # apply.py codes (KEEP / DELETE / OUTLINE); None before ADR-029
    why: np.ndarray | None              # tools.Result.attribution; None before ADR-029 or outside toolbox modes
    kind: np.ndarray                    # layer type per item (dictionary + Claude's fixes for this drawing)


def load(out_dir: Path) -> Loaded:
    from cadai.placement.place import load_cleaned
    scene, kept = load_cleaned(out_dir)
    applied = why = None
    with np.load(out_dir / "decision.npz") as z:
        if "applied" in z.files:
            applied = z["applied"]
        if "why" in z.files:
            why = z["why"]
    kinds, _ = layers.classify(scene)
    agent_file = out_dir / "agent.json"
    if agent_file.exists():
        fixes = (json.loads(agent_file.read_text(encoding="utf-8")).get("fixes") or {}).get("layer_types") or {}
        kinds = [fixes.get(name, k) for name, k in zip(scene.meta["layers"], kinds)]
    return Loaded(scene, kept, applied, why, layers.item_types(scene, kinds))


def overrides(out_dir: Path) -> dict[int, bool]:
    f = out_dir / STATE
    if not f.exists():
        return {}
    return {int(k): v for k, v in json.loads(f.read_text(encoding="utf-8")).items()}


def _save(out_dir: Path, ov: dict[int, bool]) -> None:
    tmp = out_dir / (STATE + ".part")
    tmp.write_text(json.dumps({str(k): v for k, v in sorted(ov.items())}), encoding="utf-8")
    tmp.replace(out_dir / STATE)


def codes(L: Loaded, ov: dict[int, bool]) -> np.ndarray:
    """Render code per item: system's kept / deleted, engineer's keep / delete on top."""
    c = np.where(L.kept, KEPT, DELETED).astype(np.int8)
    if ov:
        idx = np.fromiter(ov.keys(), dtype=np.int64)
        c[idx] = np.where(np.fromiter(ov.values(), dtype=bool), SAY_KEEP, SAY_DELETE)
    return c


def reason(L: Loaded, i: int) -> Msg:
    """Why the system kept or deleted item i."""
    kind = layers.LABELS.get(str(L.kind[i])) or say(str(L.kind[i]), str(L.kind[i]))
    if L.why is not None:
        w = int(L.why[i])
        if w == -2:
            return say("silindi: kat planı dışında", "deleted: outside the floor plans")
        if w >= 0:
            tool = tools.LABELS[tools.TOOLS[w]]
            return say(f"silindi: {tool.tr} aracı", f"deleted: {tool.en} tool")
        if L.kept[i]:
            return say(f"kaldı: planda, hiçbir araç silmedi (katman tipi {kind.tr})",
                       f"kept: in the plan, no tool deleted it (layer type {kind.en})")
    if L.kept[i]:
        return say(f"kaldı: katman tipi {kind.tr}", f"kept: layer type {kind.en}")
    return say(f"silindi: katman tipi {kind.tr}", f"deleted: layer type {kind.en}")


# ------------------------------------------------------------------------------- selection
def pick(L: Loaded, x: float, y: float, tol: float, leaf: bool = False) -> list[int]:
    """Items of the entity nearest to (x, y) within `tol` metres (the whole top-level entity unless `leaf`)."""
    s = L.scene
    segs = s.segs
    ok = np.isfinite(segs).all(1)
    near = ok & (np.minimum(segs[:, 0], segs[:, 2]) <= x + tol) & (np.maximum(segs[:, 0], segs[:, 2]) >= x - tol) \
        & (np.minimum(segs[:, 1], segs[:, 3]) <= y + tol) & (np.maximum(segs[:, 1], segs[:, 3]) >= y - tol)
    k = np.flatnonzero(near)
    if not len(k):
        return []
    a, b = segs[k, :2], segs[k, 2:]
    ab = b - a
    t = np.clip(((np.array([x, y]) - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    q = a + t[:, None] * ab
    d = np.hypot(q[:, 0] - x, q[:, 1] - y)
    if d.min() > tol:
        return []
    item = int(s.seg_item[k[np.argmin(d)]])
    if leaf:
        return [item]
    return np.flatnonzero(s.top == s.top[item]).tolist()


def box(L: Loaded, x0: float, y0: float, x1: float, y1: float) -> list[int]:
    """Items lying wholly inside the box."""
    bb = L.scene.bboxes()
    inside = np.isfinite(bb).all(1) & (bb[:, 0] >= x0) & (bb[:, 1] >= y0) & (bb[:, 2] <= x1) & (bb[:, 3] <= y1)
    return np.flatnonzero(inside)[:MAX_SELECT].tolist()


def describe(L: Loaded, items: list[int], ov: dict[int, bool]) -> dict:
    """What the side panel shows for a selection."""
    s = L.scene
    idx = np.asarray(items, dtype=np.int64)
    if not len(idx):
        return {"items": [], "count": 0}
    by_layer: dict[int, list] = {}                # layer index -> [count, type label]
    for i in idx.tolist():
        row = by_layer.setdefault(int(s.layer[i]), [0, layers.LABELS.get(str(L.kind[i])) or str(L.kind[i])])
        row[0] += 1
    reasons: dict[Msg, int] = {}
    for i in idx.tolist():
        r = reason(L, i)
        reasons[r] = reasons.get(r, 0) + 1
    said = [ov[i] for i in idx.tolist() if i in ov]
    types = np.array(s.meta["types"])[s.etype[idx]]
    blocks = {s.meta["blocks"][b] for b in s.block_top[idx].tolist() if b >= 0}
    texts = [t for t in s.text[idx].tolist() if t][:5]
    seg_mask = np.isin(s.seg_item, idx) & np.isfinite(s.segs).all(1)
    segs = s.segs[seg_mask][:MAX_HIGHLIGHT_SEGS]
    return {"items": idx.tolist() if len(idx) <= 2000 else None, "count": int(len(idx)),
            "kept": int(L.kept[idx].sum()), "deleted": int((~L.kept[idx]).sum()),
            "layers": [{"name": s.meta["layers"][li], "type": t, "count": c}
                       for li, (c, t) in sorted(by_layer.items(), key=lambda kv: -kv[1][0])[:8]],
            "reasons": [{"text": t, "count": c} for t, c in sorted(reasons.items(), key=lambda kv: -kv[1])],
            "said": None if not said else ("keep" if all(said) else "delete" if not any(said) else "mixed"),
            "said_count": len(said),
            "types": sorted(set(types.tolist())), "blocks": sorted(blocks)[:5], "texts": texts,
            "segs": np.round(segs, 3).tolist()}


# ------------------------------------------------------------------------------- actions
def edit(out_dir: Path, L: Loaded, items: list[int], keep: bool | None, reasons: list[str] = (),
         note: str = "", user: str = "") -> dict:
    """Mark items keep (True) / delete (False) / back to the system's decision (None), and log it."""
    idx = [int(i) for i in items if 0 <= int(i) < L.scene.n][:MAX_SELECT]
    if not idx:
        raise ValueError(str(say("seçim boş", "nothing selected")))
    ov = overrides(out_dir)
    before = {i: ov.get(i) for i in idx}
    for i in idx:
        if keep is None:
            ov.pop(i, None)
        else:
            ov[i] = bool(keep)
    _save(out_dir, ov)
    s = L.scene
    bb = s.bboxes()[idx]
    bb = bb[np.isfinite(bb).all(1)]
    where = [round(float(v), 2) for v in (*bb[:, :2].min(0), *bb[:, 2:].max(0))] if len(bb) else None
    return store.append(out_dir, {
        "kind": "cleanup", "action": "mark", "user": user, "keep": keep, "items": idx, "bbox": where,
        "before": [before[i] for i in idx], "system_kept": L.kept[idx].tolist(),
        "layers": sorted({s.meta["layers"][li] for li in s.layer[idx].tolist()}),
        "system_reason": [m.both() for m in sorted({reason(L, i) for i in idx}, key=lambda m: m.tr)],
        "reasons": [str(r) for r in reasons][:20], "note": str(note)[:2000]})


def undo(out_dir: Path, user: str = "") -> dict:
    entries = [e for e in store.read(out_dir) if e.get("kind") == "cleanup"]
    undone = {e["target"] for e in entries if e["action"] == "undo"}
    for e in reversed(entries):
        if e["action"] != "mark" or e["seq"] in undone:
            continue
        ov = overrides(out_dir)
        for i, b in zip(e["items"], e["before"]):
            if b is None:
                ov.pop(i, None)
            else:
                ov[i] = b
        _save(out_dir, ov)
        return store.append(out_dir, {"kind": "cleanup", "action": "undo", "user": user, "target": e["seq"]})
    raise ValueError(str(say("geri alınacak işlem yok", "nothing to undo")))


def marks(out_dir: Path) -> list[dict]:
    """The engineer's marks still in force, newest first (the Gerekçeler list)."""
    entries = [e for e in store.read(out_dir) if e.get("kind") == "cleanup"]
    undone = {e["target"] for e in entries if e["action"] == "undo"}
    return [{k: e.get(k) for k in ("seq", "t", "user", "keep", "bbox", "layers", "system_reason", "reasons", "note")}
            | {"count": len(e["items"])} for e in reversed(entries) if e["action"] == "mark" and e["seq"] not in undone]


def export_job(job, output: str) -> dict:
    """clean_corrected.dxf / .dwg: the source drawing cleaned with the system's decision plus the overrides."""
    from cadai.cad import oda
    from cadai.cleanup.apply import DELETE, KEEP, write_clean
    from cadai.placement.place import OUTPUTS
    out_dir = (OUTPUTS / output).resolve()
    if OUTPUTS.resolve() not in out_dir.parents:
        raise RuntimeError(str(say(f"çıktı yok: {output}", f"no such output: {output}")))
    L = load(out_dir)
    ov = overrides(out_dir)
    decision = (L.applied.copy() if L.applied is not None else np.where(L.kept, KEEP, DELETE)).astype(np.int8)
    if L.applied is None:
        job.log(say("eski çıktı: taramaların yalnız sınırı bilgisi yok, kalan taramalar tam yazılır",
                    "older output: hatch outlines were not recorded, kept hatches are written whole"))
    for i, keep in ov.items():
        if not keep:
            decision[i] = DELETE
        elif decision[i] == DELETE:
            decision[i] = KEEP
    keep_n, del_n = sum(ov.values()), sum(not v for v in ov.values())
    job.log(say(f"mühendis düzeltmesi: {keep_n} nesne kalsın, {del_n} nesne silinsin",
                f"engineer's correction: keep {keep_n} items, delete {del_n} items"))
    src = Path(json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["source"])
    if not src.exists():
        raise RuntimeError(str(say(f"kaynak çizim bulunamadı: {src}", f"source drawing not found: {src}")))
    dxf = out_dir / "source.dxf"
    oda.to_dxf(src, dxf)
    try:
        stats = write_clean(dxf, L.scene, decision, out_dir / "clean_corrected.dxf", plain=True, to_cm=True)
    finally:
        dxf.unlink(missing_ok=True)
    job.log(say(f"düzeltilmiş temiz DXF: {stats}", f"corrected clean DXF: {stats}"))
    try:
        oda.to_dwg(out_dir / "clean_corrected.dxf", out_dir / "clean_corrected.dwg")
    except Exception as exc:  # noqa: BLE001 - the DXF is still usable
        job.log(say(f"DWG yazılamadı: {exc}", f"could not write the DWG: {exc}"))
    return {"output": output, **stats}
