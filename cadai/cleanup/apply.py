"""Clean a new drawing: the engineer's way, by layers, or item by item when layers can't separate it.

    DWG/DXF -> scene -> model (plans, unknown layers) -> mode (extract.py) -> decisions -> clean DXF + DWG

The architect's drawing is never redrawn or edited (ADR-020): the clean DXF is the original with
entities removed, every colour white, line weights off and the drawing scaled to centimetres. A block
reference whose contents are all deleted is removed whole; one with mixed contents is exploded
(ezdxf keeps the order of `virtual_entities()`, then the attributes as text, the same order the
scene paths use) and the rule applied again to its parts. The drawing gets the model's best guess
(p(delete) >= 0.5 deletes, guards keep walls); the uncertain band is listed in questions.json for
review (ADR-014).

Outputs go to DATA/outputs/<name>-<time>/: clean.dxf, clean.dwg, preview.png, questions.json,
checks.json (rooms vs the architect's area stamps, check/cleanup_checks.py) and summary.json.
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
from ezdxf import recover
from ezdxf.math import Matrix44

from cadai import config
from cadai.i18n import say
from cadai.cad import oda
from cadai.cad.render import render
from cadai.cad.scene import Scene, build
from cadai.check.cleanup_checks import check_cleanup
from cadai.cleanup import compare
from cadai.cleanup import agent, extract, features, layers, train, views

DELETE, KEEP, QUESTION = 0, 1, 3
OUTLINE = 2            # keep only the boundary of a hatch (columns and walls drawn as hatches, ADR-018)
CM_PER_M = 100.0
OUTPUT_DIR = config.DATA / "outputs"


def latest_model() -> Path:
    found = sorted(config.MODEL_DIR.glob("cleanup-*/item.txt"))
    if not found:
        raise RuntimeError(str(say("Eğitilmiş temizlik modeli yok; önce modeli eğitin.", "No trained cleanup model; train the model first.")))
    return found[-1].parent


def decide(scene: Scene, model_dir: Path, keep_furniture: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """p(delete), the decision and p(in a floor plan) per item. `keep_furniture` = the style switch (train.py): leave
    furniture, fixtures and cars in the base drawing instead of the strict cleanup."""
    p, decision, p_view, _, _ = predict(scene, model_dir, keep_furniture)
    return p, decision, p_view


def predict(scene: Scene, model_dir: Path, keep_furniture: bool = False, raw_features=None):
    """decide() plus the model's feature matrix and column names (the region finder reads them).
    `raw_features`: (X, names) already computed for this scene (the archive's feature cache)."""
    report = json.loads((model_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("version", 1) < 2:
        raise RuntimeError(str(say("Bu model eski biçimde; temizlik modelini yeniden eğitin.", "This model is in an old format; train the cleanup model again.")))
    view_model = lgb.Booster(model_file=str(model_dir / "view.txt"))
    item_model = lgb.Booster(model_file=str(model_dir / "item.txt"))
    if raw_features is None:
        X, names = features.compute(scene)
        V, vnames = views.item_features(scene, X, names)
        X, names = np.hstack([X, V]), names + vnames
    else:
        X, names = raw_features
    X, names = train.model_columns(X, names, float(keep_furniture))
    if names != report["features"]:
        raise RuntimeError(str(say("Model başka bir özellik sürümüyle eğitilmiş; modeli yeniden eğitin.",
                                   "The model was trained with another feature version; train it again.")))
    col = {n: i for i, n in enumerate(names)}
    p_view = view_model.predict(X)
    p = train.guarded(X, col, p_view, train.combine(p_view, item_model.predict(X)))
    t = report["thresholds"]
    decision = np.where(p >= t["delete"], DELETE, np.where(p <= t["keep"], KEEP, QUESTION))
    decision[np.bincount(scene.seg_item, minlength=scene.n) == 0] = KEEP        # nothing to judge
    return p, decision.astype(np.int8), p_view, X, names


def best_guess(p: np.ndarray, decision: np.ndarray) -> np.ndarray:
    """What goes into the drawing: sure decisions as they are, uncertain ones by p(delete) >= 0.5.
    Guarded items (walls, columns in a plan) are below 0.5 by construction, so they stay."""
    return np.where(decision == QUESTION, np.where(p >= 0.5, DELETE, KEEP), decision).astype(np.int8)


MTEXT_COLOUR = re.compile(r"\\[Cc]\d+;")          # inline "\C1;" (ACI) and "\c16711680;" (true colour)


def plain_style(doc) -> None:
    """The engineer's base drawing is plain thin white lines (ADR-016, ADR-020): every layer white
    (ACI 7: white on a dark screen, black on paper) with the default line weight, every entity in
    model space and in every block definition by layer, true colours and MTEXT colour codes removed,
    line weight display off."""
    for layer in doc.layers:
        layer.dxf.color = 7
        layer.dxf.discard("true_color")
        layer.dxf.lineweight = -3                                  # default
    for block in doc.blocks:
        for e in block:
            if e.dxf.hasattr("color"):
                e.dxf.color = 256
            e.dxf.discard("true_color")
            if e.dxf.is_supported("lineweight"):
                e.dxf.lineweight = -1                              # by layer
            if e.dxftype() == "MTEXT":
                e.text = MTEXT_COLOUR.sub("", e.text)
    doc.header["$LWDISPLAY"] = 0


def hatch_to_outline(layout, hatch, chord: float) -> int:
    """Replace a hatch by closed polylines along its boundary, on the same layer. `chord`: curve
    flattening tolerance in drawing units."""
    from ezdxf import path as ezpath
    added = 0
    for p in ezpath.from_hatch(hatch):
        pts = [(v.x, v.y) for v in p.flattening(chord)]
        if len(pts) >= 3:
            layout.add_lwpolyline(pts, close=True, dxfattribs={"layer": hatch.dxf.layer})
            added += 1
    layout.delete_entity(hatch)
    return added


def write_clean(dxf_in: Path, scene: Scene, decision: np.ndarray, dxf_out: Path, plain: bool = False,
                to_cm: bool = False) -> dict:
    """`decision` per scene item: DELETE, KEEP, or OUTLINE (hatch -> its boundary). `to_cm`: scale
    the drawing to centimetres, the unit of the engineers' drawings (256 of 258 finals, ADR-018)."""
    doc, _ = recover.readfile(str(dxf_in))
    msp = doc.modelspace()
    by_top: dict[int, list[tuple[list[str], int]]] = defaultdict(list)
    for path, top, d in zip(scene.path, scene.top, decision):
        by_top[int(top)].append((str(path).split("/")[1:], int(d)))
    stats = {"deleted": 0, "kept": 0, "exploded": 0, "outlined": 0}

    def resolve(entity, leaves):
        decs = {d for _, d in leaves}
        if not leaves or decs <= {KEEP, QUESTION}:
            stats["kept"] += 1
            return
        if decs == {DELETE}:
            msp.delete_entity(entity)
            stats["deleted"] += 1
            return
        if entity.dxftype() == "HATCH" and decs == {OUTLINE}:
            hatch_to_outline(msp, entity, 0.005 / scene.meta["unit"]["scale"])
            stats["outlined"] += 1
            return
        if entity.dxftype() != "INSERT":                    # one leaf can't be mixed; keep to be safe
            stats["kept"] += 1
            return
        n_virtual = len(list(entity.virtual_entities()))
        parts = entity.explode()
        stats["exploded"] += 1
        groups: dict[str, list] = defaultdict(list)
        for rel, d in leaves:
            groups[rel[0]].append((rel[1:], d))
        for j, child in enumerate(parts):
            key = str(j) if j < n_virtual else f"a{j - n_virtual}"
            resolve(child, groups.get(key, []))

    tops = list(msp)
    for i, entity in enumerate(tops):
        resolve(entity, by_top.get(i, []))
    if to_cm:
        factor = scene.meta["unit"]["scale"] * CM_PER_M
        if abs(factor - 1.0) > 1e-9:
            m = Matrix44.scale(factor)
            for e in list(msp):
                try:
                    e.transform(m)
                except Exception:  # noqa: BLE001 - an entity ezdxf can't transform stays as it is
                    stats["not_scaled"] = stats.get("not_scaled", 0) + 1
            doc.header["$LTSCALE"] = float(doc.header.get("$LTSCALE", 1.0)) * factor
        doc.header["$INSUNITS"] = 5                                   # centimetres
        stats["scaled_to_cm"] = factor
    if plain:
        plain_style(doc)
    tmp = dxf_out.with_suffix(".part")
    doc.saveas(tmp)
    tmp.replace(dxf_out)
    return stats


def questions(scene: Scene, p: np.ndarray, decision: np.ndarray) -> list[dict]:
    """Questions grouped by top-level entity, biggest first."""
    q = np.flatnonzero(decision == QUESTION)
    bb = scene.bboxes()
    length = scene.lengths()
    groups: dict[int, list[int]] = defaultdict(list)
    for i in q:
        groups[int(scene.top[i])].append(int(i))
    out = []
    for top, items in groups.items():
        b = bb[items]
        out.append({
            "handle": scene.meta["top_handles"][top],
            "layer": scene.meta["layers"][scene.layer[items[0]]],
            "items": len(items),
            "length_m": round(float(length[items].sum()), 2),
            "p_delete": round(float(p[items].mean()), 3),
            "bbox": [round(float(v), 2) for v in (np.nanmin(b[:, 0]), np.nanmin(b[:, 1]), np.nanmax(b[:, 2]), np.nanmax(b[:, 3]))],
        })
    return sorted(out, key=lambda r: -r["length_m"])


def layer_pictures(scene: Scene, table: list[dict], in_plan: np.ndarray, folder: Path, limit: int = 120) -> None:
    """A small picture of every plan layer (its plan geometry over the faint whole plan), for the
    engineer to see what a layer holds before choosing its type."""
    from PIL import Image, ImageDraw
    folder.mkdir(parents=True, exist_ok=True)
    segs, owner = scene.segs, scene.seg_item
    plan = in_plan[owner] & np.isfinite(segs).all(axis=1)
    if not plan.any():
        return
    s = segs[plan]
    x0, y0 = np.percentile(np.r_[s[:, 0], s[:, 2]], 0.5), np.percentile(np.r_[s[:, 1], s[:, 3]], 0.5)
    x1, y1 = np.percentile(np.r_[s[:, 0], s[:, 2]], 99.5), np.percentile(np.r_[s[:, 1], s[:, 3]], 99.5)
    w = 360
    scale = w / max(x1 - x0, 1e-9)
    h = int(min(max((y1 - y0) * scale, 60), 360))
    px = np.column_stack([(segs[:, 0] - x0) * scale, (y1 - segs[:, 1]) * scale,
                          (segs[:, 2] - x0) * scale, (y1 - segs[:, 3]) * scale])
    base = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(base)
    step = max(1, int(plan.sum() // 60000))                    # the faint background need not be complete
    for a, b, c, e in px[np.flatnonzero(plan)[::step]].tolist():
        d.line((a, b, c, e), fill=(225, 225, 225))
    layer_of_seg = scene.layer[owner]
    for row in table[:limit]:
        img = base.copy()
        d = ImageDraw.Draw(img)
        for a, b, c, e in px[plan & (layer_of_seg == row["index"])].tolist():
            d.line((a, b, c, e), fill=(200, 30, 30))
        img.save(folder / f"{row['index']}.png")


def clean_job(job, path: str, model: str | None = None, keep_furniture: bool = False, mode: str = "agent") -> dict:
    """`mode`: "agent" (toolbox, Claude picks the plans and checks the result, ADR-021), "tools" (toolbox
    with the statistical view model, no Claude), or the older "auto" / "layers" / "items" (ADR-020)."""
    src = Path(path)
    model_dir = config.MODEL_DIR / model if model else latest_model()
    out = OUTPUT_DIR / f"{src.stem[:40]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    job.progress(0, 5)
    t0 = time.time()
    dxf = out / "source.dxf"
    oda.to_dxf(src, dxf)
    job.log(say(f"DXF hazır ({time.time() - t0:.0f} s)", f"DXF ready ({time.time() - t0:.0f} s)"))
    job.progress(1)
    scene = build(dxf)
    job.log(say(f"sahne: {scene.n} nesne, birim {scene.meta['unit']['unit']}", f"scene: {scene.n} items, unit {scene.meta['unit']['unit']}"))
    job.progress(2)
    p, decision, p_view, X, names = predict(scene, model_dir, keep_furniture)
    in_plan = p_view >= 0.5
    # The model's best guess (ADR-014) judges unknown layers, or everything in item mode.
    model_applied = best_guess(p, decision)
    kinds, sources = layers.classify(scene)
    chosen, share = extract.choose_mode(scene, kinds, in_plan)
    agent_record = None
    why = None                       # which tool removed each item (tools.Result.attribution), for the Düzelt page
    if mode in ("agent", "tools"):
        used = mode
        job.log(say("yöntem: araç kutusu" + (" + Claude (görünüş seçimi ve kontrol)" if mode == "agent" else ""),
                    "method: toolbox" + (" + Claude (view choice and check)" if mode == "agent" else "")))
        result, in_plan, agent_record = agent.clean(scene, p_view, p,
                                                    X, names, out, use_claude=mode == "agent", log=job.log)
        applied, finishing, why = result.decision, result.stats, result.attribution
        (out / "agent.json").write_text(json.dumps(agent_record, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        used = chosen if mode == "auto" else mode
        known, wall = f"{share['known_share'] * 100:.0f}", f"{share['wall_share'] * 100:.0f}"
        job.log(say(f"yöntem: {'katmanlar' if used == 'layers' else 'nesne nesne'} (bilinen katman payı %{known}, duvar/kolon %{wall})",
                    f"method: {'layers' if used == 'layers' else 'item by item'} (known layer share {known}%, wall/column {wall}%)"))
        applied, finishing = extract.decide(scene, kinds, in_plan, model_applied, used)
    job.log(say(f"kurallar: {finishing}", f"rules: {finishing}"))
    counts = {"delete": int((applied == DELETE).sum()), "keep": int((applied != DELETE).sum()),
              "question": int((decision == QUESTION).sum()) if used == "items" else 0}
    job.log(f"model {model_dir.name}: {counts}")
    table = extract.layer_table(scene, kinds, sources, in_plan, applied)
    (out / "layers.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    layer_pictures(scene, table, in_plan, out / "layers")
    job.progress(3)
    stats = write_clean(dxf, scene, applied, out / "clean.dxf", plain=True, to_cm=True)
    # placement reads furniture from the raw drawing and needs what the cleanup kept (ADR-023); the Düzelt page
    # shows why each item went and writes the corrected drawing from the same decision (ADR-029)
    scene.save(out / "scene.npz")
    np.savez_compressed(out / "decision.npz", kept=applied != DELETE, applied=applied,
                        **({"why": why} if why is not None else {}))
    job.log(say(f"temiz DXF: {stats}", f"clean DXF: {stats}"))
    try:
        oda.to_dwg(out / "clean.dxf", out / "clean.dwg")
    except Exception as exc:  # noqa: BLE001 - the DXF is still usable
        job.log(say(f"DWG yazılamadı: {exc}", f"could not write the DWG: {exc}"))
    job.progress(4)
    # preview: sure delete pale red, sure keep dark, uncertain kept orange, uncertain deleted pale orange
    kept = applied != DELETE
    shown = np.where(kept, KEEP, DELETE)
    if used == "items":
        shown = np.where(decision == QUESTION, np.where(kept, 3, 4), shown)
    render(scene, out / "preview.png", shown, width=2400,
           colours={DELETE: (245, 190, 190), 4: (250, 215, 160), KEEP: (40, 40, 40), 3: (240, 140, 0)})
    if kept.any():
        bb = scene.bboxes()[kept]
        bbox = (np.nanpercentile(bb[:, 0], 0.5) - 2, np.nanpercentile(bb[:, 1], 0.5) - 2,
                np.nanpercentile(bb[:, 2], 99.5) + 2, np.nanpercentile(bb[:, 3], 99.5) + 2)
        render(scene, out / "clean.png", np.where(kept, shown, -9), bbox=bbox,
               colours={KEEP: (40, 40, 40), 3: (240, 140, 0)}, width=2400)
    qs = questions(scene, p, decision) if used == "items" else []
    (out / "questions.json").write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
    check = check_cleanup(scene, kept)
    (out / "checks.json").write_text(json.dumps(check, ensure_ascii=False, indent=1), encoding="utf-8")
    n_st, n_is = check.get("stamps_in_rooms", 0), len(check["issues"])
    job.log(say(f"kontrol: {check['rooms']} oda, {n_st} alan damgası karşılaştırıldı, {n_is} uyarı",
                f"check: {check['rooms']} rooms, {n_st} area labels compared, {n_is} warnings"))
    summary = {"source": str(src), "model": model_dir.name, "engine": "model", "keep_furniture": keep_furniture,
               "mode": used, "mode_auto": chosen, "layer_share": share, "finishing": finishing,
               "agent": {k: agent_record[k] for k in ("views", "cost_usd", "fixes") if k in agent_record} if agent_record else None,
               "counts": counts, "write": stats,
               "questions": len(qs), "seconds": round(time.time() - t0, 1), "unit": scene.meta["unit"],
               "check": {k: v for k, v in check.items() if k != "issues"}, "issues": len(check["issues"])}
    if compare.engineer_png(src, out / "engineer.png"):
        job.log(say("mühendisin temizliği de çizildi (arşiv projesi)", "the engineer's cleanup was drawn too (archive project)"))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    dxf.unlink(missing_ok=True)
    job.progress(5)
    return {"output": out.name, **summary}
