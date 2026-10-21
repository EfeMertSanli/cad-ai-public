"""Train and evaluate the cleanup models (docs/plan-cleanup.md §4, §6; ADR-004, ADR-006, ADR-011).

Two LightGBM models, combined:
- view model:  "is this raw item part of a floor plan?"  (target: alignment label not out-of-view)
- item model:  "inside a floor plan, does the engineer delete it?"  (keep/delete rows only)
p(delete) = 1 - p(in plan) * (1 - p(delete | in plan)).
Kept separate so the many out-of-view rows (all "delete") can't drag the in-plan decision towards
deleting walls (measured on the first, single-model version: 34 % of wall-like length deleted at 0.5).

Style: engineers differ in whether furniture, fixtures and cars stay in the base drawing (blocks
kept in a project: 10 % to 90 %, correlated -0.72 with in-plan accuracy). Each training project gets
`style_keep_blocks` = 1 when most of its in-plan block items were kept. On a new drawing the user
chooses; default 0 = strict cleanup (furniture deleted).

Split: by project, fixed by a hash of the project id (15 % test, 10 % validation). Nothing of a test
project is seen in training. Three bands (plan §0): thresholds on the validation projects so each
automatic band is >= 98 % precise; between them -> question.
Output: DATA/models/cleanup-<time>/ view.txt, item.txt, report.json and an error PNG per test project
(blue thick = deleted but should stay, magenta = kept but should go, at 0.5).
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np

from cadai import config
from cadai.i18n import say
from cadai.cad.render import render
from cadai.cad.scene import Scene
from cadai.cleanup import features, views
from cadai.cleanup.align import DELETE, KEEP, OUT_OF_VIEW
from cadai.corpus import inventory, process

TEST_SHARE = 0.15
VAL_SHARE = 0.10
# Memory: training rows x ~130 float32 features must fit next to everything else on a 16 GB
# machine. Test projects are evaluated one at a time, never held together.
MAX_ITEM_ROWS = 25_000             # in-plan rows per training project
MAX_OUT_ROWS = 10_000              # out-of-view rows per training project (view model)
MAX_VAL_ROWS = 30_000              # per validation project, uniform sample (thresholds stay unbiased)
TARGET_PRECISION = 0.98            # labels come from an alignment, not a human: 99.5 % is above their noise
DROP_FEATURES = {"rel_x", "rel_y"}  # position on the sheet is project-specific (ranked 6th in v1)
STYLE = "style_keep_blocks"
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "verbose": -1, "num_threads": 0}


def split_of(project: str) -> str:
    h = int(hashlib.sha1(project.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < TEST_SHARE:
        return "test"
    return "val" if h < TEST_SHARE + VAL_SHARE else "train"


def feature_matrix(raw_sha: str) -> tuple[np.ndarray, list[str]]:
    """Item + neighbourhood features of a raw scene, cached on D:."""
    cache = config.FEATURE_DIR / f"{process.key(raw_sha)}_f{features.VERSION}v{views.VERSION}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return z["X"], [str(s) for s in z["names"]]
    scene = Scene.load(process.scene_path(raw_sha))
    X, names = features.compute(scene)
    V, vnames = views.item_features(scene, X, names)
    X, names = np.hstack([X, V]), names + vnames
    np.savez_compressed(cache, X=X, names=np.array(names))
    return X, names


def model_columns(X: np.ndarray, names: list[str], style: float) -> tuple[np.ndarray, list[str]]:
    """The columns the models use: all features but DROP_FEATURES, plus the style switch."""
    keep = [i for i, n in enumerate(names) if n not in DROP_FEATURES]
    out = np.hstack([X[:, keep], np.full((len(X), 1), style, dtype=np.float32)])
    return out, [names[i] for i in keep] + [STYLE]


def wallish(X: np.ndarray, col: dict) -> np.ndarray:
    return (X[:, col["wall_twin"]] > 0.5) | (X[:, col["layer_wall"]] > 0) | (X[:, col["layer_column"]] > 0)


def project_style(X: np.ndarray, names: list[str], labels: np.ndarray) -> float:
    """1.0 when the engineer kept most in-plan block items that aren't walls (furniture, fixtures)."""
    col = {n: i for i, n in enumerate(names)}
    blocks = (X[:, col["in_block"]] > 0) & ~wallish(X, col) & np.isin(labels, (KEEP, DELETE))
    if blocks.sum() < 20:
        return 0.0
    return float((labels[blocks] == KEEP).mean() > 0.5)


def labelled_pairs() -> list[dict]:
    out = []
    for p in inventory.load()["projects"]:
        for pair in p["pairs"]:
            info = process.label_info(p["id"], pair["final"]) if pair["raw"] else None
            if info and info.get("ok"):
                out.append({"project": p["id"], "folder": p["folder"], "raw": pair["raw"], "final": pair["final"],
                            "split": split_of(p["id"])})
    return out


def _load_pair(pair, rng, full: bool = False):
    """Rows of one pair: model columns, alignment labels, view weights. `full` = every usable row."""
    X, names = feature_matrix(pair["raw"])
    with np.load(process.label_path(pair["project"], pair["final"])) as z:
        labels, shadow = z["labels"], z["shadow"]
    style = project_style(X, names, labels)
    usable = np.flatnonzero(np.isin(labels, (KEEP, DELETE, OUT_OF_VIEW)) & ~shadow)
    rows, view_w = usable, np.ones(len(usable))
    if not full and pair["split"] == "train":
        inside = usable[labels[usable] != OUT_OF_VIEW]
        out = usable[labels[usable] == OUT_OF_VIEW]
        if len(inside) > MAX_ITEM_ROWS:
            inside = rng.choice(inside, MAX_ITEM_ROWS, replace=False)
        out_w = 1.0
        if len(out) > MAX_OUT_ROWS:
            out_w = len(out) / MAX_OUT_ROWS
            out = rng.choice(out, MAX_OUT_ROWS, replace=False)
        rows = np.concatenate([inside, out])
        view_w = np.concatenate([np.ones(len(inside)), np.full(len(out), out_w)])
    elif not full and pair["split"] == "val" and len(usable) > MAX_VAL_ROWS:
        rows = np.sort(rng.choice(usable, MAX_VAL_ROWS, replace=False))
        view_w = np.ones(len(rows))
    Xm, mnames = model_columns(X[rows], names, style)
    return {"pair": pair, "rows": rows, "X": Xm, "labels": labels[rows], "view_w": view_w, "style": style}, mnames


def combine(p_view: np.ndarray, p_item: np.ndarray) -> np.ndarray:
    """p(delete) of the whole cleanup, before the guards."""
    return 1.0 - p_view * (1.0 - p_item)


GUARD_P = 0.49                     # below 0.5: a guarded item is never deleted automatically


def guarded(X: np.ndarray, col: dict, p_view: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Guards (plan §5, ADR-013): an item on a layer named wall / column / curtain wall that the view
    model places inside a floor plan is never deleted automatically; at most it becomes a question.
    Measured need: long exterior-wall polylines look like axes and frames to the item model."""
    protected = ((X[:, col["layer_wall"]] > 0) | (X[:, col["layer_column"]] > 0)) & (p_view >= 0.5)
    return np.where(protected, np.minimum(p, GUARD_P), p)


def _band_thresholds(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """t_delete: lowest p whose band (p >= t) is >= TARGET_PRECISION true deletes; t_keep likewise."""
    order = np.argsort(-p)
    prec = np.cumsum(y[order] == 1) / np.arange(1, len(p) + 1)
    ok = np.flatnonzero(prec >= TARGET_PRECISION)
    t_delete = float(p[order][ok.max()]) if len(ok) else 1.0
    order = np.argsort(p)
    prec = np.cumsum(y[order] == 0) / np.arange(1, len(p) + 1)
    ok = np.flatnonzero(prec >= TARGET_PRECISION)
    t_keep = float(p[order][ok.max()]) if len(ok) else 0.0
    return max(t_delete, 0.5), min(t_keep, 0.5)


def _evaluate(p, y, length, wall, t_delete, t_keep) -> dict:
    if not len(y):
        return {}
    auto_del, auto_keep = p >= t_delete, p <= t_keep
    question = ~(auto_del | auto_keep)
    wrong_del = auto_del & (y == 0)          # should stay, model deletes: the costly error
    wrong_keep = auto_keep & (y == 1)
    kept_wall = (y == 0) & wall
    return {
        "items": int(len(y)),
        "keep_items": int((y == 0).sum()),
        "accuracy_at_0.5": round(float(((p >= 0.5) == (y == 1)).mean()), 4),
        "wall_lost_share_at_0.5": round(float(length[kept_wall & (p >= 0.5)].sum() / max(length[kept_wall].sum(), 1e-9)), 4),
        "auto_share": round(float(1 - question.mean()), 4),
        "auto_precision": round(float(1 - (wrong_del.sum() + wrong_keep.sum()) / max((~question).sum(), 1)), 4),
        "question_items": int(question.sum()),
        "wrong_delete_items": int(wrong_del.sum()),
        "wrong_delete_m": round(float(length[wrong_del].sum()), 1),
        "wrong_delete_wall_m": round(float(length[wrong_del & wall].sum()), 1),
        "kept_wall_m": round(float(length[kept_wall].sum()), 1),
        "wrong_keep_items": int(wrong_keep.sum()),
        "wrong_keep_m": round(float(length[wrong_keep].sum()), 1),
    }


def _fit(X, y, w, names, Xv=None, yv=None, rounds=800):
    dtrain = lgb.Dataset(X, y, weight=w, feature_name=names, categorical_feature=["etype"], free_raw_data=True)
    valid = [lgb.Dataset(Xv, yv, reference=dtrain)] if Xv is not None and len(yv) else []
    return lgb.train(PARAMS, dtrain, num_boost_round=rounds, valid_sets=valid or None,
                     callbacks=[lgb.early_stopping(40, verbose=False)] if valid else None)


def train_job(job, rounds: int = 800) -> dict:
    pairs = labelled_pairs()
    splits = ", ".join(f"{s} {sum(1 for p in pairs if p['split'] == s)}" for s in ("train", "val", "test"))
    job.log(say(f"{len(pairs)} etiketli çift: {splits}", f"{len(pairs)} labelled pairs: {splits}"))
    if len(pairs) < 5:
        raise RuntimeError(str(say("Eğitim için en az 5 etiketli çift gerekiyor; önce dönüştürme ve etiketleme işlerini çalıştırın.",
                                   "Training needs at least 5 labelled pairs; run the conversion and labelling jobs first.")))
    rng = np.random.default_rng(0)
    data: dict[str, list] = {"train": [], "val": [], "test": []}
    names: list[str] = []
    job.progress(0, len(pairs) + 4)
    for i, pair in enumerate(pairs, 1):
        if pair["split"] == "test":                    # evaluated one by one after training
            data["test"].append({"pair": pair})
        else:
            part, names = _load_pair(pair, rng)
            data[pair["split"]].append(part)
        job.progress(i)
        if job.cancelled:
            return {}
    col = {n: i for i, n in enumerate(names)}
    styles = [p["style"] for p in data["train"]]
    job.log(say(f"tarz: eğitim projelerinin {sum(styles)}/{len(styles)} tanesinde mühendis mobilya/blokları tutmuş",
                f"style: in {sum(styles)}/{len(styles)} training projects the engineer kept furniture/blocks"))

    def stacked(split):
        parts = data[split]
        return np.vstack([p["X"] for p in parts]), np.concatenate([p["labels"] for p in parts]), \
            np.concatenate([p["view_w"] for p in parts])

    Xtr, Ltr, Wtr = stacked("train")
    Xva, Lva, _ = stacked("val") if data["val"] else (None, np.zeros(0), None)
    length_w = 1.0 + np.log1p(Xtr[:, col["length"]] * 10)                # long lines matter more on the sheet

    t0 = time.time()
    view_model = _fit(Xtr, (Ltr != OUT_OF_VIEW).astype(np.int8), Wtr * length_w, names,
                      Xva, (Lva != OUT_OF_VIEW).astype(np.int8) if Xva is not None else None, rounds)
    trees = view_model.best_iteration or rounds
    job.log(say(f"görünüş modeli: {trees} ağaç, {time.time() - t0:.0f} s", f"view model: {trees} trees, {time.time() - t0:.0f} s"))
    job.progress(len(pairs) + 1)
    t0 = time.time()
    tr_in = Ltr != OUT_OF_VIEW
    va_in = Lva != OUT_OF_VIEW
    item_model = _fit(Xtr[tr_in], (Ltr[tr_in] == DELETE).astype(np.int8), length_w[tr_in], names,
                      Xva[va_in] if Xva is not None else None, (Lva[va_in] == DELETE).astype(np.int8), rounds)
    trees = item_model.best_iteration or rounds
    job.log(say(f"plan içi model: {trees} ağaç, {time.time() - t0:.0f} s", f"in-plan model: {trees} trees, {time.time() - t0:.0f} s"))
    job.progress(len(pairs) + 2)
    del Xtr

    if Xva is not None:
        pv = view_model.predict(Xva)
        t_delete, t_keep = _band_thresholds(guarded(Xva, col, pv, combine(pv, item_model.predict(Xva))),
                                            (Lva != KEEP).astype(np.int8))
    else:
        t_delete, t_keep = 0.9, 0.1
    job.log(say(f"eşikler: sil ≥ {t_delete:.3f}, tut ≤ {t_keep:.3f}", f"thresholds: delete ≥ {t_delete:.3f}, keep ≤ {t_keep:.3f}"))

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = config.MODEL_DIR / f"cleanup-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    view_model.save_model(str(out_dir / "view.txt"))
    item_model.save_model(str(out_dir / "item.txt"))

    per_project = []
    acc: dict[str, list] = {k: [] for k in ("p", "p_raw", "y", "length", "wall", "in_view")}
    for part in data["test"]:
        full, _ = _load_pair(part["pair"], rng, full=True)              # this one project, all rows
        X, L = full["X"], full["labels"]
        pv = view_model.predict(X)
        raw_p = combine(pv, item_model.predict(X))
        p, y = guarded(X, col, pv, raw_p), (L != KEEP).astype(np.int8)
        named = (X[:, col["layer_wall"]] > 0) | (X[:, col["layer_column"]] > 0)
        m = _evaluate(p, y, X[:, col["length"]], named, t_delete, t_keep)
        m.update(project=part["pair"]["project"], folder=part["pair"]["folder"], final=part["pair"]["final"],
                 style=full["style"])
        per_project.append(m)
        _error_png(out_dir, full, p)
        for k, v in (("p", p), ("p_raw", raw_p), ("y", y), ("length", X[:, col["length"]]), ("wall", named),
                     ("in_view", L != OUT_OF_VIEW)):
            acc[k].append(v)
        part["n"] = len(y)
    job.progress(len(pairs) + 3)

    def importance(model):
        return sorted(({"feature": n, "gain": round(float(g), 1)} for n, g in
                       zip(names, model.feature_importance("gain"))), key=lambda r: -r["gain"])[:30]

    report = {
        "kind": "cleanup",
        "version": 2,
        "created": stamp,
        "pairs": {s: len(data[s]) for s in data},
        "rows": {s: int(sum(len(p["labels"]) if "labels" in p else p.get("n", 0) for p in data[s])) for s in data},
        "trees": {"view": int(view_model.best_iteration or rounds), "item": int(item_model.best_iteration or rounds)},
        "thresholds": {"delete": t_delete, "keep": t_keep, "target_precision": TARGET_PRECISION},
        "style_train_share": round(float(np.mean(styles)), 3) if styles else None,
        "test": {},
        "test_in_view": {},
        "test_projects": sorted(per_project, key=lambda m: -m.get("wall_lost_share_at_0.5", 0)),
        "importance": importance(item_model),
        "importance_view": importance(view_model),
        "features": names,
    }
    if acc["p"]:
        pte, yte, lte, wte = (np.concatenate(acc[k]) for k in ("p", "y", "length", "wall"))
        inv = np.concatenate(acc["in_view"]).astype(bool)
        report["test"] = _evaluate(pte, yte, lte, wte, t_delete, t_keep)
        report["test_in_view"] = _evaluate(pte[inv], yte[inv], lte[inv], wte[inv], t_delete, t_keep)
        report["test_in_view_without_guards"] = _evaluate(np.concatenate(acc["p_raw"])[inv], yte[inv], lte[inv],
                                                          wte[inv], t_delete, t_keep)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    job.progress(len(pairs) + 4)
    job.log(say(f"test (uçtan uca): {report['test']}", f"test (end to end): {report['test']}"))
    job.log(say(f"test (plan içi): {report['test_in_view']}", f"test (in plan): {report['test_in_view']}"))
    return {"model": out_dir.name, "test": report["test"], "test_in_view": report["test_in_view"]}


def _error_png(out_dir, part, p) -> None:
    """Raw drawing of a test project at threshold 0.5: dark kept, pale red deleted; errors thick
    (blue = deleted but should stay, magenta = kept but should go)."""
    pair, rows, L = part["pair"], part["rows"], part["labels"]
    scene = Scene.load(process.scene_path(pair["raw"]))
    info = process.label_info(pair["project"], pair["final"])
    y = L != KEEP
    pred_del = p >= 0.5
    colour = np.full(scene.n, 2, dtype=np.int8)                       # not evaluated
    c = np.where(pred_del, 10, 11)
    c = np.where(pred_del & ~y, 13, c)
    c = np.where(~pred_del & y, 14, c)
    colour[rows] = c
    colours = {2: (225, 225, 235), 10: (245, 200, 200), 11: (60, 60, 60), 14: (200, 0, 200), 13: (0, 90, 255)}
    r = np.array([t["region"] for t in info["transforms"]])
    pad = 15
    bbox = (r[:, 0].min() - pad, r[:, 1].min() - pad, r[:, 2].max() + pad, r[:, 3].max() + pad)
    render(scene, out_dir / f"{pair['project']}__{process.key(pair['final'])}.png", colour, bbox=bbox,
           width=2400, colours=colours, thick={13: 3, 14: 2})
