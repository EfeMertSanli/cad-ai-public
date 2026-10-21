"""Placement model: how many devices per room, and where (plan-placement M2.1-M2.4, ADR-010).

Two learned parts per device class, both LightGBM:
- count: Poisson regression on room rows (type, area, proportions, doors, furniture, and the room's
  geometry summed up from its candidates: perimeter, window length, nearest furniture of each size);
- rank: candidate positions (candidates.py) ranked against the other candidates of the same room
  (lambdarank, ADR-026): "of this room's points, the engineer put one here".
Placing a room = take the predicted count, then the best-scored candidates of the right kind
(wall or ceiling) at least 0.5 m apart. Rules only check afterwards (M3b), they don't place.

Baseline it must beat (ADR-022): "copy the most similar archive room": the training room of the same
type closest in area and proportions lends its devices, in coordinates relative to its bounding
box, each snapped to the nearest free candidate of the right kind.

Evaluation on the test projects (same project split as the cleanup):
- count: mean absolute error per class, and the share of rooms with the exact count;
- position: precision / recall of placed devices against the engineer's within 0.5 m, with the
  predicted count and with the true count (the second isolates the ranking).
"""
from __future__ import annotations

import json
from datetime import datetime

import lightgbm as lgb
import numpy as np
from scipy.spatial import cKDTree

from cadai import config
from cadai.i18n import say
from cadai.cleanup.train import split_of
from cadai.placement import dataset
from cadai.placement.candidates import CEILING_CLASSES, FEATURES, NEIGHBOUR_FEATURES, ROOM_CODES, SIZE_BINS

ROOM_FEATURES = ["room_type", "area", "width", "height", "aspect", "n_doors", "n_furniture",
                 "n_wet_doors", "n_neighbours", "n_out_doors"]
# Features a class's models leave out: the rooms behind the doors lifted lights, sockets and switches
# but lowered wet sockets on both validation and test projects (ADR-027).
EXCLUDED = {"socket_wet": set(NEIGHBOUR_FEATURES)}
CATEGORICAL = ["room_type", "nb_type"]
MIN_SPACING_M = 0.5
HIT_M = 0.5
NEAR_M = 1.0               # second, looser measure: right wall, a little off (engineers differ this much)
MIN_POSITIVES = 30
# A floor plan the engineer left without devices (the typical floor is drawn once, the others stay
# empty) would teach "rooms get nothing": such views are left out of training and evaluation (ADR-023).
DRAWN_CLASSES = ("light_ceiling", "socket", "socket_wet", "switch")
LIVING_TYPES = ("bedroom", "living", "living_kitchen", "kitchen", "bathroom", "hall")
MIN_VIEW_DEVICES = 1.0      # devices of DRAWN_CLASSES per living room of the view


def _load():
    rows = [json.loads(line) for line in open(dataset.DATASET, encoding="utf-8")]
    X, Y, room, xy = [], [], [], []
    for f in sorted(dataset.CANDIDATE_DIR.glob("*.npz")):
        with np.load(f) as z:
            X.append(z["X"])
            Y.append(z["Y"])
            room.append(z["room"])
            xy.append(z["xy"])
    return rows, np.vstack(X), np.vstack(Y), np.concatenate(room), np.vstack(xy)


def _room_matrix(rows, X, room):
    """Count model rows (COUNT_FEATURES), one per room, from the room's candidates."""
    order = np.argsort(room, kind="stable")
    starts = np.searchsorted(room[order], np.arange(len(rows) + 1))
    M = np.full((len(rows), len(COUNT_FEATURES)), np.nan, dtype=np.float32)
    for k, row in enumerate(rows):
        c = order[starts[k]:starts[k + 1]]
        if len(c):
            M[k] = room_row(X[c])
        else:                                         # rooms without candidates still have type and area
            M[k, 0], M[k, 1] = ROOM_CODES.get(row["type"], 0), row["area_m2"]
    return M


# Count = floor(predicted mean + threshold). Plain rounding (0.5) under-places a class the model is
# unsure about; per class best 1 m F1 on the validation projects (ADR-026; ADR-023 had tuned switch
# 0.7 on the test projects, with the richer room rows 0.5 is best).
COUNT_ROUND = {"socket_wet": 0.6, "panel": 0.7, "light_emergency": 0.7}


def device_counts(mean: np.ndarray, cls: str) -> np.ndarray:
    """Device counts from the count model's predicted means."""
    return np.floor(np.asarray(mean) + COUNT_ROUND.get(cls, 0.5)).astype(int)


# Rooms whose name the raw drawing doesn't give (28 % of test rooms) get a type from their geometry:
# area, proportions, doors, windows, furniture sizes. 70 % right on typed test rooms; on untyped
# rooms it lifts 1 m F1 (light .567 -> .579, socket .329 -> .361, wet 0 -> .255, switch 0 -> .146),
# more than Haiku naming the room from a picture (39 % right) did (ADR-023).
TYPE_BASE = ["area", "width", "height", "aspect", "n_doors", "n_furniture"]
TYPE_MIN = ["d_window", "d_furniture"] + SIZE_BINS
TYPE_FEATURES = TYPE_BASE + [f"min_{f}" for f in TYPE_MIN] + ["max_d_wall", "max_furniture_long"]
TYPE_NAMES = list(ROOM_CODES)             # in code order: a new type is appended, older models still read


def type_features(Xc: np.ndarray) -> np.ndarray:
    """One room's geometry from its candidates' feature rows."""
    col = FEATURES.index
    return np.r_[Xc[0, [col(f) for f in TYPE_BASE]], Xc[:, [col(f) for f in TYPE_MIN]].min(0),
                 Xc[:, col("d_wall")].max(), Xc[:, col("furniture_long")].max()].astype(np.float32)


# The count model sees the room's geometry summed up from its candidates too: the room-type
# features, the perimeter (wall candidates every 25 cm), window length and inner depth (ADR-026).
COUNT_FEATURES = ROOM_FEATURES + [f"t_{f}" for f in TYPE_FEATURES] + ["perimeter", "window_len", "inner_depth"]


def room_row(Xc: np.ndarray) -> np.ndarray:
    """COUNT_FEATURES of one room from its candidates' feature rows."""
    col = FEATURES.index
    wall = Xc[:, col("on_wall")] == 1
    return np.r_[Xc[0, [col(f) for f in ROOM_FEATURES]], type_features(Xc),
                 wall.sum() * 0.25, (Xc[wall, col("d_window")] < 0.3).sum() * 0.25,
                 Xc[~wall, col("d_wall")].max() if (~wall).any() else 0.0].astype(np.float32)


def train_room_types(rows, X, room, split, out_dir) -> float:
    """Room type classifier -> out_dir/room_type.txt; accuracy on the typed test rooms."""
    order = np.argsort(room, kind="stable")
    starts = np.searchsorted(room[order], np.arange(len(rows) + 1))
    F = np.full((len(rows), len(TYPE_FEATURES)), np.nan, dtype=np.float32)
    for r in range(len(rows)):
        c = order[starts[r]:starts[r + 1]]
        if len(c):
            F[r] = type_features(X[c])
    lab = np.array([TYPE_NAMES.index(r["type"]) if r["type"] in TYPE_NAMES else -1 for r in rows])
    ok = (lab >= 0) & ~np.isnan(F[:, 0])
    tr, te = ok & (split == "train"), ok & (split == "test")
    m = lgb.train({"objective": "multiclass", "num_class": len(TYPE_NAMES), "learning_rate": 0.05, "num_leaves": 31,
                   "verbose": -1}, lgb.Dataset(F[tr], lab[tr], feature_name=TYPE_FEATURES), num_boost_round=300)
    m.save_model(str(out_dir / "room_type.txt"))
    return round(float((m.predict(F[te]).argmax(1) == lab[te]).mean()), 3) if te.any() else None


# A "room" of more candidates is a merged open area, not a room (lambdarank also takes at most
# 10 000 rows per query).
RANK_MAX_CANDIDATES = 5000


def rank_groups(room: np.ndarray, sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Candidate indices of `sel` ordered by room, and the size of each room's group (lambdarank)."""
    idx = np.flatnonzero(sel)
    idx = idx[np.argsort(room[idx], kind="stable")]
    idx = idx[np.bincount(room[idx])[room[idx]] <= RANK_MAX_CANDIDATES]
    sizes = np.bincount(room[idx])
    return idx, sizes[sizes > 0]


def drawn_views(rows: list[dict]) -> np.ndarray:
    """Per room: does its floor plan carry the engineer's devices (MIN_VIEW_DEVICES per living room)?"""
    per_view: dict[tuple, list[int]] = {}
    for r in rows:
        if r["type"] in LIVING_TYPES:
            k = (r["project"], r["final"], r["view"])
            per_view.setdefault(k, []).append(sum(d["type"] in DRAWN_CLASSES for d in r["devices"]))
    ok = {k for k, n in per_view.items() if np.mean(n) >= MIN_VIEW_DEVICES}
    return np.array([(r["project"], r["final"], r["view"]) in ok for r in rows])


# Least distance between two devices of a class in one room, from the archive's same-room nearest
# neighbours (lights: 10th percentile 1.1 m, quartile 1.8 m; switches 1.0; wet sockets 0.3 = drawn in
# pairs). Lights at 0.5 / 1.0 / 1.5 / 2.0 m: 1 m F1 0.624 / 0.651 / 0.663 / 0.665 (ADR-023).
SPACING_M = {"light_ceiling": 1.5, "switch": 1.0, "socket_wet": 0.3}


def place_room(scores: np.ndarray, xy: np.ndarray, n: int, spacing: float = MIN_SPACING_M) -> np.ndarray:
    """Indices of up to n candidates, best first, at least `spacing` apart."""
    chosen: list[int] = []
    for i in np.argsort(-scores):
        if len(chosen) >= n:
            break
        if all(np.hypot(*(xy[i] - xy[j])) >= spacing for j in chosen):
            chosen.append(int(i))
    return np.array(chosen, dtype=int)


def _match(pred: np.ndarray, truth: np.ndarray, radius: float = HIT_M) -> int:
    """Greedy one-to-one matches within `radius`."""
    if not len(pred) or not len(truth):
        return 0
    d, idx = cKDTree(truth).query(pred, k=min(3, len(truth)), distance_upper_bound=radius)
    d, idx = d.reshape(len(pred), -1), idx.reshape(len(pred), -1)
    used, hits = set(), 0
    for row_d, row_i in zip(d, idx):
        for dd, ii in zip(row_d, row_i):
            if np.isfinite(dd) and ii not in used:
                used.add(ii)
                hits += 1
                break
    return hits


def _shape(row: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = row["bbox"]
    w, h = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    return float(np.log(max(row["area_m2"], 0.1))), float(max(w, h) / min(w, h))


def copy_similar(rows: list[dict], library: np.ndarray, target: int, cls: str, cand_idx: np.ndarray,
                 xy: np.ndarray) -> np.ndarray:
    """Candidate indices for room `target`: the devices of class `cls` of the most similar library
    room (same type, nearest in log-area and aspect), moved by relative bounding-box position and
    snapped to the nearest unused candidate."""
    t = rows[target]
    same = [k for k in library if rows[k]["type"] == t["type"]]
    if not same or not len(cand_idx):
        return np.zeros(0, dtype=int)
    la, asp = _shape(t)
    best = min(same, key=lambda k: abs(_shape(rows[k])[0] - la) + 0.5 * abs(_shape(rows[k])[1] - asp))
    src = rows[best]
    sx0, sy0, sx1, sy1 = src["bbox"]
    tx0, ty0, tx1, ty1 = t["bbox"]
    pick: list[int] = []
    free = np.ones(len(cand_idx), dtype=bool)
    for d in src["devices"]:
        if d["type"] != cls:
            continue
        u = (d["x"] - sx0) / max(sx1 - sx0, 1e-3)
        v = (d["y"] - sy0) / max(sy1 - sy0, 1e-3)
        q = np.array([tx0 + u * (tx1 - tx0), ty0 + v * (ty1 - ty0)])
        dist = np.where(free, np.hypot(*(xy[cand_idx] - q).T), np.inf)
        if np.isfinite(dist).any():
            j = int(np.argmin(dist))
            free[j] = False
            pick.append(int(cand_idx[j]))
    return np.array(pick, dtype=int)


def train_job(job) -> dict:
    rows, X, Y, room, xy = _load()
    proposals = any(r.get("proposals_used") for r in rows)
    split = np.array([split_of(r["project"]) for r in rows])
    drawn = drawn_views(rows)
    split = np.where(drawn, split, "empty_view")
    cand_split = split[room]
    R = _room_matrix(rows, X, room)
    on_wall = X[:, FEATURES.index("on_wall")] == 1
    off = int((~drawn).sum())
    job.log(say(f"{len(rows)} oda ({off} cihazsız katta, dışarıda), {len(X)} aday" + (" · LLM önerisi tiplerle (ön sonuç)" if proposals else ""),
                f"{len(rows)} rooms ({off} on floors without devices, left out), {len(X)} candidates"
                + (" · with LLM-proposed types (preliminary)" if proposals else "")))
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = config.MODEL_DIR / f"placement-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"kind": "placement", "created": stamp, "proposals_used": proposals, "rooms": len(rows),
              "candidates": int(len(X)), "classes": {}}
    classes = [c for c in dataset.CLASSES if Y[:, dataset.CLASSES.index(c)].sum() >= MIN_POSITIVES]
    job.progress(0, len(classes))
    for k, cls in enumerate(classes):
        c = dataset.CLASSES.index(cls)
        counts = np.array([sum(1 for d in r["devices"] if d["type"] == cls) for r in rows], dtype=float)
        tr, te = split == "train", split == "test"
        rf = [f for f in COUNT_FEATURES if f not in EXCLUDED.get(cls, ())]
        ff = [f for f in FEATURES if f not in EXCLUDED.get(cls, ())]
        rcols, fcols = [COUNT_FEATURES.index(f) for f in rf], [FEATURES.index(f) for f in ff]
        count_model = lgb.train({"objective": "poisson", "learning_rate": 0.05, "num_leaves": 31,
                                 "min_data_in_leaf": 20, "verbose": -1},
                                lgb.Dataset(R[np.ix_(tr, rcols)], counts[tr], feature_name=rf,
                                            categorical_feature=[f for f in CATEGORICAL if f in rf]), num_boost_round=300)
        kind = ~on_wall if cls in CEILING_CLASSES else on_wall
        idx, groups = rank_groups(room, (cand_split == "train") & kind)
        rank_model = lgb.train({"objective": "lambdarank", "lambdarank_truncation_level": 10, "learning_rate": 0.05,
                                "num_leaves": 127, "min_data_in_leaf": 30, "verbose": -1},
                               lgb.Dataset(X[np.ix_(idx, fcols)], Y[idx, c].astype(int), group=groups, feature_name=ff,
                                           categorical_feature=[f for f in CATEGORICAL if f in ff]), num_boost_round=800)
        count_model.save_model(str(out_dir / f"count_{cls}.txt"))
        rank_model.save_model(str(out_dir / f"rank_{cls}.txt"))

        # evaluation on the test rooms
        pred_count = device_counts(count_model.predict(R[np.ix_(te, rcols)]), cls) if te.any() else np.zeros(0, int)
        true_count = counts[te].astype(int)
        hits = placed = truth_n = hits_oracle = placed_oracle = hits_copy = placed_copy = hits_near = 0
        library = np.flatnonzero(tr & np.array([bool(r["type"]) for r in rows]))
        test_rooms = np.flatnonzero(te)
        cand_by_room = {r: np.flatnonzero((room == r) & kind) for r in test_rooms}
        scores_all = np.zeros(len(X))
        sel = (cand_split == "test") & kind
        if sel.any():
            scores_all[sel] = rank_model.predict(X[np.ix_(sel, fcols)])
        for r, n_pred, n_true in zip(test_rooms, pred_count, true_count):
            idx = cand_by_room[r]
            truth = np.array([[d["x"], d["y"]] for d in rows[r]["devices"] if d["type"] == cls]).reshape(-1, 2)
            truth_n += len(truth)
            if not len(idx):
                continue
            copy = copy_similar(rows, library, int(r), cls, idx, xy)
            hits_copy, placed_copy = hits_copy + _match(xy[copy], truth), placed_copy + len(copy)
            for n, oracle in ((n_pred, False), (n_true, True)):
                pick = idx[place_room(scores_all[idx], xy[idx], int(n), SPACING_M.get(cls, MIN_SPACING_M))]
                h = _match(xy[pick], truth)
                if oracle:
                    hits_oracle, placed_oracle = hits_oracle + h, placed_oracle + len(pick)
                else:
                    hits, placed = hits + h, placed + len(pick)
                    hits_near += _match(xy[pick], truth, NEAR_M)
        report["classes"][cls] = {
            "train_rooms": int(tr.sum()), "test_rooms": int(te.sum()), "test_devices": int(truth_n),
            "count_mae": round(float(np.abs(pred_count - true_count).mean()), 3) if te.any() else None,
            "count_exact": round(float((pred_count == true_count).mean()), 3) if te.any() else None,
            "precision": round(hits / placed, 3) if placed else None,
            "recall": round(hits / truth_n, 3) if truth_n else None,
            "precision_true_count": round(hits_oracle / placed_oracle, 3) if placed_oracle else None,
            "recall_true_count": round(hits_oracle / truth_n, 3) if truth_n else None,
            "precision_1m": round(hits_near / placed, 3) if placed else None,
            "recall_1m": round(hits_near / truth_n, 3) if truth_n else None,
            "copy_precision": round(hits_copy / placed_copy, 3) if placed_copy else None,
            "copy_recall": round(hits_copy / truth_n, 3) if truth_n else None,
            "rank_importance": sorted(({"feature": n, "gain": round(float(g), 1)} for n, g in
                                       zip(FEATURES, rank_model.feature_importance("gain"))), key=lambda x: -x["gain"])[:8],
        }
        m = report["classes"][cls]
        job.log(say(f"{cls}: sayı hatası {m['count_mae']}, model kesinlik {m['precision']} duyarlılık {m['recall']} · "
                    f"benzer oda kopyası kesinlik {m['copy_precision']} duyarlılık {m['copy_recall']}",
                    f"{cls}: count error {m['count_mae']}, model precision {m['precision']} recall {m['recall']} · "
                    f"similar-room copy precision {m['copy_precision']} recall {m['copy_recall']}"))
        job.progress(k + 1)
    report["room_type_accuracy"] = train_room_types(rows, X, room, split, out_dir)
    job.log(say(f"oda tipi modeli (adsız odalar için): test doğruluğu {report['room_type_accuracy']}",
                f"room type model (for unnamed rooms): test accuracy {report['room_type_accuracy']}"))
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"model": out_dir.name, "classes": {c: {k: v for k, v in m.items() if k != "rank_importance"}
                                                for c, m in report["classes"].items()}}
