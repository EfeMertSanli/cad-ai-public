"""Synthetic rooms and candidates -> placement train job -> report."""
import json

import numpy as np

from cadai import jobs
from cadai.placement import dataset, model
from cadai.placement.candidates import FEATURES


def test_train_placement_on_synthetic_rooms(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "DATASET", tmp_path / "rooms.jsonl")
    monkeypatch.setattr(dataset, "CANDIDATE_DIR", tmp_path / "cand")
    dataset.CANDIDATE_DIR.mkdir()
    rng = np.random.default_rng(0)
    rows, X, Y, room, xy = [], [], [], [], []
    for r in range(120):
        n = 40
        pts = rng.uniform(0, 4, size=(n, 2)) + (r * 10, 0)
        feats = np.zeros((n, len(FEATURES)), dtype=np.float32)
        feats[:, FEATURES.index("on_wall")] = 1
        feats[:, FEATURES.index("d_door")] = rng.uniform(0, 4, n)
        feats[:, FEATURES.index("area")] = 12
        y = np.zeros((n, len(dataset.CLASSES)), dtype=bool)
        near_door = np.argsort(feats[:, FEATURES.index("d_door")])[:2]           # sockets next to the door
        y[near_door, dataset.CLASSES.index("socket")] = True
        rows.append({"project": f"p{r % 12}", "final": "f", "view": 0, "type": "bedroom", "area_m2": 12.0, "bbox": [r * 10, 0, r * 10 + 4, 4],
                     "devices": [{"type": "socket", "x": float(pts[i, 0]), "y": float(pts[i, 1])} for i in near_door]})
        X.append(feats)
        Y.append(y)
        room.append(np.full(n, r))
        xy.append(pts)
    dataset.DATASET.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    np.savez(dataset.CANDIDATE_DIR / "all.npz", X=np.vstack(X), Y=np.vstack(Y), room=np.concatenate(room),
             xy=np.vstack(xy))
    monkeypatch.setattr(model, "split_of", lambda pid: "test" if pid in ("p0", "p1") else "train")
    job = jobs.Job(id="t", kind="t", title="t", params={})
    res = model.train_job(job)
    socket = res["classes"]["socket"]
    assert socket["test_rooms"] == 20 and socket["test_devices"] == 40
    # the two best candidates can be < 0.5 m apart, then spacing keeps only one: recall below 1 is right
    assert socket["count_exact"] > 0.9 and socket["recall_true_count"] >= 0.75
