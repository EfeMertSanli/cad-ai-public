"""Synthetic end to end: scenes -> labelling worker -> training job -> report."""
import json

import numpy as np

from cadai import config, jobs
from cadai.cleanup import train
from cadai.corpus import process
from conftest import make_scene


def _project(seed):
    rng = np.random.default_rng(seed)
    items, final = [], []
    x = 0.0
    for _ in range(30):                                                          # enough walls to vote (MIN_VOTES)
        w, h = rng.uniform(2.5, 6), rng.uniform(3, 7)
        for off in (0.0, 0.2):
            seg = (x + off, 0.0, x + off, h)
            items.append((0, [seg], None))
            final.append((0, [(seg[0] + 50, seg[1] + 20, seg[2] + 50, seg[3] + 20)], None))
        items.append((1, [(x + 0.5, -1.0, x + w - 0.5, -1.0)], None))          # dimension: deleted
        items.append((1, [(x + 0.5, -1.2, x + 0.5, -0.8)], "315"))             # dimension text: deleted
        x += w
    for k in range(30):                                                          # a section sheet far away
        items.append((2, [(300 + k, 0, 300 + k, 8)], None))
    items.append((2, [(300, -2, 310, -2)], "A-A KESIT"))
    return make_scene(items, layers=["DUVAR", "OLCU", "KESIT"]), make_scene(final, layers=["m-mimari"])


def test_label_then_train(monkeypatch):
    config.ensure_dirs()
    projects = []
    for i in range(8):
        raw, final = _project(i)
        rs, fs = f"{i:02d}" * 32, f"f{i:01d}" * 32
        raw.save(process.scene_path(rs))
        final.save(process.scene_path(fs))
        pid = f"p{i}"
        projects.append({"id": pid, "folder": f"Proje {i}", "files": [],
                         "pairs": [{"raw": rs, "final": fs, "why": "test"}]})
        info = process._label_one(pid, rs, fs)
        assert info["ok"], info
    config.INVENTORY.write_text(json.dumps({"projects": projects}), encoding="utf-8")
    splits = {"p0": "test", "p1": "test", "p2": "val"}
    monkeypatch.setattr(train, "split_of", lambda pid: splits.get(pid, "train"))

    job = jobs.Job(id="t", kind="train_cleanup", title="t", params={})
    res = train.train_job(job, rounds=40)
    report = json.loads((config.MODEL_DIR / res["model"] / "report.json").read_text(encoding="utf-8"))
    assert report["pairs"] == {"train": 5, "val": 1, "test": 2}
    assert report["test"]["items"] > 0 and report["test_in_view"]["items"] > 0
    assert report["test"]["accuracy_at_0.5"] > 0.9
    assert len(list((config.MODEL_DIR / res["model"]).glob("*.png"))) == 2

    # the trained pair of models cleans a drawing it has never seen, in both styles
    from cadai.cleanup import apply
    raw, _ = _project(99)
    for style in (False, True):
        p, decision, _ = apply.decide(raw, config.MODEL_DIR / res["model"], keep_furniture=style)
        assert len(p) == raw.n and set(decision.tolist()) <= {apply.DELETE, apply.KEEP, apply.QUESTION}
