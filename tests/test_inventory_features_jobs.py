import time

from cadai import jobs
from cadai.cleanup import features
from cadai.corpus import inventory
from conftest import make_scene


def test_classify_files():
    assert inventory.classify("Örnek SAHİBİ(100-1).dwg") == "final"
    assert inventory.classify("Deneme KİŞİ ve Hiss.(200-2).dwg") == "final"
    assert inventory.classify("100-1_ÖRNEK_SAHİBİ_MİMARİ_24.07.2025.dwg") == "raw"
    assert inventory.classify("Drawing1.dwg") == "scratch"


def test_choose_raw_prefers_parcel_then_hint():
    final = {"name": "Başka KİŞİ(300-5).dwg"}
    raws = [{"name": "24 A-B C BLOK.dwg", "size": 9}, {"name": "300-5 MİMARİ.dwg", "size": 1},
            {"name": "300-6 MİMARİ.dwg", "size": 5}]
    best, why = inventory.choose_raw(raws, final)
    assert best["name"] == "300-5 MİMARİ.dwg"
    assert "parsel" in why["tr"] and "parcel" in why["en"]


def test_fold():
    assert inventory.fold("Gülşen ÇAĞLAR ışık") == "GULSEN CAGLAR ISIK"


def test_wall_twin_feature():
    scene = make_scene([
        (0, [(0, 0, 5, 0)], None), (0, [(0, 0.2, 5, 0.2)], None),     # a wall: two faces 20 cm apart
        (1, [(0, 3, 5, 3)], None),                                      # a lone line
    ], layers=["DUVAR", "OLCU"])
    X, names = features.compute(scene)
    twin = X[:, names.index("wall_twin")]
    assert twin[0] > 0.9 and twin[1] > 0.9 and twin[2] == 0
    assert X[0, names.index("layer_wall")] == 1 and X[2, names.index("layer_dim")] == 1


def test_job_runner_records_failures():
    runner = jobs.JobRunner()
    runner.register("ok", "ok", lambda job, n=3: (job.log("hi"), job.progress(n, n), {"n": n})[-1])
    runner.register("bad", "bad", lambda job: 1 / 0)
    a, b = runner.submit("ok", n=2), runner.submit("bad")
    for _ in range(100):
        if a.status in ("done", "failed") and b.status in ("done", "failed"):
            break
        time.sleep(0.05)
    assert a.status == "done" and a.result == {"n": 2} and a.lines
    assert b.status == "failed" and "ZeroDivisionError" in b.error
