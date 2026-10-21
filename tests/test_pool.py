from cadai import jobs
from cadai.corpus import process


def square(x):
    return {"ok": True, "x": x * x}


def test_pool_runs_every_task_and_logs():
    job = jobs.Job(id="t", kind="t", title="t", params={})
    results = process._run_pool(job, [(i,) for i in range(5)], square, lambda task, res: f"{task[0]} -> {res['x']}")
    assert sorted(r["x"] for r in results) == [0, 1, 4, 9, 16]
    assert job.done == 5 and job.total == 5
    assert any("paralel süreç" in line["tr"] and "parallel processes" in line["en"] for line in job.lines)


def test_workers_counts_busy_processes(monkeypatch):
    monkeypatch.setattr(process, "memory", lambda: {"free_gb": 3.0})
    assert process.workers(busy=0) == 1
    assert process.workers(busy=2) == 3
    monkeypatch.setattr(process, "memory", lambda: {"free_gb": 0.5})
    assert process.workers(busy=0) == 1            # always at least one


def test_cache_paths_use_short_hash():
    sha = "ab" * 32
    assert process.scene_path(sha).name == "abababababababab.npz"
    assert process.label_path("proj", sha).name == "proj__abababababababab.npz"
    assert process.scene_info(sha) is None and process.label_info("proj", sha) is None


def crowded(x, folder):
    """Runs out of memory when another task runs at the same time (a marker file per running task)."""
    import time
    from pathlib import Path
    mark = Path(folder) / str(x)
    mark.touch()
    time.sleep(0.5)
    others = len(list(Path(folder).iterdir())) - 1
    mark.unlink()
    if others:
        return {"ok": False, "error": "MemoryError: Unable to allocate 272. MiB"}
    return {"ok": True, "x": x}


def test_task_out_of_memory_beside_others_runs_again_alone(tmp_path, monkeypatch):
    """Dataset build: 4 of 247 pairs failed with MemoryError while 4 ran in parallel and were lost."""
    monkeypatch.setattr(process, "memory", lambda: {"free_gb": 20.0})
    job = jobs.Job(id="t", kind="t", title="t", params={})
    results = process._run_pool(job, [(i, str(tmp_path)) for i in range(3)], crowded, lambda t, r: str(r))
    assert sorted(r["x"] for r in results) == [0, 1, 2] and all(r["ok"] for r in results)
    assert any("tek başına" in line["tr"] and "retry alone" in line["en"] for line in job.lines)
