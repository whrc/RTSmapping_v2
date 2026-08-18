"""Background prob-COG writer (inference/runner.py::_ProbWriter): writes run in a
thread pool so the per-tile GCS upload never stalls the GPU (the A100 throughput
bottleneck, benchmark 2026-07-07). A tile is marked done only after its write
succeeds — crash-safe resume.
"""

from __future__ import annotations

import numpy as np
import pytest

import inference.runner as runner
from inference.runner import _ProbWriter
from inference.writer import Manifest

BOUNDS = (0.0, 0.0, 512 * 4.777, 512 * 4.777)


def test_writes_all_tiles_and_marks_done_after_success(tmp_path):
    man = Manifest(str(tmp_path / "m.json"), {})
    w = _ProbWriter(man, "scaled_uint8", max_workers=4, max_inflight=8)
    prob = np.full((512, 512), 0.5, np.float32)
    for k in range(20):
        w.submit(str(tmp_path / f"t{k}.tif"), prob.copy(), BOUNDS, f"t{k}")
    w.flush()
    assert all((tmp_path / f"t{k}.tif").exists() for k in range(20))
    assert all(man.is_done(f"t{k}") for k in range(20))
    assert man.counts()["n_tiles_processed"] == 20


def test_backpressure_caps_inflight(tmp_path, monkeypatch):
    """Pending writes never exceed max_inflight (submit blocks until drained)."""
    seen = []
    real = runner.write_probability_tile

    def slow(*a, **k):
        seen.append(1)
        return real(*a, **k)

    monkeypatch.setattr(runner, "write_probability_tile", slow)
    man = Manifest(str(tmp_path / "m.json"), {})
    w = _ProbWriter(man, "scaled_uint8", max_workers=2, max_inflight=4)
    prob = np.full((64, 64), 0.5, np.float32)
    for k in range(30):
        w.submit(str(tmp_path / f"t{k}.tif"), prob.copy(), BOUNDS, f"t{k}")
        assert len(w._pending) <= 4  # invariant held on every submit
    w.flush()
    assert man.counts()["n_tiles_processed"] == 30


def test_write_error_propagates_and_tile_not_marked_done(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("simulated write failure")

    monkeypatch.setattr(runner, "write_probability_tile", boom)
    man = Manifest(str(tmp_path / "m.json"), {})
    w = _ProbWriter(man, "float32", max_workers=2)
    prob = np.zeros((8, 8), np.float32)
    with pytest.raises(OSError):
        for k in range(5):
            w.submit(str(tmp_path / f"t{k}.tif"), prob, BOUNDS, f"t{k}")
        w.flush()
    assert not man.is_done("t0")  # a failed write is never marked done


# --- fork-safe DataLoader + stall watchdog (South-readiness, 2026-07-07) --------

def test_make_loader_uses_forkserver_only_with_workers():
    """Workers must not inherit the parent's gRPC/CUDA threads (the Banks GPU-0
    fork deadlock): num_workers>0 pins forkserver; num_workers=0 stays in-process."""
    ds = [{"x": i} for i in range(4)]
    single = runner._make_loader(ds, batch_size=2, num_workers=0, collate_fn=list)
    multi = runner._make_loader(ds, batch_size=2, num_workers=2, collate_fn=list)
    # DataLoader resolves the "forkserver" string into a ForkServerContext object;
    # None (the default) means in-process for the num_workers=0 case.
    assert single.multiprocessing_context is None
    assert multi.multiprocessing_context._name == "forkserver"
