"""Smoke tests for training.visualizations.

These tests verify the figure functions don't crash with reasonable inputs
and produce PNG files; they do not verify pixel-level correctness of the
output (matplotlib rendering).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.visualizations import (
    confusion_matrix_pixel,
    pick_preview_tiles_pass1,
    pr_curves_at_ratios,
    prediction_preview_grid,
    probability_histogram,
)


def test_prediction_preview_grid_writes_png(tmp_path):
    rng = np.random.default_rng(42)
    tiles = [
        {
            "tile_id": "tile_0",
            "image": rng.standard_normal((3, 16, 16)).astype(np.float32),
            "label": np.zeros((16, 16), dtype=np.int64),
            "prob": rng.random((16, 16), dtype=np.float32),
        },
        {
            "tile_id": "tile_1",
            "image": rng.standard_normal((3, 16, 16)).astype(np.float32),
            "label": (rng.random((16, 16)) > 0.7).astype(np.int64),
            "prob": rng.random((16, 16), dtype=np.float32),
        },
    ]
    out = tmp_path / "preview.png"
    mean = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    std = np.array([64.0, 64.0, 64.0], dtype=np.float32)
    path = prediction_preview_grid(tiles, mean, std, out)
    assert path.exists()
    assert path.stat().st_size > 0


def test_pr_curves_at_ratios_handles_zero_positives(tmp_path):
    # One ratio has only negatives; plot should still render without exception.
    rng = np.random.default_rng(42)
    data = {
        200: (rng.standard_normal(100).astype(np.float32), rng.integers(0, 2, 100)),
        500: (rng.standard_normal(100).astype(np.float32), np.zeros(100, dtype=np.int64)),
    }
    out = tmp_path / "pr.png"
    path = pr_curves_at_ratios(data, out)
    assert path.exists()


def test_probability_histogram_log_scale_safe(tmp_path):
    out = tmp_path / "hist.png"
    # Degenerate all-zeros input should still produce a figure with log y.
    path = probability_histogram(np.zeros(1000), out)
    assert path.exists()


def test_confusion_matrix_pixel_subsampled(tmp_path):
    out = tmp_path / "cm.png"
    path = confusion_matrix_pixel(tp=10, fp=1000, fn=5, tn=100000, out_path=out, subsample_ratio=500)
    assert path.exists()


def test_pick_preview_tiles_pass1_partitions_positives_and_negatives():
    """The function must return 3+3 disjoint tile IDs from the val split."""
    # Build a synthetic metadata table.
    rows = []
    for i in range(20):
        rows.append({
            "Tile_id": f"tile_{i:03d}",
            "TrainClass": "Positive" if i < 10 else "Negative",
            "Pos_frac": 0.1 * ((i % 10) + 1),
            "centroid_x": float(i * 100),
            "centroid_y": 0.0,
        })
    md = pd.DataFrame(rows).set_index("Tile_id")
    val_ids = list(md.index)

    picked = pick_preview_tiles_pass1(md, val_ids, n_positive=3, n_negative=3, seed=42)
    assert len(picked["positive"]) == 3
    assert len(picked["negative"]) == 3
    assert set(picked["positive"]).isdisjoint(picked["negative"])
    # Positives must come from Positive tiles only.
    for t in picked["positive"]:
        assert md.loc[t, "TrainClass"] == "Positive"
    for t in picked["negative"]:
        assert md.loc[t, "TrainClass"] != "Positive"
