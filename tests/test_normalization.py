"""Unit tests for data/normalization.py — Welford vs numpy reference, save/load roundtrip."""

from __future__ import annotations

import numpy as np

from data.normalization import (
    WelfordStats,
    build_stats_dict,
    load_stats,
    save_stats,
    stats_to_arrays,
)


def test_welford_matches_numpy():
    """Welford online mean/std must match np.mean / np.std to ~1e-6 on random data."""
    rng = np.random.default_rng(0)
    n_channels = 3
    full = rng.standard_normal((n_channels, 1024, 1024)).astype(np.float32) * 37 + 10

    stats = WelfordStats(channel_names=["c0", "c1", "c2"])
    # Stream in chunks to exercise the update path.
    for chunk_idx in range(8):
        chunk = full[:, chunk_idx * 128 : (chunk_idx + 1) * 128, :]
        stats.update(chunk)

    for i in range(n_channels):
        np.testing.assert_allclose(stats.means()[i], full[i].mean(), atol=1e-4)
        np.testing.assert_allclose(stats.stds()[i], full[i].std(), rtol=1e-4)


def test_build_stats_no_extra():
    rgb = WelfordStats(channel_names=["R", "G", "B"])
    rgb.update(np.ones((3, 16, 16)) * 100)
    d = build_stats_dict(rgb, extra=None, dataset_version="2.0", n_tiles_used=1)
    assert "extra" not in d
    assert d["rgb"]["channel_names"] == ["R", "G", "B"]
    np.testing.assert_allclose(d["rgb"]["mean"], [100, 100, 100])


def test_build_stats_with_extra_variable_channels():
    """The key flexibility test: EXTRA channel set is whatever the user named."""
    rgb = WelfordStats(channel_names=["R", "G", "B"])
    rgb.update(np.zeros((3, 8, 8)))
    extra = WelfordStats(channel_names=["ndvi", "custom_signal", "random_band"])
    extra.update(np.ones((3, 8, 8)))
    d = build_stats_dict(rgb, extra=extra, dataset_version="2.0", n_tiles_used=1)
    assert d["extra"]["channel_names"] == ["ndvi", "custom_signal", "random_band"]
    assert len(d["extra"]["mean"]) == 3


def test_save_load_roundtrip(tmp_path):
    rgb = WelfordStats(channel_names=["R", "G", "B"])
    rgb.update(np.ones((3, 4, 4)) * 5)
    d = build_stats_dict(rgb, extra=None, dataset_version="2.0", n_tiles_used=1)
    save_stats(d, tmp_path / "stats.json")

    loaded = load_stats(tmp_path / "stats.json")
    assert loaded["dataset_version"] == "2.0"
    assert loaded["rgb"]["mean"] == [5, 5, 5]


def test_stats_to_arrays_rgb_only():
    d = {"rgb": {"channel_names": ["R", "G", "B"], "mean": [1, 2, 3], "std": [4, 5, 6]}}
    mean, std = stats_to_arrays(d, with_extra=False)
    assert mean.tolist() == [1, 2, 3]
    assert std.tolist() == [4, 5, 6]


def test_stats_to_arrays_with_extra():
    d = {
        "rgb":   {"channel_names": ["R", "G", "B"], "mean": [1, 2, 3], "std": [4, 5, 6]},
        "extra": {"channel_names": ["a", "b"],     "mean": [7, 8],    "std": [9, 10]},
    }
    mean, std = stats_to_arrays(d, with_extra=True)
    assert mean.tolist() == [1, 2, 3, 7, 8]
    assert std.tolist() == [4, 5, 6, 9, 10]
