"""Unit tests for data/normalization.py — Welford vs numpy reference, save/load roundtrip."""

from __future__ import annotations

import numpy as np

from data.normalization import (
    WelfordStats,
    apply_norm,
    build_norm_arrays,
    build_stats_dict,
    fill_nodata_with_mean,
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


def test_fill_nodata_inference_convention_chw_perpixel_float():
    """Inference path: CHW float32, per-pixel mask broadcast across channels, no rounding.

    Locks train/inference parity (Rule 3): a filled pixel sits exactly at the mean so it
    z-scores to ~0, matching the training-side substitute_nodata fill.
    """
    rgb = np.full((3, 4, 4), 100.0, dtype=np.float32)
    nodata = np.zeros((4, 4), dtype=bool)
    nodata[0, 0] = True  # one NoData pixel, all channels
    mask = np.broadcast_to(nodata, rgb.shape)  # per-pixel -> per-channel
    means = np.array([50.4, 60.6, 30.2], dtype=np.float32)
    out = fill_nodata_with_mean(rgb, mask, means, channel_axis=0)
    # float raster: exact (unrounded) means at the NoData pixel
    assert np.allclose(out[:, 0, 0], [50.4, 60.6, 30.2])
    # everything else untouched
    assert (out[:, 1:, :] == 100.0).all() and (out[:, 0, 1:] == 100.0).all()


def test_fill_nodata_rounds_for_integer_raster():
    """uint8 raster: mean rounded to dtype so the on-disk raw-value contract holds."""
    rgb = np.zeros((1, 2, 1), dtype=np.uint8)  # HWC, single channel, both pixels NoData
    mask = rgb == 0
    out = fill_nodata_with_mean(rgb, mask, np.array([50.6]), channel_axis=-1)
    assert out.dtype == np.uint8 and out.flatten().tolist() == [51, 51]


# --- per-channel normalization dispatch (data.md §9) -------------------------

def _stats_with_modes():
    """Stats: RGB + 2 EXTRA channels — one zscore+clip, one fixed_scale (SE_PROTO)."""
    return {
        "rgb": {"channel_names": ["R", "G", "B"], "mean": [10, 20, 30], "std": [2, 4, 5]},
        "extra": {
            "channel_names": ["ndvi", "se_proto"],
            "mean": [0.5, 0.0], "std": [0.25, 1.0],
            "mode": ["zscore", "fixed_scale"],
            "clip": [[0.1, 0.9], None],
            "scale": [None, 0.5],
        },
    }


def test_build_norm_arrays_modes_and_clip():
    p = build_norm_arrays(_stats_with_modes(), with_extra=True)
    # RGB (idx 0-2): plain zscore, no clip, not fixed.
    assert not p["is_fixed"][:3].any()
    assert np.isnan(p["clip_lo"][:3]).all()
    # ndvi (idx 3): zscore with clip [0.1, 0.9].
    assert p["is_fixed"][3] == False  # noqa: E712
    assert np.isclose(p["clip_lo"][3], 0.1) and np.isclose(p["clip_hi"][3], 0.9)
    # se_proto (idx 4): fixed_scale, scale 0.5, no clip.
    assert p["is_fixed"][4] and p["scale"][4] == 0.5
    assert np.isnan(p["clip_lo"][4])


def test_apply_norm_zscore_clip_and_fixed_scale():
    p = build_norm_arrays(_stats_with_modes(), with_extra=True)
    img = np.zeros((5, 1, 1), dtype=np.float32)
    img[0, 0, 0] = 14.0    # R: (14-10)/2 = 2.0
    img[3, 0, 0] = 5.0     # ndvi: clip 5.0->0.9, then (0.9-0.5)/0.25 = 1.6
    img[4, 0, 0] = 0.3     # se_proto: fixed_scale 0.3/0.5 = 0.6 (no clip, no z-score)
    out = apply_norm(img, p)
    assert np.isclose(out[0, 0, 0], 2.0)
    assert np.isclose(out[3, 0, 0], 1.6)   # clipped before z-score
    assert np.isclose(out[4, 0, 0], 0.6)   # fixed_scale bypasses z-score


def test_apply_norm_rgb_only_matches_plain_zscore():
    """No EXTRA / no modes ⇒ build_norm_arrays + apply_norm == plain (x-μ)/σ."""
    stats = {"rgb": {"channel_names": ["R", "G", "B"], "mean": [1, 2, 3], "std": [2, 2, 2]}}
    p = build_norm_arrays(stats, with_extra=False)
    img = np.full((3, 2, 2), 5.0, dtype=np.float32)
    out = apply_norm(img, p)
    np.testing.assert_allclose(out, (img - np.array([1, 2, 3])[:, None, None]) / 2.0)


def test_build_stats_dict_records_modes():
    rgb = WelfordStats(channel_names=["R", "G", "B"])
    rgb.update(np.zeros((3, 4, 4)))
    extra = WelfordStats(channel_names=["ndvi", "se_proto"])
    extra.update(np.ones((2, 4, 4)))
    d = build_stats_dict(rgb, extra=extra, dataset_version="1.0", n_tiles_used=1,
                         extra_modes=["zscore", "fixed_scale"],
                         extra_clips=[[0.1, 0.9], None], extra_scales=[None, 0.5])
    assert d["extra"]["mode"] == ["zscore", "fixed_scale"]
    assert d["extra"]["clip"] == [[0.1, 0.9], None]
    assert d["extra"]["scale"] == [None, 0.5]
