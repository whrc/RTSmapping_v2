"""Unit tests for data/extra_channels.py — SE derivation math + norm-mode map.

No Earth Engine needed: ``fetch_se_raw`` is monkeypatched, so only the pure
projection/cosine math is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

import data.extra_channels as ec


def test_band_norm_mode():
    assert ec.band_norm_mode(0) == "zscore"        # NDVI
    assert ec.band_norm_mode(2) == "zscore"        # SE_PCA1
    assert ec.band_norm_mode(5) == "fixed_scale"   # SE_PROTO
    assert ec.band_norm_mode(6) == "zscore"        # TCB
    with pytest.raises(ValueError):
        ec.band_norm_mode(99)


def test_se_bands_projection_and_cosine(monkeypatch):
    h = w = 4
    n = ec.SE_N_BANDS
    rng = np.random.default_rng(0)
    se = rng.standard_normal((n, h, w)).astype("float32")
    monkeypatch.setattr(ec, "fetch_se_raw", lambda b, g, y: se)

    comps = rng.standard_normal((3, n)).astype("float32")
    proto = rng.standard_normal(n).astype("float32")
    artifacts = {"pca_components": comps, "pca_mean": np.zeros(n, "float32"),
                 "prototype": proto}

    out = ec.se_bands((0, 0, 1, 1), {}, 2024, artifacts)

    assert set(out) == {2, 3, 4, 5}
    for b in (2, 3, 4, 5):
        assert out[b].shape == (h, w)

    # SE_PROTO is a cosine similarity → [-1, 1].
    p = out[5]
    assert np.all(p >= -1 - 1e-5) and np.all(p <= 1 + 1e-5)

    # SE_PCA1 == projection onto the first global component (zero mean here).
    flat = se.reshape(n, -1).T
    expect_pc1 = (flat @ comps[0]).reshape(h, w)
    np.testing.assert_allclose(out[2], expect_pc1, rtol=1e-4, atol=1e-4)


def test_se_bands_nan_propagates(monkeypatch):
    """A pixel with no SE coverage (NaN) yields NaN SE bands (like the S2 path)."""
    h = w = 3
    n = ec.SE_N_BANDS
    se = np.ones((n, h, w), dtype="float32")
    se[:, 0, 0] = np.nan
    monkeypatch.setattr(ec, "fetch_se_raw", lambda b, g, y: se)
    artifacts = {"pca_components": np.ones((3, n), "float32"),
                 "pca_mean": np.zeros(n, "float32"),
                 "prototype": np.ones(n, "float32")}
    out = ec.se_bands((0, 0, 1, 1), {}, 2024, artifacts)
    assert np.isnan(out[2][0, 0]) and np.isnan(out[5][0, 0])
    assert np.isfinite(out[2][1, 1]) and np.isfinite(out[5][1, 1])
