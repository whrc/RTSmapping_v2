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
    for b in ec.DEM_BAND_IDX:                      # relev/slope/tpi/curv
        assert ec.band_norm_mode(b) == "zscore"
    with pytest.raises(ValueError):
        ec.band_norm_mode(99)


def test_group_bands_have_no_duplicate_indices():
    """Band indices must be unique across groups — band_norm_mode resolves by index."""
    flat = [b for bands in ec.GROUP_BANDS.values() for b in bands]
    assert len(flat) == len(set(flat))
    assert sorted(flat) == list(range(ec.N_EXTRA_BANDS_DEM))


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


def test_se_bands_zero_vector_is_nan(monkeypatch):
    """No-coverage SE pixels come back as an all-zero vector (not NaN). Both SE_PCA and
    SE_PROTO must be NaN there — otherwise (0 - pca_mean) @ comps.T leaks a nonzero artifact."""
    h = w = 3
    n = ec.SE_N_BANDS
    se = np.ones((n, h, w), dtype="float32")
    se[:, 0, 0] = 0.0                                    # no-coverage zero vector
    monkeypatch.setattr(ec, "fetch_se_raw", lambda b, g, y: se)
    artifacts = {"pca_components": np.ones((3, n), "float32"),
                 "pca_mean": np.full(n, 0.5, "float32"),  # nonzero mean → artifact risk
                 "prototype": np.ones(n, "float32")}
    out = ec.se_bands((0, 0, 1, 1), {}, 2024, artifacts)
    assert np.isnan(out[2][0, 0]) and np.isnan(out[5][0, 0])
    assert np.isfinite(out[2][1, 1]) and np.isfinite(out[5][1, 1])


# --- ArcticDEM terrain derivation ---------------------------------------------
# Earth Engine is never touched: dem_derivatives is pure numpy/scipy, and the two
# grid builders are pure arithmetic on the tile bbox.

HALO = 8  # 2 * ec.DEM_HALO_PX, the total halo dem_derivatives crops


def _ramp_inputs(slope_ratio: float, size: int = 32, coarse_pad: int = 4):
    """Elevation planes for a constant-gradient surface, fine + coarse grids.

    `slope_ratio` is rise per pixel along x, so the true slope depends on the
    ground scale the caller passes to dem_derivatives.
    """
    n = size + HALO
    yy, xx = np.mgrid[0:n, 0:n]
    fine = (xx * slope_ratio).astype("float32")
    c = size // ec.DEM_COARSE_FACTOR + 2 * coarse_pad
    cyy, cxx = np.mgrid[0:c, 0:c]
    coarse = ((cxx - coarse_pad) * slope_ratio * ec.DEM_COARSE_FACTOR).astype("float32")
    return fine, coarse, coarse_pad


def test_dem_slope_is_in_ground_degrees():
    """A 1 m rise per pixel at 1 m/px ground scale is exactly 45 degrees."""
    fine, coarse, pad = _ramp_inputs(slope_ratio=1.0)
    out = ec.dem_derivatives(fine, coarse, ground_scale=1.0,
                             coarse_scale=float(ec.DEM_COARSE_FACTOR),
                             coarse_pad_px=pad)
    interior = out[9][2:-2, 2:-2]
    np.testing.assert_allclose(interior, 45.0, atol=1e-3)


def test_dem_slope_scales_with_ground_scale_not_map_scale():
    """Regression for the Web-Mercator bug: doubling ground metres per pixel
    halves the gradient, so tan(slope) halves. Computing on the map grid instead
    would make slope latitude-dependent (docs/arcticdem_diagnostic.md)."""
    fine, coarse, pad = _ramp_inputs(slope_ratio=1.0)
    kw = dict(coarse_scale=float(ec.DEM_COARSE_FACTOR), coarse_pad_px=pad)
    at_1m = ec.dem_derivatives(fine, coarse, ground_scale=1.0, **kw)[9]
    at_2m = ec.dem_derivatives(fine, coarse, ground_scale=2.0, **kw)[9]
    t1 = np.tan(np.radians(at_1m[2:-2, 2:-2]))
    t2 = np.tan(np.radians(at_2m[2:-2, 2:-2]))
    np.testing.assert_allclose(t2, t1 / 2.0, rtol=1e-5)


def test_dem_curvature_sign_and_planar_zero():
    """Curvature is 0 on a plane and positive in a concave hollow."""
    fine, coarse, pad = _ramp_inputs(slope_ratio=3.0)
    kw = dict(coarse_scale=float(ec.DEM_COARSE_FACTOR), coarse_pad_px=pad)
    planar = ec.dem_derivatives(fine, coarse, ground_scale=1.0, **kw)[11]
    np.testing.assert_allclose(planar[2:-2, 2:-2], 0.0, atol=1e-4)

    # A bowl: elevation grows with distance from centre → concave → positive.
    n = 32 + HALO
    yy, xx = np.mgrid[0:n, 0:n]
    bowl = (((xx - n / 2) ** 2 + (yy - n / 2) ** 2) * 0.01).astype("float32")
    out = ec.dem_derivatives(bowl, coarse, ground_scale=1.0, **kw)
    assert np.all(out[11][2:-2, 2:-2] > 0)


def test_dem_relative_elevation_zero_on_uniform_slope_center():
    """On a plane, elevation minus its own focal mean is ~0 at the tile centre."""
    fine, coarse, pad = _ramp_inputs(slope_ratio=1.0)
    out = ec.dem_derivatives(fine, coarse, ground_scale=1.0,
                             coarse_scale=float(ec.DEM_COARSE_FACTOR),
                             coarse_pad_px=pad)
    h, w = out[8].shape
    centre = out[8][h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2]
    assert np.abs(centre).max() < 2.0 * ec.DEM_COARSE_FACTOR


def test_dem_derivatives_shape_and_bands():
    fine, coarse, pad = _ramp_inputs(slope_ratio=0.5, size=64)
    out = ec.dem_derivatives(fine, coarse, ground_scale=1.5,
                             coarse_scale=1.5 * ec.DEM_COARSE_FACTOR,
                             coarse_pad_px=pad)
    assert set(out) == set(ec.DEM_BAND_IDX)
    for band in ec.DEM_BAND_IDX:
        assert out[band].shape == (64, 64)
        assert out[band].dtype == np.float32


def test_dem_void_stays_nan_and_does_not_poison_the_window():
    """An ArcticDEM void is NaN in the outputs it touches, but the rest of the
    tile keeps finite values — the focal mean is NaN-aware, so one void does not
    wipe out a 300-500 m window."""
    fine, coarse, pad = _ramp_inputs(slope_ratio=1.0, size=32)
    fine = fine.copy()
    fine[16, 16] = np.nan
    coarse = coarse.copy()
    coarse[pad + 1, pad + 1] = np.nan
    out = ec.dem_derivatives(fine, coarse, ground_scale=1.0,
                             coarse_scale=float(ec.DEM_COARSE_FACTOR),
                             coarse_pad_px=pad)
    core_y, core_x = 16 - ec.DEM_HALO_PX, 16 - ec.DEM_HALO_PX
    assert np.isnan(out[9][core_y, core_x])          # slope at the void
    # Far from the void, every band is still finite.
    for band in ec.DEM_BAND_IDX:
        assert np.isfinite(out[band][2, 2])
    # The focal-mean bands keep the vast majority of the tile finite.
    assert np.isfinite(out[8]).mean() > 0.9


def test_nan_uniform_filter_matches_plain_mean_without_nans():
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((16, 16)).astype("float32")
    from scipy.ndimage import uniform_filter
    np.testing.assert_allclose(ec._nan_uniform_filter(arr, 5),
                               uniform_filter(arr, size=5, mode="nearest"),
                               rtol=1e-5, atol=1e-5)


def test_ground_scale_shrinks_with_latitude():
    """Web Mercator scale factor is 1/cos(lat): the same map-unit tile is fewer
    ground metres the further north it is."""
    span = 512 * 4.77

    def bounds_at(lat_deg: float):
        r = 6378137.0
        y = r * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
        return (0.0, y - span / 2, span, y + span / 2)

    gs60 = ec.ground_scale_m(bounds_at(60.0))
    gs74 = ec.ground_scale_m(bounds_at(74.0))
    assert gs60 > gs74
    np.testing.assert_allclose(gs60, 4.77 * np.cos(np.radians(60.0)), rtol=2e-3)
    np.testing.assert_allclose(gs74, 4.77 * np.cos(np.radians(74.0)), rtol=2e-3)


def test_coarse_grid_pads_by_the_relev_radius_in_ground_metres():
    """The coarse grid must cover the tile plus DEM_RELEV_RADIUS_M of *ground*,
    so the focal window is never edge-extended."""
    span = 512 * 4.77
    bounds = (0.0, 8.0e6, span, 8.0e6 + span)
    grid = ec.tile_grid(bounds, size_px=512)
    gs = ec.ground_scale_m(bounds, 512)
    cgrid, cscale, pad_px = ec._coarse_grid(bounds, grid, gs)

    assert cscale == gs * ec.DEM_COARSE_FACTOR
    # Padding, converted back to ground metres, covers the focal radius.
    assert pad_px * cscale >= ec.DEM_RELEV_RADIUS_M
    # The grid is square, centred on the tile, and spans tile + 2 * pad.
    t = cgrid["affineTransform"]
    assert cgrid["dimensions"]["width"] == cgrid["dimensions"]["height"]
    np.testing.assert_allclose(t["translateX"], bounds[0] - pad_px * t["scaleX"])
    np.testing.assert_allclose(t["translateY"], bounds[3] + pad_px * t["scaleX"])


def test_halo_grid_grows_symmetrically():
    bounds = (0.0, 8.0e6, 2442.0, 8.0e6 + 2442.0)
    grid = ec.tile_grid(bounds, size_px=512)
    halo = ec.DEM_HALO_PX
    hgrid = ec._halo_grid(grid, halo)
    assert hgrid["dimensions"]["width"] == 512 + 2 * halo
    assert hgrid["dimensions"]["height"] == 512 + 2 * halo
    assert hgrid["crsCode"] == grid["crsCode"]
    sx = grid["affineTransform"]["scaleX"]
    np.testing.assert_allclose(hgrid["affineTransform"]["translateX"],
                               bounds[0] - halo * sx)
    np.testing.assert_allclose(hgrid["affineTransform"]["translateY"],
                               bounds[3] + halo * sx)
