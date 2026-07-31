"""Unit tests for scripts/generate_extra_tiles.py — CSV-bbox footprint source (doc §6.5).

The GEE fetch (s2_bands/se_bands) is not exercised here; these cover the pure
footprint-source logic added for the inference handoff: load ids + bounds from
either CSV schema, build a co-registered EPSG:3857 profile, and write/resume the
8-band stack without needing on-disk RGB tiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from data.extra_channels import (
    DEM_BAND_IDX, N_EXTRA_BANDS, N_EXTRA_BANDS_DEM, S2_BAND_IDX,
)
from scripts.generate_extra_tiles import (
    _load_ids_and_bounds, _needs_work, _profile_from_bounds, _write_bands,
)


def test_load_ids_and_bounds_inference_schema(tmp_path):
    """tile_id + bbox cols (inference grid) -> ids and a {id: bounds} map."""
    csv = tmp_path / "grid.csv"
    pd.DataFrame({"tile_id": ["t0_1", "t2_3"],
                  "minx": [0.0, 100.0], "miny": [0.0, 100.0],
                  "maxx": [10.0, 110.0], "maxy": [10.0, 110.0]}).to_csv(csv, index=False)
    ids, bounds = _load_ids_and_bounds(csv)
    assert ids == ["t0_1", "t2_3"]
    assert bounds is not None
    assert bounds["t0_1"] == (0.0, 0.0, 10.0, 10.0)
    assert bounds["t2_3"] == (100.0, 100.0, 110.0, 110.0)


def test_load_ids_training_schema_has_no_bounds(tmp_path):
    """Legacy Tile_ID CSV without bbox cols -> ids only, bounds is None."""
    csv = tmp_path / "meta.csv"
    pd.DataFrame({"Tile_ID": ["a", "b"], "split": ["train", "val"]}).to_csv(csv, index=False)
    ids, bounds = _load_ids_and_bounds(csv)
    assert ids == ["a", "b"]
    assert bounds is None


def test_profile_from_bounds_coregisters():
    """Profile is EPSG:3857, 512x512, 8-band, with a transform mapping the bbox corners."""
    bounds = (-20037508.34, 9627434.80, -20035062.36, 9629880.79)
    prof = _profile_from_bounds(bounds, N_EXTRA_BANDS, size_px=512)
    assert prof["crs"] == "EPSG:3857"
    assert prof["width"] == prof["height"] == 512
    assert prof["count"] == N_EXTRA_BANDS
    assert prof["dtype"] == "float32"
    t = prof["transform"]
    # pixel (0,0) -> (minx, maxy); pixel (512,512) -> (maxx, miny)
    np.testing.assert_allclose((t.c, t.f), (bounds[0], bounds[3]), atol=1e-6)
    x, y = t * (512, 512)
    np.testing.assert_allclose((x, y), (bounds[2], bounds[1]), atol=1e-3)


def test_write_bands_creates_then_resumes(tmp_path):
    """First write creates the 8-band NaN stack + fills band 0; a tile is 'done' for
    --groups s2 only once all of {0,1,6,7} are non-NaN (resumability contract)."""
    prof = _profile_from_bounds((0.0, 0.0, 5120.0, 5120.0), N_EXTRA_BANDS,
                               size_px=512)
    path = tmp_path / "t0_0.tif"
    assert _needs_work(path, S2_BAND_IDX, N_EXTRA_BANDS)                # missing file

    val = np.full((512, 512), 0.3, dtype="float32")
    _write_bands(path, prof, {0: val}, N_EXTRA_BANDS)                   # NDVI only
    assert _needs_work(path, S2_BAND_IDX, N_EXTRA_BANDS)                # NBR/TCB/TCW still NaN
    with rasterio.open(path) as ds:
        assert ds.count == N_EXTRA_BANDS
        assert ds.crs.to_epsg() == 3857
        np.testing.assert_allclose(ds.read(1), 0.3, atol=1e-6)
        assert np.isnan(ds.read(2)).all()

    _write_bands(path, prof, {1: val, 6: val, 7: val}, N_EXTRA_BANDS)   # fill the rest
    assert not _needs_work(path, S2_BAND_IDX, N_EXTRA_BANDS)


def test_dem_sidecar_is_12_bands_and_leaves_canonical_alone(tmp_path):
    """--groups dem writes a 12-band sidecar. The canonical 8-band width must stay
    'complete' at its own count, or a DEM run would mark every EXTRA/ tile stale."""
    bounds = (0.0, 0.0, 5120.0, 5120.0)
    canonical = tmp_path / "canon.tif"
    sidecar = tmp_path / "side.tif"
    val = np.full((512, 512), 0.3, dtype="float32")

    canon_prof = _profile_from_bounds(bounds, N_EXTRA_BANDS, size_px=512)
    _write_bands(canonical, canon_prof, {b: val for b in S2_BAND_IDX}, N_EXTRA_BANDS)
    assert not _needs_work(canonical, S2_BAND_IDX, N_EXTRA_BANDS)

    dem_prof = _profile_from_bounds(bounds, N_EXTRA_BANDS_DEM, size_px=512)
    assert dem_prof["count"] == N_EXTRA_BANDS_DEM == 12
    assert _needs_work(sidecar, DEM_BAND_IDX, N_EXTRA_BANDS_DEM)   # missing
    _write_bands(sidecar, dem_prof, {b: val for b in DEM_BAND_IDX},
                 N_EXTRA_BANDS_DEM)
    assert not _needs_work(sidecar, DEM_BAND_IDX, N_EXTRA_BANDS_DEM)

    with rasterio.open(sidecar) as ds:
        assert ds.count == 12
        for b in DEM_BAND_IDX:
            np.testing.assert_allclose(ds.read(b + 1), 0.3, atol=1e-6)
        # NDVI (band 0) is unset until --copy-ndvi-from runs; 1-7 stay NaN.
        for b in range(0, 8):
            assert np.isnan(ds.read(b + 1)).all()


def test_needs_work_flags_wrong_band_count(tmp_path):
    """An 8-band file cannot satisfy a 12-band request — bands 8-11 do not exist,
    which is why DEM goes to a sidecar instead of being appended in place."""
    bounds = (0.0, 0.0, 5120.0, 5120.0)
    path = tmp_path / "t.tif"
    val = np.full((512, 512), 1.0, dtype="float32")
    _write_bands(path, _profile_from_bounds(bounds, N_EXTRA_BANDS, size_px=512),
                 {b: val for b in S2_BAND_IDX}, N_EXTRA_BANDS)
    assert _needs_work(path, DEM_BAND_IDX, N_EXTRA_BANDS_DEM)


def test_read_band_roundtrips_ndvi(tmp_path):
    """--copy-ndvi-from reads band 0 verbatim, so the sidecar's NDVI is bit-identical
    to the canonical stack's (no GEE re-query, no drift vs the comparator)."""
    from scripts.generate_extra_tiles import _read_band
    bounds = (0.0, 0.0, 5120.0, 5120.0)
    canonical = tmp_path / "canon.tif"
    rng = np.random.default_rng(0)
    ndvi = rng.standard_normal((512, 512)).astype("float32")
    _write_bands(canonical, _profile_from_bounds(bounds, N_EXTRA_BANDS, size_px=512),
                 {0: ndvi}, N_EXTRA_BANDS)
    np.testing.assert_array_equal(_read_band(canonical, 0), ndvi)
