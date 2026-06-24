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

from data.extra_channels import N_EXTRA_BANDS, S2_BAND_IDX
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
    prof = _profile_from_bounds(bounds, size_px=512)
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
    prof = _profile_from_bounds((0.0, 0.0, 5120.0, 5120.0), size_px=512)
    path = tmp_path / "t0_0.tif"
    assert _needs_work(path, S2_BAND_IDX)                # missing file

    val = np.full((512, 512), 0.3, dtype="float32")
    _write_bands(path, prof, {0: val})                   # NDVI only
    assert _needs_work(path, S2_BAND_IDX)                # NBR/TCB/TCW still NaN
    with rasterio.open(path) as ds:
        assert ds.count == N_EXTRA_BANDS
        assert ds.crs.to_epsg() == 3857
        np.testing.assert_allclose(ds.read(1), 0.3, atol=1e-6)
        assert np.isnan(ds.read(2)).all()

    _write_bands(path, prof, {1: val, 6: val, 7: val})   # fill the rest
    assert not _needs_work(path, S2_BAND_IDX)
