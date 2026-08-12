"""Vectorization (scripts/vectorize_predictions.py): mask → RTS polygons with the
pixel blob filter and windowed probability pixel-stats.

Guards two things the Tier-2 smoke missed: the deployment `vectorize_min_blob_px`
filter (added 2026-07-06; renamed from `min_blob_size_px` 2026-08-12 to separate it
from the eval-stage `metrics.min_blob_size_px`) and the rasterio-1.4 `Window` rounding fix (the old
`round_offsets("floor")` positional API was removed and raised at runtime).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inference.writer import write_binary_mask, write_probability_tile
from scripts.vectorize_predictions import vectorize


def _write_case(d: Path):
    """300×300 mask with a 3600 px blob (keep) + a 144 px blob (drop @2000)."""
    h = w = 300
    mask = np.zeros((h, w), np.uint8)
    mask[20:80, 20:80] = 1        # 60×60 = 3600 px
    mask[200:212, 200:212] = 1    # 12×12 = 144 px
    prob = np.full((h, w), 0.8, np.float32)
    prob[mask == 0] = -1.0
    res = 10.0
    bounds = (0.0, -h * res, w * res, 0.0)
    write_binary_mask(str(d / "m.tif"),
                      np.where(mask == 1, 1, 255).astype("uint8"), bounds)
    write_probability_tile(str(d / "p.tif"), prob, bounds, dtype="float32")
    pd.DataFrame([dict(tile_id="t0", minx=bounds[0], miny=bounds[1],
                       maxx=bounds[2], maxy=bounds[3])]).to_csv(d / "t.csv",
                                                                index=False)
    return d / "m.tif", d / "p.tif", d / "t.csv"


def test_min_blob_filter_drops_small_and_keeps_large(tmp_path):
    m, p, t = _write_case(tmp_path)
    gdf = vectorize(str(m), str(p), str(t), scales=[1.0], min_blob_px=2000)
    assert len(gdf) == 1                       # 144 px blob filtered out
    assert gdf["rts_id"].tolist() == [1]       # compact ids over kept blobs
    assert abs(gdf["mean_prob"].iloc[0] - 0.8) < 1e-3  # windowed prob read
    assert gdf["max_prob"].iloc[0] <= 1.0
    assert gdf.crs.to_epsg() == 3857


def test_no_filter_keeps_both_blobs(tmp_path):
    m, p, t = _write_case(tmp_path)
    gdf = vectorize(str(m), str(p), str(t), scales=[1.0], min_blob_px=0)
    assert len(gdf) == 2  # both blobs vectorized when the filter is off
