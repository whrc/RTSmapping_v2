"""Parallel region vectorization (scripts/vectorize_region.py): block-parallel
polygonize + cross-seam dissolve must reassemble slumps split across block
boundaries, and apply min_blob AFTER the dissolve (a split slump can be
sub-min_blob in each half but ≥ min_blob combined). GPU-free; two synthetic
adjacent block masks sharing a seam.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inference.writer import write_binary_mask, write_probability_tile
from scripts.vectorize_region import vectorize_region

RES = 1.0  # 1 map-unit per px for easy pixel math


def _write_blocks(d: Path):
    """Block A x[0,100) + Block B x[100,200), y[0,100). Blobs:
      - interior 20×20=400px in A (kept)
      - tiny 5×5=25px in A (dropped)
      - a 20×20=400px slump crossing the seam: 200px in A + 200px in B
        (each half < a 300px min_blob → must survive only via the dissolve).
    """
    a = np.zeros((100, 100), np.uint8)
    b = np.zeros((100, 100), np.uint8)
    # interior blob in A: cols 10-29, rows 70-89
    a[70:90, 10:30] = 1
    # tiny interior blob in A: rows/cols 50-54
    a[50:55, 50:55] = 1
    # seam-crossing slump, rows 40-59: A cols 90-99 (touches right edge),
    # B cols 0-9 (touches left edge)
    a[40:60, 90:100] = 1
    b[40:60, 0:10] = 1

    def mask_arr(x):
        return np.where(x == 1, 1, 255).astype("uint8")

    write_binary_mask(str(d / "mask_0000000_0000000.tif"), mask_arr(a),
                      (0.0, 0.0, 100.0, 100.0))       # block A
    write_binary_mask(str(d / "mask_0000000_0000100.tif"), mask_arr(b),
                      (100.0, 0.0, 200.0, 100.0))     # block B
    # probability COG spanning both blocks (constant 0.8 where mask, else NoData)
    prob = np.full((100, 200), -1.0, np.float32)
    prob[70:90, 10:30] = 0.8
    prob[50:55, 50:55] = 0.8
    prob[40:60, 90:110] = 0.8
    write_probability_tile(str(d / "prob.tif"), prob, (0.0, 0.0, 200.0, 100.0),
                           dtype="float32")
    pd.DataFrame([dict(tile_id="t0", minx=0, miny=0, maxx=200, maxy=100)]).to_csv(
        d / "t.csv", index=False)


def test_seam_split_slump_reassembles_and_survives_min_blob(tmp_path):
    _write_blocks(tmp_path)
    gdf = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"),
                           str(tmp_path / "t.csv"), scales=[1.0],
                           min_blob_px=300, workers=2)
    # interior 400px blob + reassembled 400px seam slump = 2; tiny 25px dropped;
    # the seam slump (200px each half) survives only because min_blob is applied
    # AFTER the dissolve.
    assert len(gdf) == 2
    areas = sorted(gdf["area_m2"])
    # both ≈ 400 map-units² (1 unit/px), the seam one proving no double-count
    assert all(380 <= a <= 420 for a in areas), areas
    assert abs(gdf["mean_prob"].iloc[0] - 0.8) < 1e-3


def test_scaled_uint8_prob_raster_decodes_mean_prob(tmp_path):
    """With a scaled_uint8 prob COG (the South product encoding), mean_prob must
    decode back to the true 0.8 — not read the raw 0-250 pixel value."""
    _write_blocks(tmp_path)
    prob = np.full((100, 200), -1.0, np.float32)
    prob[70:90, 10:30] = 0.8
    prob[50:55, 50:55] = 0.8
    prob[40:60, 90:110] = 0.8
    write_probability_tile(str(tmp_path / "prob_u8.tif"), prob,
                           (0.0, 0.0, 200.0, 100.0), dtype="scaled_uint8")
    gdf = vectorize_region(str(tmp_path), str(tmp_path / "prob_u8.tif"),
                           str(tmp_path / "t.csv"), scales=[1.0],
                           min_blob_px=300, workers=2)
    assert len(gdf) == 2
    assert abs(gdf["mean_prob"].iloc[0] - 0.8) < 1.0 / 250 + 1e-6  # decoded, not 200


def test_min_blob_zero_keeps_all_including_tiny(tmp_path):
    _write_blocks(tmp_path)
    gdf = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"),
                           str(tmp_path / "t.csv"), scales=[1.0],
                           min_blob_px=0, workers=2)
    assert len(gdf) == 3  # interior + tiny + reassembled seam slump
