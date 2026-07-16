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


# --- threshold mode: polygonize probability_*.tif COGs directly (no mask blocks) ---

def _write_prob_supertiles(d: Path, dtype: str = "float32"):
    """Two adjacent probability super-tile COGs mirroring _write_blocks geometry
    (blobs at prob 0.8, elsewhere NoData), named per assemble_region COG shards."""
    a = np.full((100, 100), -1.0, np.float32)
    b = np.full((100, 100), -1.0, np.float32)
    a[70:90, 10:30] = 0.8          # interior blob
    a[50:55, 50:55] = 0.8          # tiny blob
    a[40:60, 90:100] = 0.8         # seam slump, A half
    b[40:60, 0:10] = 0.8           # seam slump, B half
    write_probability_tile(str(d / "probability_0000_0000.tif"), a,
                           (0.0, 0.0, 100.0, 100.0), dtype=dtype)
    write_probability_tile(str(d / "probability_0000_0001.tif"), b,
                           (100.0, 0.0, 200.0, 100.0), dtype=dtype)
    # region prob VRT stand-in for _record stats: one raster spanning both
    prob = np.full((100, 200), -1.0, np.float32)
    prob[70:90, 10:30] = 0.8
    prob[50:55, 50:55] = 0.8
    prob[40:60, 90:110] = 0.8
    write_probability_tile(str(d / "prob.tif"), prob, (0.0, 0.0, 200.0, 100.0),
                           dtype="float32")
    pd.DataFrame([dict(tile_id="t0", minx=0, miny=0, maxx=200, maxy=100)]).to_csv(
        d / "t.csv", index=False)


def test_threshold_mode_matches_mask_mode(tmp_path):
    """threshold=0.65 on probability COGs must reproduce the mask-mode result
    (the delivered mask was cut at 0.65): 2 polygons, seam slump reassembled —
    including across the window grid (window_px=50 splits the seam slump)."""
    _write_prob_supertiles(tmp_path)
    gdf = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"),
                           str(tmp_path / "t.csv"), scales=[1.0],
                           min_blob_px=300, workers=2,
                           threshold=0.65, window_px=50)
    assert len(gdf) == 2
    areas = sorted(gdf["area_m2"])
    assert all(380 <= a <= 420 for a in areas), areas


def test_threshold_mode_scaled_u8_nodata_not_above_threshold(tmp_path):
    """scaled_uint8 NoData (255) is numerically > any scaled threshold and must
    be excluded: only the real blobs polygonize, nothing over the NoData sea."""
    _write_prob_supertiles(tmp_path, dtype="scaled_uint8")
    gdf = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"),
                           str(tmp_path / "t.csv"), scales=[1.0],
                           min_blob_px=300, workers=2,
                           threshold=0.65, window_px=50)
    assert len(gdf) == 2
    assert gdf["area_m2"].sum() < 1000  # NoData sea (20000 px) not polygonized


def test_threshold_mode_lower_threshold_recovers_lower_prob_blob(tmp_path):
    """A prob-0.5 blob is invisible at threshold 0.65 but appears at 0.30."""
    a = np.full((100, 100), -1.0, np.float32)
    a[10:40, 10:40] = 0.5   # 900 px blob at prob 0.5
    write_probability_tile(str(tmp_path / "probability_0000_0000.tif"), a,
                           (0.0, 0.0, 100.0, 100.0), dtype="scaled_uint8")
    write_probability_tile(str(tmp_path / "prob.tif"), a,
                           (0.0, 0.0, 100.0, 100.0), dtype="float32")
    pd.DataFrame([dict(tile_id="t0", minx=0, miny=0, maxx=100, maxy=100)]).to_csv(
        tmp_path / "t.csv", index=False)
    hi = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"),
                          str(tmp_path / "t.csv"), scales=[1.0],
                          min_blob_px=300, workers=2, threshold=0.65)
    lo = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"),
                          str(tmp_path / "t.csv"), scales=[1.0],
                          min_blob_px=300, workers=2, threshold=0.30)
    assert len(hi) == 0
    assert len(lo) == 1
    assert abs(lo["max_prob"].iloc[0] - 0.5) < 1e-3


def test_multi_threshold_area_attributes(tmp_path):
    """area_m2_t45/t65/t80 scale area_m2 by the fraction of in-polygon pixels
    above each threshold: blob is half prob 0.9, half prob 0.5."""
    a = np.full((100, 100), -1.0, np.float32)
    a[10:30, 10:30] = 0.5   # lower half of the blob (rows 10-29)
    a[30:50, 10:30] = 0.9   # upper half (rows 30-49) — one 40x20 = 800 px blob
    write_probability_tile(str(tmp_path / "probability_0000_0000.tif"), a,
                           (0.0, 0.0, 100.0, 100.0), dtype="float32")
    write_probability_tile(str(tmp_path / "prob.tif"), a,
                           (0.0, 0.0, 100.0, 100.0), dtype="float32")
    pd.DataFrame([dict(tile_id="t0", minx=0, miny=0, maxx=100, maxy=100)]).to_csv(
        tmp_path / "t.csv", index=False)
    gdf = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"),
                           str(tmp_path / "t.csv"), scales=[1.0],
                           min_blob_px=300, workers=2, threshold=0.30)
    assert len(gdf) == 1
    r = gdf.iloc[0]
    assert abs(r["area_m2_t45"] - r["area_m2"]) / r["area_m2"] < 0.01       # all >= 0.45
    assert abs(r["area_m2_t65"] - 0.5 * r["area_m2"]) / r["area_m2"] < 0.01  # 0.9 half
    assert abs(r["area_m2_t80"] - 0.5 * r["area_m2"]) / r["area_m2"] < 0.01


# --- geodesic MMU (--min-area-m2), parallel _record, arithmetic tile join ---

def _two_lat_shards(d: Path):
    """Identical 10×10-px (res 10 m) blobs at lat≈0 and ≈60°N: same pixel count,
    geodesic areas ≈ 10,000 vs ≈ 2,500 m² (cos²60°). A pixel-count filter can
    never separate them; a geodesic-m² filter must."""
    y60 = 8399737.89
    for name, y0 in (("probability_0000_0000.tif", 0.0),
                     ("probability_0001_0000.tif", y60)):
        a = np.full((50, 50), -1.0, np.float32)
        a[20:30, 20:30] = 0.9
        write_probability_tile(str(d / name), a, (0.0, y0, 500.0, y0 + 500.0),
                               dtype="scaled_uint8")
    prob = np.full((50, 50), -1.0, np.float32)  # stats VRT stand-in (unused vals)
    write_probability_tile(str(d / "prob.tif"), prob,
                           (0.0, 0.0, 500.0, 8400237.89), dtype="float32")
    pd.DataFrame([dict(tile_id="t0_0", minx=0, miny=0, maxx=500,
                       maxy=8400237.89)]).to_csv(d / "t.csv", index=False)


def test_min_area_m2_is_latitude_invariant(tmp_path):
    _two_lat_shards(tmp_path)
    kw = dict(tile_list=str(tmp_path / "t.csv"), scales=[1.0], min_blob_px=0,
              workers=2, threshold=0.65)
    both = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"),
                            min_area_m2=1000.0, **kw)
    assert len(both) == 2  # floor below both geodesic areas
    one = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"),
                           min_area_m2=5000.0, **kw)
    # 5,000 m² sits between the two geodesic areas: only the low-lat blob stays
    assert len(one) == 1
    assert one.iloc[0]["centroid_lat"] < 1.0
    assert one.iloc[0]["area_m2"] >= 5000.0


def test_parallel_record_matches_serial(tmp_path):
    _write_prob_supertiles(tmp_path)
    kw = dict(tile_list=str(tmp_path / "t.csv"), scales=[1.0], min_blob_px=300,
              threshold=0.65, window_px=50)
    a = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"), workers=1, **kw)
    b = vectorize_region(str(tmp_path), str(tmp_path / "prob.tif"), workers=4, **kw)
    cols = ["area_m2", "mean_prob", "max_prob", "centroid_lat", "centroid_lon",
            "area_m2_t65", "tile_ids"]
    a = a.sort_values("centroid_lon").reset_index(drop=True)
    b = b.sort_values("centroid_lon").reset_index(drop=True)
    for c in cols:
        assert list(a[c]) == list(b[c]), c


def test_arithmetic_tile_join_matches_scan(tmp_path):
    """tile_ids from the stride-grid arithmetic must equal the bbox scan of the
    tile list, on the real t{col}_{row} convention (WORLD_MIN origin)."""
    from scripts.vectorize_region import tile_codes_from_list, tiles_for_bounds
    WORLD_MIN = -20037508.34
    stride_m, tile_m = 344 * 4.777314267158508, 512 * 4.777314267158508
    rows = []
    for c in range(12190, 12200):
        for r in range(7000, 7008):
            if (c + r) % 3:  # holes: grid is not fully populated
                minx = WORLD_MIN + c * stride_m
                miny = WORLD_MIN + r * stride_m
                rows.append(dict(tile_id=f"t{c}_{r}", minx=minx, miny=miny,
                                 maxx=minx + tile_m, maxy=miny + tile_m))
    df = pd.DataFrame(rows)
    codes = tile_codes_from_list(df["tile_id"])
    for bounds in [(WORLD_MIN + 12193.2 * stride_m, WORLD_MIN + 7003.1 * stride_m,
                    WORLD_MIN + 12194.7 * stride_m, WORLD_MIN + 7004.9 * stride_m),
                   (WORLD_MIN + 12190.5 * stride_m, WORLD_MIN + 7000.5 * stride_m,
                    WORLD_MIN + 12190.6 * stride_m, WORLD_MIN + 7000.6 * stride_m)]:
        scan = set(df[(df.minx < bounds[2]) & (df.maxx > bounds[0])
                      & (df.miny < bounds[3]) & (df.maxy > bounds[1])]["tile_id"])
        arith = set(tiles_for_bounds(bounds, codes, stride_m, tile_m))
        assert arith == scan, (bounds, arith ^ scan)
