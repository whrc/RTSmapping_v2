"""Streaming probability-canvas aggregation (scripts/aggregate_probability.py):
per-cell threshold-free expected RTS area (Σ decoded P × geodesic pixel area)
on a metric EPSG:3857 grid and a 0.5° WGS84 grid, one pass over the
probability_*.tif shards. GPU-free; small synthetic scaled_uint8 shards.
"""

from __future__ import annotations

from math import cos, radians
from pathlib import Path

import numpy as np

import geopandas as gpd
from shapely.geometry import box

from inference.writer import write_probability_tile
from scripts.aggregate_probability import aggregate_shards, write_grids

RES = 10.0  # synthetic pixel size in 3857 metres (easy cell math)


def _shard(d: Path, name: str, arr: np.ndarray, bounds: tuple) -> str:
    path = str(d / name)
    write_probability_tile(path, arr.astype(np.float32), bounds,
                           dtype="scaled_uint8")
    return path


def test_expected_area_near_equator_matches_analytic(tmp_path):
    """100 px at P=0.5 near lat 0 (cos²≈1): expected ≈ 0.5·100·RES²; the
    NoData sea contributes nothing to expected or valid area."""
    a = np.full((50, 50), -1.0)
    a[10:20, 10:20] = 0.5
    p = _shard(tmp_path, "probability_0000_0000.tif", a, (0.0, 0.0, 500.0, 500.0))
    res = aggregate_shards([p], cell_m=1000.0, deg=0.5, window_px=32, workers=2)
    g = res["grid_3857"]
    total = g["expected_m2"].sum()
    assert abs(total - 0.5 * 100 * RES * RES) / (0.5 * 100 * RES * RES) < 0.005
    assert abs(g["valid_m2"].sum() - 100 * RES * RES) / (100 * RES * RES) < 0.005


def test_expected_area_at_60N_applies_cos2_correction(tmp_path):
    """Same blob placed at ~60°N: geodesic pixel area shrinks by cos²(60°)=0.25."""
    y60 = 8399737.89  # 3857 northing of ~60°N
    a = np.full((50, 50), -1.0)
    a[10:20, 10:20] = 0.5
    p = _shard(tmp_path, "probability_0000_0000.tif", a,
               (0.0, y60, 500.0, y60 + 500.0))
    res = aggregate_shards([p], cell_m=1000.0, deg=0.5, window_px=32, workers=2)
    total = res["grid_3857"]["expected_m2"].sum()
    expect = 0.5 * 100 * RES * RES * cos(radians(60.0)) ** 2
    assert abs(total - expect) / expect < 0.02  # lat varies slightly across blob


def test_blobs_land_in_their_own_cells(tmp_path):
    """Two blobs 2 cells apart bin into distinct cells at the right offsets."""
    a = np.full((100, 300), -1.0)
    a[10:20, 10:20] = 1.0     # cell col 0 (x 100..200)
    a[10:20, 210:220] = 1.0   # x 2100..2200 → cell col 2
    p = _shard(tmp_path, "probability_0000_0000.tif", a, (0.0, 0.0, 3000.0, 1000.0))
    res = aggregate_shards([p], cell_m=1000.0, deg=0.5, window_px=64, workers=2)
    g = res["grid_3857"]
    exp = g["expected_m2"]
    occupied = np.argwhere(exp > 0)
    assert len(occupied) == 2
    (r1, c1), (r2, c2) = sorted(map(tuple, occupied))
    assert r1 == r2 and c2 - c1 == 2


def test_half_degree_grid_bins_by_lonlat(tmp_path):
    """A blob at lon≈0.3°, lat≈0.1° lands in 0.5° cell (lon bin of 0.0–0.5,
    lat bin of 0.0–0.5) with the analytic expected area."""
    R = 6378137.0
    x0 = radians(0.3) * R          # ≈ lon 0.3°
    y0 = radians(0.1) * R          # ≈ lat 0.1° (Mercator ≈ linear near 0)
    a = np.full((20, 20), -1.0)
    a[5:15, 5:15] = 1.0
    p = _shard(tmp_path, "probability_0000_0000.tif", a,
               (x0, y0, x0 + 200.0, y0 + 200.0))
    res = aggregate_shards([p], cell_m=1000.0, deg=0.5, window_px=16, workers=2)
    gd = res["grid_deg"]
    exp = gd["expected_m2"]
    occupied = np.argwhere(exp > 0)
    assert len(occupied) == 1
    r, c = occupied[0]
    # verify the occupied cell's lon/lat window contains (0.3, 0.1)
    lon_lo = gd["lon0"] + c * gd["deg"]
    lat_hi = gd["lat_top"] - r * gd["deg"]
    assert lon_lo <= 0.3 < lon_lo + gd["deg"]
    assert lat_hi - gd["deg"] <= 0.1 < lat_hi
    expect = 100 * RES * RES  # P=1.0, cos²≈1 near equator
    assert abs(exp.sum() - expect) / expect < 0.005


def test_multi_shard_sums_are_additive(tmp_path):
    """Two shards contribute to one total; per-grid Σ equals the analytic sum."""
    a = np.full((50, 50), -1.0)
    a[0:10, 0:10] = 0.8
    p1 = _shard(tmp_path, "probability_0000_0000.tif", a, (0.0, 0.0, 500.0, 500.0))
    p2 = _shard(tmp_path, "probability_0000_0001.tif", a,
                (500.0, 0.0, 1000.0, 500.0))
    res = aggregate_shards([p1, p2], cell_m=1000.0, deg=0.5, window_px=32,
                           workers=2)
    expect = 2 * 0.8 * 100 * RES * RES
    total = res["grid_3857"]["expected_m2"].sum()
    assert abs(total - expect) / expect < 0.005
    # 3857 and 0.5° grids must agree on the canvas total (same pixels)
    assert abs(res["grid_deg"]["expected_m2"].sum() - total) / total < 1e-6


def test_write_grids_products_and_per_class_join(tmp_path):
    """write_grids emits cell GPKGs (+ GeoTIFFs) with expected_rts_m2, and
    joins per-conf_class polygon count/area into the cell a centroid falls in."""
    a = np.full((100, 300), -1.0)
    a[:, :] = 0.5
    p = _shard(tmp_path, "probability_0000_0000.tif", a, (0.0, 0.0, 3000.0, 1000.0))
    res = aggregate_shards([p], cell_m=1000.0, deg=0.5, window_px=64, workers=2)
    cands = gpd.GeoDataFrame(
        {"conf_class": ["high", "low"], "area_m2": [500.0, 300.0]},
        geometry=[box(100, 100, 200, 200),      # centroid → cell col 0
                  box(2100, 100, 2200, 200)],   # centroid → cell col 2
        crs="EPSG:3857")
    cands.to_file(tmp_path / "cands.gpkg", driver="GPKG")
    write_grids(res, tmp_path, candidates=str(tmp_path / "cands.gpkg"))

    g = gpd.read_file(tmp_path / "density_1km.gpkg")
    assert g.crs.to_epsg() == 3857
    assert (g["expected_rts_m2"] > 0).all()          # only valid cells written
    hi = g[g["n_high"] > 0]
    lo = g[g["n_low"] > 0]
    assert len(hi) == 1 and len(lo) == 1
    assert hi.iloc[0]["rts_m2_high"] == 500.0
    assert lo.iloc[0]["rts_m2_low"] == 300.0
    assert hi.iloc[0].geometry.bounds[0] == 0.0      # first cell column
    assert lo.iloc[0].geometry.bounds[0] == 2000.0   # third cell column
    assert (tmp_path / "density_1km_expected_m2.tif").exists()
    assert (tmp_path / "density_0.5deg.gpkg").exists()
    assert (tmp_path / "density_0.5deg_expected_m2.tif").exists()
