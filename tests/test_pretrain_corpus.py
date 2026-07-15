"""Unit tests for pretraining/corpus.py — CPU-only, no GCS (spec: pretraining/pretraining.md)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import box as shp_box
from shapely.ops import unary_union

from pretraining import corpus


def _grid(n_x: int, n_y: int, size: float = 2446.0, x0: float = 0.0, y0: float = 0.0):
    """A regular tile grid as the domain CSV has it (tile_id + bounds)."""
    rows = []
    for j in range(n_y):
        for i in range(n_x):
            minx, miny = x0 + i * size, y0 + j * size
            rows.append({"tile_id": f"t{i}_{j}", "minx": minx, "miny": miny,
                         "maxx": minx + size, "maxy": miny + size})
    return pd.DataFrame(rows)


def test_filter_to_s2_footprint_keeps_only_intersecting():
    tiles = _grid(10, 1)                       # 10 tiles along x
    # One S2 cell covering the first ~3 tiles.
    s2 = pd.DataFrame([{"minx": -100, "miny": -100, "maxx": 7000, "maxy": 3000}])
    kept = corpus.filter_to_s2_footprint(tiles, s2)
    assert set(kept["tile_id"]) == {"t0_0", "t1_0", "t2_0"}


def test_drop_excluded_removes_tiles_over_polygons():
    tiles = _grid(5, 1)
    # Exclusion polygon overlapping tile t2 only.
    poly = shp_box(2446 * 2 + 100, 100, 2446 * 2 + 500, 500)
    kept = corpus.drop_excluded(tiles, STRtree([poly]))
    assert "t2_0" not in set(kept["tile_id"])
    assert len(kept) == 4


def test_drop_excluded_empty_tree_is_noop():
    tiles = _grid(3, 1)
    kept = corpus.drop_excluded(tiles, STRtree([]))
    assert len(kept) == len(tiles)


def test_stratified_sample_balances_across_strata():
    # Two far-apart clusters → two strata; ask for 20, expect ~even draw.
    a = _grid(50, 1, x0=0.0)
    b = _grid(50, 1, x0=corpus._LON_SECTOR_M * 3)
    tiles = pd.concat([a, b], ignore_index=True)
    sample = corpus.stratified_sample(tiles, n_target=20, seed=0)
    assert len(sample) == 20
    labels = corpus.stratum_labels(sample)
    counts = pd.Series(labels).value_counts()
    assert len(counts) == 2 and counts.min() >= 8   # roughly balanced


def test_stratified_sample_returns_all_when_target_exceeds_pool():
    tiles = _grid(5, 1)
    sample = corpus.stratified_sample(tiles, n_target=999, seed=1)
    assert len(sample) == len(tiles)


def test_stratified_sample_oversamples_marked_tiles():
    tiles = _grid(100, 1)                       # single stratum
    mask = np.zeros(len(tiles), dtype=bool)
    mask[:10] = True                            # 10 "near-label" tiles
    # Small draw; oversampled tiles are drawn first within the stratum.
    sample = corpus.stratified_sample(tiles, n_target=10, seed=2,
                                      oversample_mask=mask, oversample_factor=2.0)
    picked_marked = mask[[int(t.split("_")[0][1:]) for t in sample["tile_id"]]].sum()
    assert picked_marked >= 8                    # heavily favoured


def _write_geojson(tmp_path, epsg, coords, name="Canadian Low Arctic tundra"):
    gj = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}},
          "features": [{"type": "Feature", "properties": {"ECO_NAME": name},
                        "geometry": {"type": "Polygon", "coordinates": [coords]}}]}
    p = tmp_path / "regions.geojson"
    p.write_text(json.dumps(gj))
    return p


def test_load_exclusion_polygons_reprojects_from_3413(tmp_path):
    # A polar-stereographic (EPSG:3413) box near the pole must come back in EPSG:3857
    # Arctic-latitude coordinates — not left in raw 3413 metres (the leakage bug).
    coords = [[-1_800_000, -2_700_000], [-1_700_000, -2_700_000],
              [-1_700_000, -2_600_000], [-1_800_000, -2_600_000], [-1_800_000, -2_700_000]]
    tree = corpus.load_exclusion_polygons(_write_geojson(tmp_path, 3413, coords), ["Canadian Low Arctic tundra"])
    poly = unary_union(list(tree.geometries))
    assert poly.bounds[1] > 5_000_000, f"not reprojected to 3857 (miny={poly.bounds[1]})"

    # A tile at the reprojected polygon's centroid is dropped; a far-away one is kept.
    cx, cy = poly.centroid.x, poly.centroid.y
    tiles = pd.DataFrame([
        {"tile_id": "hit", "minx": cx - 500, "miny": cy - 500, "maxx": cx + 500, "maxy": cy + 500},
        {"tile_id": "miss", "minx": 0, "miny": 0, "maxx": 100, "maxy": 100},
    ])
    kept = set(corpus.drop_excluded(tiles, tree)["tile_id"])
    assert "hit" not in kept and "miss" in kept


def test_load_exclusion_polygons_no_reproject_when_already_3857(tmp_path):
    coords = [[0, 8_000_000], [100_000, 8_000_000], [100_000, 8_100_000], [0, 8_100_000], [0, 8_000_000]]
    tree = corpus.load_exclusion_polygons(_write_geojson(tmp_path, 3857, coords), ["Canadian Low Arctic tundra"])
    (minx, miny, _, _) = unary_union(list(tree.geometries)).bounds
    assert abs(minx - 0) < 1 and abs(miny - 8_000_000) < 1   # unchanged


def test_quality_ok_rejects_high_nodata_and_empty_ndvi():
    rgb = np.zeros((3, 8, 8), np.float32)
    good_ndvi = np.full((8, 8), 0.3, np.float32)
    assert corpus.quality_ok(rgb, np.zeros((8, 8), bool), good_ndvi)
    # >50% nodata
    nd = np.zeros((8, 8), bool); nd[:, :5] = True
    assert not corpus.quality_ok(rgb, nd, good_ndvi)
    # all-NaN NDVI
    assert not corpus.quality_ok(rgb, np.zeros((8, 8), bool),
                                 np.full((8, 8), np.nan, np.float32))
