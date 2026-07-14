"""Tiered-inventory packaging (scripts/export_south_products.py): conf_class
assignment from max_prob and the four D1 access forms (flagship / high-only /
centroids / attribute table). GPU-free; tiny synthetic GeoDataFrame.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from scripts.export_south_products import assign_conf_class, export_products


def _cands(path: Path) -> Path:
    # a C-shape so centroid-inside is a real check for representative_point
    cshape = Polygon([(0, 0), (30, 0), (30, 10), (10, 10), (10, 20), (30, 20),
                      (30, 30), (0, 30)])
    sq = Polygon([(100, 0), (110, 0), (110, 10), (100, 10)])
    tri = Polygon([(200, 0), (220, 0), (200, 20)])
    gdf = gpd.GeoDataFrame(
        {"rts_id": [1, 2, 3], "max_prob": [0.30, 0.45, 0.65],
         "mean_prob": [0.3, 0.4, 0.6], "area_m2": [700.0, 100.0, 200.0]},
        geometry=[cshape, sq, tri], crs="EPSG:3857")
    f = path / "raw.gpkg"
    gdf.to_file(f, driver="GPKG")
    return f


def test_conf_class_boundaries_are_inclusive():
    gdf = gpd.GeoDataFrame({"max_prob": [0.30, 0.449, 0.45, 0.649, 0.65, 0.9]},
                           geometry=gpd.points_from_xy([0] * 6, [0] * 6))
    out = assign_conf_class(gdf)
    assert list(out["conf_class"]) == ["low", "low", "medium", "medium",
                                       "high", "high"]


def test_export_products_writes_four_access_forms(tmp_path):
    raw = _cands(tmp_path)
    export_products(str(raw), tmp_path)

    flag = gpd.read_file(tmp_path / "south_rts_candidates.gpkg")
    assert list(flag["conf_class"]) == ["low", "medium", "high"]
    assert flag.crs.to_epsg() == 3857

    high = gpd.read_file(tmp_path / "south_rts_high.gpkg")
    assert len(high) == 1 and high.iloc[0]["rts_id"] == 3

    pts = gpd.read_file(tmp_path / "south_rts_centroids.gpkg")
    assert len(pts) == 3 and (pts.geometry.geom_type == "Point").all()
    # representative points must fall INSIDE their polygons (C-shape centroid
    # would fall outside — the whole point of representative_point)
    for pt, poly in zip(pts.geometry, flag.geometry):
        assert poly.contains(pt)
    assert "conf_class" in pts.columns

    csv = pd.read_csv(tmp_path / "south_rts_attributes.csv")
    assert len(csv) == 3
    assert {"rts_id", "conf_class", "max_prob", "area_m2"} <= set(csv.columns)
    assert "geometry" not in csv.columns
    assert (tmp_path / "south_rts_attributes.parquet").exists()
