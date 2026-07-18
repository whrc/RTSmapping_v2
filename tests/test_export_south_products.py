"""Tiered-inventory packaging (scripts/export_south_products.py): conf_class
assignment from max_prob and the four D1 access forms (flagship / high-only /
centroids / attribute table). GPU-free; tiny synthetic GeoDataFrame.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from scripts.export_south_products import (assign_conf_class,
                                           assign_rts_class, export_products)


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


def test_rts_class_qc_calibrated_rule():
    """2026-07 South QC rule (grid measured on 279 ratings): high_confidence = all
    high tier (incl. <500 m² by monotone tier extension); candidate = medium
    tier under 500 m² (measured 0.53); marginal = everything else."""
    gdf = gpd.GeoDataFrame(
        {"conf_class": ["high", "high", "medium", "medium", "low", "low"],
         "area_m2": [300.0, 50000.0, 300.0, 500.0, 300.0, 50000.0]},
        geometry=gpd.points_from_xy([0] * 6, [0] * 6))
    out = assign_rts_class(gdf)
    assert list(out["rts_class"]) == ["high_confidence", "high_confidence", "candidate",
                                      "marginal", "marginal", "marginal"]


def test_export_products_writes_four_access_forms(tmp_path):
    raw = _cands(tmp_path)
    export_products(str(raw), tmp_path)

    flag = gpd.read_file(tmp_path / "south_rts_candidates.gpkg")
    assert list(flag["conf_class"]) == ["low", "medium", "high"]
    # rts_class: rts_id 3 is high→high_confidence; 2 is medium at 100 m²→candidate
    assert list(flag["rts_class"]) == ["marginal", "candidate", "high_confidence"]
    assert flag.crs.to_epsg() == 3857

    conf = gpd.read_file(tmp_path / "south_rts_high_confidence.gpkg")
    assert len(conf) == 1 and conf.iloc[0]["rts_id"] == 3
    assert not (tmp_path / "south_rts_high.gpkg").exists()

    pts = gpd.read_file(tmp_path / "south_rts_centroids.gpkg")
    assert len(pts) == 3 and (pts.geometry.geom_type == "Point").all()
    # representative points must fall INSIDE their polygons (C-shape centroid
    # would fall outside — the whole point of representative_point)
    for pt, poly in zip(pts.geometry, flag.geometry):
        assert poly.contains(pt)
    assert "conf_class" in pts.columns

    csv = pd.read_csv(tmp_path / "south_rts_attributes.csv")
    assert len(csv) == 3
    assert {"rts_id", "conf_class", "rts_class", "max_prob",
            "area_m2"} <= set(csv.columns)
    assert "geometry" not in csv.columns
    assert (tmp_path / "south_rts_attributes.parquet").exists()


def test_nodata_frac_from_probability_raster(tmp_path):
    """nodata_frac = fraction of NoData (255) pixels in the polygon's padded
    bbox on the probability raster — a soft triage attribute (QC found FPs
    concentrate on high-NoData context; hard veto forbidden, real RTS can
    contain NoData)."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    from scripts.export_south_products import add_nodata_frac

    arr = np.zeros((100, 100), np.uint8)
    arr[:, 50:] = 255                       # right half NoData
    with rasterio.open(tmp_path / "prob.tif", "w", driver="GTiff", width=100,
                       height=100, count=1, dtype="uint8", nodata=255,
                       crs="EPSG:3857",
                       transform=from_bounds(0, 0, 1000, 1000, 100, 100)) as d:
        d.write(arr, 1)

    gdf = gpd.GeoDataFrame(
        {"rts_id": [1, 2]},
        geometry=[Polygon([(100, 100), (300, 100), (300, 300), (100, 300)]),
                  Polygon([(400, 400), (600, 400), (600, 600), (400, 600)])],
        crs="EPSG:3857")
    out = add_nodata_frac(gdf, str(tmp_path / "prob.tif"), pad_frac=0.0)
    assert abs(out.loc[0, "nodata_frac"] - 0.0) < 1e-9   # fully clean half
    assert abs(out.loc[1, "nodata_frac"] - 0.5) < 1e-9   # straddles the edge


def test_export_products_plumbs_nodata_frac(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    arr = np.zeros((50, 50), np.uint8)
    with rasterio.open(tmp_path / "prob.tif", "w", driver="GTiff", width=50,
                       height=50, count=1, dtype="uint8", nodata=255,
                       crs="EPSG:3857",
                       transform=from_bounds(0, 0, 500, 500, 50, 50)) as d:
        d.write(arr, 1)
    raw = _cands(tmp_path)
    export_products(str(raw), tmp_path, prob_raster=str(tmp_path / "prob.tif"))
    flag = gpd.read_file(tmp_path / "south_rts_candidates.gpkg")
    assert "nodata_frac" in flag.columns
    csv = pd.read_csv(tmp_path / "south_rts_attributes.csv")
    assert "nodata_frac" in csv.columns
