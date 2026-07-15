"""Stratified QC sampler (scripts/sample_qc_polygons.py): fixed-seed sample of
n polygons per conf_class band, spread across longitude × area strata, with an
empty qc_verdict column for the ArcGIS rating pass. GPU-free; synthetic gdf.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from scripts.sample_qc_polygons import sample_qc


def _cands(n_per_class: int = 200) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(0)
    rows = []
    geoms = []
    for cls in ("low", "medium", "high"):
        for i in range(n_per_class):
            lon = float(rng.uniform(-160, 160))
            x = lon / 180.0 * 20037508.34
            y = float(rng.uniform(8.4e6, 1.3e7))
            a = float(rng.lognormal(9, 1))
            rows.append(dict(rts_id=len(rows) + 1, conf_class=cls,
                             centroid_lon=lon, area_m2=a))
            geoms.append(box(x, y, x + 100, y + 100))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:3857")


def test_sample_counts_and_verdict_column():
    gdf = _cands()
    s = sample_qc(gdf, n_per_band=50, seed=42)
    assert len(s) == 150
    assert (s["conf_class"].value_counts() == 50).all()
    assert (s["qc_verdict"] == "").all()
    # reproducible
    s2 = sample_qc(gdf, n_per_band=50, seed=42)
    assert list(s["rts_id"]) == list(s2["rts_id"])


def test_sample_spreads_across_longitude():
    gdf = _cands()
    s = sample_qc(gdf, n_per_band=50, seed=42)
    for cls in ("low", "medium", "high"):
        lons = s[s["conf_class"] == cls]["centroid_lon"]
        # samples must span the domain, not cluster: hits in ≥ 5 of 6 bins
        bins = np.histogram(lons, bins=6, range=(-160, 160))[0]
        assert (bins > 0).sum() >= 5, bins


def test_small_band_returns_all_its_polygons():
    gdf = _cands(n_per_class=8)
    s = sample_qc(gdf, n_per_band=50, seed=42)
    assert (s["conf_class"].value_counts() == 8).all()
