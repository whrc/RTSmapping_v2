"""Stratified QC sampler (scripts/sample_qc_polygons.py): fixed-seed sample of
n polygons per (conf_class tier × size band) cell, spread across longitude,
with an empty qc_verdict column for the rating pass. GPU-free; synthetic gdf.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from scripts.sample_qc_polygons import sample_qc
from scripts.score_qc_ratings import SIZE_BANDS


def _cands(n_per_cell: int = 60) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(0)
    rows, geoms = [], []
    for cls in ("low", "medium", "high"):
        for _, lo, hi in SIZE_BANDS:
            hi_f = min(hi, 60000.0)
            for _ in range(n_per_cell):
                lon = float(rng.uniform(-160, 160))
                x = lon / 180.0 * 20037508.34
                y = float(rng.uniform(8.4e6, 1.3e7))
                a = float(rng.uniform(lo, hi_f))
                rows.append(dict(rts_id=len(rows) + 1, conf_class=cls,
                                 centroid_lon=lon, area_m2=a))
                geoms.append(box(x, y, x + 100, y + 100))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:3857")


def test_sample_counts_per_cell_and_verdict_column():
    gdf = _cands()
    s = sample_qc(gdf, n_per_cell=20, seed=42)
    assert len(s) == 3 * len(SIZE_BANDS) * 20
    for cls in ("low", "medium", "high"):
        for label, lo, hi in SIZE_BANDS:
            cell = s[(s.conf_class == cls) & (s.area_m2 >= lo) & (s.area_m2 < hi)]
            assert len(cell) == 20, (cls, label, len(cell))
    assert (s["qc_verdict"] == "").all()
    s2 = sample_qc(gdf, n_per_cell=20, seed=42)
    assert list(s["rts_id"]) == list(s2["rts_id"])  # reproducible


def test_sample_spreads_across_longitude():
    gdf = _cands()
    s = sample_qc(gdf, n_per_cell=20, seed=42)
    for cls in ("low", "medium", "high"):
        lons = s[s["conf_class"] == cls]["centroid_lon"]
        bins = np.histogram(lons, bins=6, range=(-160, 160))[0]
        assert (bins > 0).sum() >= 5, bins


def test_sparse_cell_returns_all_its_polygons():
    gdf = _cands(n_per_cell=7)
    s = sample_qc(gdf, n_per_cell=20, seed=42)
    assert len(s) == 3 * len(SIZE_BANDS) * 7  # every cell returned whole
