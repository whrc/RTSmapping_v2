"""Offline QC rating page generator (scripts/build_qc_rating_page.py):
self-contained HTML with pre-rendered chip crops per polygon — instant
navigation (the GEE rater's per-polygon tile loads were the bottleneck),
localStorage autosave, CSV file export. GPU-free; synthetic chips + sample.
"""

from __future__ import annotations

import re

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Polygon

from scripts.build_qc_rating_page import build_page


def _fixture(tmp_path):
    # a 200×200 px RGB "chip mosaic" covering (0,0)-(2000,2000) in 3857
    arr = np.random.default_rng(0).integers(60, 200, (3, 200, 200), np.uint8)
    with rasterio.open(tmp_path / "chips.tif", "w", driver="GTiff", width=200,
                       height=200, count=3, dtype="uint8", crs="EPSG:3857",
                       transform=from_bounds(0, 0, 2000, 2000, 200, 200)) as d:
        d.write(arr)
    gdf = gpd.GeoDataFrame(
        {"rts_id": [3, 8], "conf_class": ["high", "low"],
         "area_m2": [40000.0, 700.0], "qc_verdict": ["", ""]},
        geometry=[Polygon([(400, 400), (700, 400), (700, 700), (400, 700)]),
                  Polygon([(1200, 1200), (1230, 1200), (1230, 1230),
                           (1200, 1230)])],
        crs="EPSG:3857")
    gdf.to_file(tmp_path / "qc_sample.gpkg", driver="GPKG")
    return str(tmp_path / "qc_sample.gpkg"), str(tmp_path / "chips.tif")


def test_page_embeds_images_and_rating_machinery(tmp_path):
    sample, chips = _fixture(tmp_path)
    out = tmp_path / "qc_rater.html"
    build_page(sample, chips, str(out), png_px=64)
    html = out.read_text()
    # one tight + one wide crop per polygon, embedded as data URIs
    assert html.count("data:image/jpeg;base64,") == 4
    m = re.search(r"var ITEMS = (\[.*?\]);", html, re.S)
    assert m
    assert '"id": 3' in m.group(1).replace("'", '"') or '"id":3' in m.group(1)
    for token in ("localStorage", "qc_ratings.csv", "keydown", "unsure",
                  "download"):
        assert token in html, token


def test_tiny_polygon_gets_minimum_context_window(tmp_path):
    """A 30 m polygon must not produce a 30 m crop — the tight view floors at
    ~250 m so the feature has context; the wide view floors at ~1500 m
    (clamped to the mosaic)."""
    sample, chips = _fixture(tmp_path)
    from review.crops import crop_bounds
    tight, wide = crop_bounds((1200, 1200, 1230, 1230))
    assert tight[2] - tight[0] >= 250.0
    assert wide[2] - wide[0] >= 1500.0
    # both stay centred on the polygon
    cx = (1200 + 1230) / 2
    assert abs((tight[0] + tight[2]) / 2 - cx) < 1e-6
    assert abs((wide[0] + wide[2]) / 2 - cx) < 1e-6
