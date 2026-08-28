"""GEE QC-rater generator (scripts/build_ee_qc_rater.py): embeds the sample
polygons + chip COG URIs into a self-contained Code Editor script. GPU-free.
"""

from __future__ import annotations

import json
import re

import geopandas as gpd
from shapely.geometry import Polygon

from scripts.build_ee_qc_rater import build_rater


def test_rater_embeds_features_and_chip_uris(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"rts_id": [7, 9], "conf_class": ["high", "low"],
         "area_m2": [900.0, 30000.0], "qc_verdict": ["", ""],
         "tile_ids": ["t100_200,t100_201", "t50_60"]},
        geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
                  Polygon([(5000, 0), (5100, 0), (5100, 100), (5000, 100)])],
        crs="EPSG:3857")
    f = tmp_path / "qc_sample.gpkg"
    gdf.to_file(f, driver="GPKG")
    out = tmp_path / "ee_qc_rater.js"
    build_rater(str(f), str(out),
                chip_prefix="gs://rts-arctic-usc1/ee_mirror/2025q3_south/products/qc_chips/rgb_chips/")
    js = out.read_text()
    m = re.search(r"var FEATURES = (\[.*?\]);", js, re.S)
    assert m, "no FEATURES block"
    feats = json.loads(m.group(1))
    assert [ft["id"] for ft in feats] == [7, 9]
    assert feats[0]["chips"] == [
        "gs://rts-arctic-usc1/ee_mirror/2025q3_south/products/qc_chips/rgb_chips/t100_200.tif",
        "gs://rts-arctic-usc1/ee_mirror/2025q3_south/products/qc_chips/rgb_chips/t100_201.tif"]
    assert feats[0]["cls"] == "high"
    # polygon ring must be lon/lat (WGS84), not metres
    assert all(abs(x) <= 180 and abs(y) <= 90 for x, y in feats[0]["ring"])
    for token in ("loadGeoTIFF", "qc_verdict", "Export.table", "unsure"):
        assert token in js, token
