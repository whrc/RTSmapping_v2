"""Unit tests for scripts/build_unsure_package.py — what goes in the package.

The package is handed to a collaborator who has nothing but ArcGIS Pro, so the
rules worth pinning are the ones that decide *what* they get: exactly the
polygons the merge would call unsure, none of the blinded columns, every chip
listed once, and VRT paths that stay relative so the folder can move. GDAL and
GCS are faked; geopandas is not needed.

Spec: `post-inference/review_campaign.md` §7.1.
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path

import pandas as pd

from scripts import build_unsure_package as bup
from scripts.build_unsure_package import (ORDER_CELL_M, LAYER_COLUMNS,
                                          build_imagery, chip_regions,
                                          fetch_chips, layer_table,
                                          order_polygons, select_unsure,
                                          zip_package)


def _v(rts_id, verdict, injected=False, ts=1.0):
    return {"rts_id": rts_id, "verdict": verdict, "reviewer": "ann",
            "batch_id": "b00000", "injected": injected, "reviewed_at": ts}


def _cell_centre(gx: int, gy: int) -> tuple[float, float]:
    """(lon, lat) of the middle of a 25 km EPSG:3857 grid cell."""
    x, y = (gx + 0.5) * ORDER_CELL_M, (gy + 0.5) * ORDER_CELL_M
    return (math.degrees(x / bup.EARTH_R),
            math.degrees(2 * math.atan(math.exp(y / bup.EARTH_R)) - math.pi / 2))


def _poly(rts_id, gx, gy, tiles="t1_1"):
    lon, lat = _cell_centre(gx, gy)
    return {"rts_id": rts_id, "centroid_lon": lon, "centroid_lat": lat,
            "tile_ids": tiles, "area_m2": 100.0}


# --- selection ------------------------------------------------------------
def test_only_coverage_unsure_polygons_are_selected():
    v = pd.DataFrame([_v(1, "unsure"), _v(2, "rts"), _v(3, "false"),
                      _v(4, "unsure")])
    assert select_unsure(v) == [1, 4]


def test_an_injected_replicate_never_makes_a_polygon_unsure():
    """The coverage verdict is authoritative; a replicate only feeds kappa."""
    v = pd.DataFrame([_v(1, "rts"), _v(1, "unsure", injected=True)])
    assert select_unsure(v) == []


def test_a_later_coverage_verdict_replaces_an_earlier_unsure():
    v = pd.DataFrame([_v(1, "unsure", ts=1.0), _v(1, "rts", ts=2.0)])
    assert select_unsure(v) == []


def test_already_issued_ids_are_excluded():
    v = pd.DataFrame([_v(1, "unsure"), _v(2, "unsure")])
    assert select_unsure(v, exclude={1}) == [2]


def test_an_empty_campaign_selects_nothing():
    assert select_unsure(pd.DataFrame(columns=["rts_id", "verdict", "injected",
                                               "reviewed_at"])) == []


# --- ordering -------------------------------------------------------------
def test_order_walks_the_grid_in_a_serpentine():
    """Row 0 runs west->east, row 1 east->west, so the walk never jumps back."""
    df = pd.DataFrame([_poly(1, 0, 0), _poly(2, 1, 0), _poly(3, 2, 0),
                       _poly(4, 2, 1), _poly(5, 1, 1), _poly(6, 0, 1)])
    out = order_polygons(df.sample(frac=1, random_state=0))
    assert out["rts_id"].tolist() == [1, 2, 3, 4, 5, 6]
    assert out["rev_order"].tolist() == [1, 2, 3, 4, 5, 6]


def test_order_breaks_ties_on_rts_id_whatever_the_input_order():
    df = pd.DataFrame([_poly(9, 0, 0), _poly(3, 0, 0), _poly(5, 0, 0)])
    a = order_polygons(df)["rts_id"].tolist()
    b = order_polygons(df.iloc[::-1])["rts_id"].tolist()
    assert a == b == [3, 5, 9]


# --- chips ----------------------------------------------------------------
def test_a_shared_chip_is_listed_once_in_the_first_polygons_region():
    far = 20  # 500 km away -> a different 200 km region
    df = order_polygons(pd.DataFrame([_poly(1, 0, 0, "t1_1,t2_2"),
                                      _poly(2, far, 0, "t2_2,t3_3")]))
    a = chip_regions(df).set_index("chip")["region"]
    assert sorted(a.index) == ["t1_1", "t2_2", "t3_3"]
    assert a["t2_2"] == a["t1_1"] != a["t3_3"]


def test_a_polygon_with_no_tile_ids_contributes_no_chips():
    df = order_polygons(pd.DataFrame([_poly(1, 0, 0, ""), _poly(2, 1, 0, "t4_4")]))
    assert chip_regions(df)["chip"].tolist() == ["t4_4"]


# --- the layer ------------------------------------------------------------
def test_the_layer_carries_no_blinded_column():
    """max_prob primes the rater; the reviewer and batch hint at a tier."""
    df = order_polygons(pd.DataFrame([_poly(1, 0, 0), _poly(2, 1, 0)]))
    for col in ("max_prob", "conf_class", "batch_id", "reviewer", "mean_prob"):
        df[col] = 0.9
    df["geometry"] = None
    out = layer_table(df, no_imagery={2})
    assert list(out.columns) == ["rts_id", "rev_order", "area_m2", "no_planet",
                                 "geometry"]
    assert out["no_planet"].tolist() == [0, 1]


def test_no_layer_field_is_a_reserved_sql_word():
    """ORDER broke ArcGIS sorting/queries; that is why the field is rev_order."""
    assert "order" not in LAYER_COLUMNS


# --- imagery --------------------------------------------------------------
def test_fetch_copies_missing_chips_and_reports_what_it_cannot_find(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "t1_1.tif").write_bytes(b"one")
    missing = fetch_chips(["t1_1", "t9_9"], str(src), dst, workers=2)
    assert missing == ["t9_9"]
    assert (dst / "t1_1.tif").read_bytes() == b"one"
    assert not list(dst.glob("*.partial"))


def test_fetch_does_not_recopy_a_chip_that_is_already_there(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "t1_1.tif").write_bytes(b"new")
    (dst / "t1_1.tif").write_bytes(b"old")
    assert fetch_chips(["t1_1"], str(src), dst, workers=1) == []
    assert (dst / "t1_1.tif").read_bytes() == b"old"


def test_vrts_use_relative_paths_and_zero_nodata(tmp_path, monkeypatch):
    """Relative members keep the folder relocatable; nodata 0 keeps empty chip
    edges see-through over the Esri basemap."""
    calls = []

    def fake_run(cmd, check, cwd, stdout):
        listing = Path(cwd) / cmd[cmd.index("-input_file_list") + 1]
        calls.append((cmd, cwd, listing.read_text().split("\n")))

    monkeypatch.setattr(bup.subprocess, "run", fake_run)
    (tmp_path / "planet_2025q3").mkdir()
    assign = pd.DataFrame({"chip": ["t2_2", "t1_1", "t3_3"],
                           "region": ["r_0_0", "r_0_0", "r_1_0"]})
    build_imagery(assign, tmp_path)

    (c0, cwd0, m0), (c1, _, m1), (c2, cwd2, m2) = calls
    assert cwd0 == str(tmp_path / "planet_2025q3")
    assert m0 == ["chips/t1_1.tif", "chips/t2_2.tif"] and c0[-1] == "r_0_0.vrt"
    assert m1 == ["chips/t3_3.tif"] and c1[-1] == "r_1_0.vrt"
    assert cwd2 == str(tmp_path) and c2[-1] == "planet_2025q3.vrt"
    assert m2 == ["planet_2025q3/r_0_0.vrt", "planet_2025q3/r_1_0.vrt"]
    for cmd in (c0, c1, c2):
        assert cmd[cmd.index("-srcnodata") + 1] == "0"
        assert cmd[cmd.index("-vrtnodata") + 1] == "0"


def test_the_zip_keeps_the_folder_and_the_paths_relative(tmp_path):
    pkg = tmp_path / "unsure_pkg_2026-10-15"
    (pkg / "planet_2025q3" / "chips").mkdir(parents=True)
    (pkg / "README.md").write_text("hi")
    (pkg / "planet_2025q3" / "chips" / "t1_1.tif").write_bytes(b"x")
    with zipfile.ZipFile(zip_package(pkg)) as zf:
        assert sorted(zf.namelist()) == [
            "unsure_pkg_2026-10-15/README.md",
            "unsure_pkg_2026-10-15/planet_2025q3/chips/t1_1.tif"]
