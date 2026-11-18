"""Tests for scripts/diag_fp_census.py (v3-alpha Phase 0.1 / 0.2).

Synthetic polygons only; no network, no GPU. Positions are built in EPSG:3857 metres
and converted to lon/lat, so the projected-window logic is tested exactly rather than
approximately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from scripts.diag_fp_census import (
    TILE_HALF_M,
    add_bands,
    assign_split,
    confidence_profile,
    rate_table,
    removable_bounds,
    removable_summary,
)

_X0, _Y0 = 1_000_000.0, 11_000_000.0  # a point near 70N in EPSG:3857
_INV = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)


def _poly(status: str, dx: float, dy: float) -> dict:
    """One candidate row at (dx, dy) projected metres from the base point."""
    lon, lat = _INV.transform(_X0 + dx, _Y0 + dy)
    return {"status": status, "centroid_lon": lon, "centroid_lat": lat}


def test_tile_half_is_half_a_512px_web_mercator_tile() -> None:
    assert TILE_HALF_M == pytest.approx(1222.99, abs=0.05)


def test_removable_bounds_flag_neighbours_inside_the_projected_window() -> None:
    df = pd.DataFrame([
        _poly("false_positive", 0, 0),        # A
        _poly("rts", 1_100, 0),               # B inside A's window
        _poly("unsure", 0, 1_200),            # C inside A's window
        _poly("unverified", -1_300, 0),       # D just outside A's window
        _poly("false_positive", 50_000, 0),   # E alone
    ])
    out = removable_bounds(df)
    assert out.loc[0, "tile_has_rts"] and out.loc[0, "tile_has_open"]   # A sees B and C
    assert not out.loc[1, "tile_has_rts"] and out.loc[1, "tile_has_open"]  # B sees C, no other rts
    assert not out.loc[3, "tile_has_open"]                              # D is alone
    assert not out.loc[4, "tile_has_rts"] and not out.loc[4, "tile_has_open"]


def test_the_window_is_square_not_circular() -> None:
    corner = removable_bounds(pd.DataFrame([_poly("false_positive", 0, 0),
                                            _poly("rts", 1_200, 1_200)]))
    assert corner.loc[0, "tile_has_rts"]       # 1,697 m apart but inside the square
    outside = removable_bounds(pd.DataFrame([_poly("false_positive", 0, 0),
                                             _poly("rts", 1_250, 0)]))
    assert not outside.loc[0, "tile_has_rts"]  # 1,250 m > half-tile


def test_removable_bounds_are_ordered_and_separate_unreviewed_neighbours() -> None:
    df = pd.DataFrame([
        _poly("false_positive", 0, 0),                 # A: a verified rts nearby
        _poly("rts", 800, 0),
        _poly("false_positive", 50_000, 0),            # E: alone
        _poly("false_positive", 100_000, 0),           # H: only an unreviewed neighbour
        _poly("unverified", 100_500, 0),
    ])
    s = removable_summary(removable_bounds(df))
    assert s["n_false"] == 3
    assert s["removable_lower"] == pytest.approx(1 / 3)   # only E is certainly removable
    assert s["removable_upper"] == pytest.approx(2 / 3)   # E and H (neighbour unverified)
    assert s["removable_lower"] <= s["removable_upper"]


def test_rate_table_false_rate_excludes_unsure_and_shares_sum_to_one() -> None:
    rows = ([("g1", "rts")] * 3 + [("g1", "false_positive")] + [("g1", "unsure")] * 2
            + [("g2", "rts")] + [("g2", "false_positive")] * 3)
    df = pd.DataFrame(rows, columns=["g", "status"])
    tab = rate_table(df, "g")
    assert tab.loc["g1", "false_rate"] == pytest.approx(0.25)    # 1 / (3 + 1); unsure ignored
    assert tab.loc["g2", "false_rate"] == pytest.approx(0.75)
    assert tab["share_of_all_false"].sum() == pytest.approx(1.0)
    assert tab.loc["g1", "unsure"] == 2


def test_rate_table_shares_use_all_false_even_when_a_group_key_is_missing() -> None:
    df = pd.DataFrame({"g": ["a", "a", None, None], "status": ["false_positive"] * 4})
    tab = rate_table(df, "g")
    assert tab.loc["a", "share_of_all_false"] == pytest.approx(0.5)   # 2 of 4, not 2 of 2


def test_confidence_profile_precision_and_auroc() -> None:
    rows = ([("rts", 0.95, "high", "R", "val_realistic")] * 4
            + [("false_positive", 0.50, "medium", "R", "val_realistic")] * 4
            + [("false_positive", 0.97, "high", "R", "val_realistic")]   # one confident FP
            + [("unsure", 0.99, "high", "R", "val_realistic")])          # ignored
    df = pd.DataFrame(rows, columns=["status", "max_prob", "conf_class", "region", "split"])
    p = confidence_profile(df, cuts=(0.45, 0.90))
    assert p["precision_at_max_prob"]["0.45"]["n"] == 9                    # unsure excluded
    assert p["precision_at_max_prob"]["0.90"]["precision"] == pytest.approx(4 / 5)
    assert p["precision_at_max_prob"]["0.90"]["false_kept"] == 1
    # 4 rts @0.95 vs FPs @0.50 (x4) and @0.97 (x1): 16 of 20 pairs ranked correctly
    assert p["auroc_all"] == pytest.approx(0.8)
    assert p["share_of_false_below_0.9"] == pytest.approx(0.8)
    assert p["auroc_by_val_region"]["R"]["n_rts"] == 4


def test_bands_are_left_closed() -> None:
    df = pd.DataFrame({"max_prob": [0.30, 0.65, 1.0], "area_m2": [0.0, 500.0, 60000.0],
                       "nodata_frac": [0.0, 0.05, 1.0], "abs_lat": [50.0, 60.0, 89.0]})
    out = add_bands(df)
    assert out.loc[1, "max_prob_band"].left == 0.65          # 0.65 opens its band
    # 1.0 must land in the top band, not NaN (pandas rounds the label's right edge to 3 d.p.)
    assert out.loc[2, "max_prob_band"].left == 0.95
    assert out.loc[1, "area_m2_band"].left == 500.0
    assert out.loc[1, "nodata_frac_band"].left == 0.05
    assert out.loc[1, "abs_lat_band"].left == 60.0


def test_assign_split_maps_regions_through_splits_yaml(tmp_path) -> None:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    regions = gpd.GeoDataFrame(
        {"ECO_NAME": ["R_train", "R_val"]},
        geometry=[box(-10, 60, 0, 70), box(0, 60, 10, 70)],
        crs="EPSG:4326",
    )
    reg_path = tmp_path / "regions.geojson"
    regions.to_file(reg_path, driver="GeoJSON")
    splits_path = tmp_path / "splits.yaml"
    splits_path.write_text("train:\n- R_train\nval_realistic:\n- R_val\ntest_realistic: []\n")
    df = pd.DataFrame({"rts_id": [1, 2, 3],
                       "centroid_lon": [-5.0, 5.0, 20.0],
                       "centroid_lat": [65.0, 65.0, 65.0]})
    out = assign_split(df, reg_path, splits_path)
    assert list(out["split"]) == ["train", "val_realistic", "unassigned"]
    assert list(out["region"][:2]) == ["R_train", "R_val"]
    assert np.isnan(out["region"].iloc[2]) or out["region"].iloc[2] is None
