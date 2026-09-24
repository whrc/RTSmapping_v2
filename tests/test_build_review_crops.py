"""Unit tests for the crop-archive builder's resume, chip and neighbour lookups.

The resume test is a regression guard. Between 2026-08 and 2026-09 the live
archive shipped 44% of its wide views part-black: the crops were first rendered
against a 29,850-chip mosaic, the mosaic was then completed to 118,586 chips,
and the re-run skipped every polygon that already had its four JPEGs. Verdicts
were measurably skewed by it (`post-inference/review_campaign.md` §4.1).

The chip index itself is covered in `test_review_crops.py`, next to the crop
geometry it serves.

Spec: `post-inference/review_campaign.md` §4.
"""

from __future__ import annotations

import pandas as pd
import pytest
from shapely.geometry import box

from scripts.build_review_crops import (CROP_SUFFIXES, _already_rendered,
                                        _chips_for, _neighbours_for, _init,
                                        _polygon_index, build_crops)


def _write_crops(out, rts_id: int, suffixes=CROP_SUFFIXES) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for s in suffixes:
        (out / f"{rts_id}_{s}.jpg").write_bytes(b"\xff\xd8\xff stub")


# --- resume ---------------------------------------------------------------
def test_all_four_crops_present_counts_as_rendered(tmp_path):
    _write_crops(tmp_path, 7)
    assert _already_rendered(tmp_path, 7) is True


@pytest.mark.parametrize("missing", CROP_SUFFIXES)
def test_any_missing_crop_means_not_rendered(tmp_path, missing):
    """Two of four on disk is a half-written polygon, not a finished one."""
    _write_crops(tmp_path, 7, [s for s in CROP_SUFFIXES if s != missing])
    assert _already_rendered(tmp_path, 7) is False


def _one_polygon_gpkg(tmp_path):
    import geopandas as gpd

    gdf = gpd.GeoDataFrame({"rts_id": [7]}, geometry=[box(0, 0, 100, 100)],
                           crs="EPSG:3857")
    path = tmp_path / "candidates.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def test_resume_skips_a_polygon_that_already_has_crops(tmp_path):
    """Without --overwrite an existing archive is left exactly as it is —
    the mosaic is never even opened, so a stale crop is never revisited."""
    out = tmp_path / "crops"
    _write_crops(out, 7)
    rendered = build_crops(str(_one_polygon_gpkg(tmp_path)),
                           str(tmp_path / "does_not_exist.vrt"), str(out))
    assert rendered == 0


def test_overwrite_re_renders_an_existing_archive(tmp_path):
    """--overwrite must get past the resume check and reach the mosaic.

    Asserted by the failure it then hits on a bogus VRT path: reaching the
    chip index at all is the behaviour that was missing.
    """
    out = tmp_path / "crops"
    _write_crops(out, 7)
    with pytest.raises(FileNotFoundError):
        build_crops(str(_one_polygon_gpkg(tmp_path)),
                    str(tmp_path / "does_not_exist.vrt"), str(out),
                    overwrite=True)


# --- the two per-process bbox lookups -------------------------------------
@pytest.fixture
def indexes():
    chips = pd.DataFrame({
        "path": ["a.tif", "b.tif"],
        "minx": [0.0, 1000.0], "miny": [0.0, 0.0],
        "maxx": [1000.0, 2000.0], "maxy": [1000.0, 1000.0],
    })
    polys = _polygon_index_from(
        {1: box(100, 100, 200, 200),      # inside the window
         2: box(1500, 100, 1600, 200),    # outside it
         3: box(300, 300, 400, 400)})     # inside; used as the target
    _init(chips, polys)


def _polygon_index_from(geoms: dict) -> pd.DataFrame:
    """`_polygon_index` over a dict of {rts_id: geometry}."""
    import geopandas as gpd

    return _polygon_index(gpd.GeoDataFrame(
        {"rts_id": list(geoms)}, geometry=list(geoms.values()),
        crs="EPSG:3857"))


def test_chips_for_selects_only_overlapping_chips(indexes):
    assert _chips_for((0.0, 0.0, 500.0, 500.0)) == ["a.tif"]
    assert sorted(_chips_for((900.0, 0.0, 1100.0, 500.0))) == ["a.tif", "b.tif"]


def test_neighbours_exclude_the_polygon_being_rated(indexes):
    """The target is drawn in red by the caller; drawing it again in the
    neighbour colour would put a second outline on the same shape."""
    got = _neighbours_for((0.0, 0.0, 1000.0, 1000.0), self_id=3)
    assert len(got) == 1
    assert got[0].bounds == (100.0, 100.0, 200.0, 200.0)


def test_neighbours_outside_the_window_are_not_drawn(indexes):
    got = _neighbours_for((0.0, 0.0, 500.0, 500.0), self_id=3)
    assert [g.bounds for g in got] == [(100.0, 100.0, 200.0, 200.0)]


def test_polygon_index_carries_bounds_and_wkb():
    """What the workers get instead of the GeoDataFrame the parent holds."""
    from shapely import wkb as shapely_wkb

    idx = _polygon_index_from({42: box(5, 6, 7, 8)})
    assert list(idx["rts_id"]) == [42]
    assert (idx["minx"][0], idx["maxy"][0]) == (5.0, 8.0)
    assert shapely_wkb.loads(idx["wkb"][0]).bounds == (5.0, 6.0, 7.0, 8.0)
