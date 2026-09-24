"""RGB QC-chip generation (scripts/build_rgb_chips.py): windows RGB off the
same quad-mosaicking path real inference uses (inference/tiles.read_tile), but
only for the tiles a detected RTS polygon actually references — see
post-inference.md's ArcGIS-QC package plan. GPU-free; synthetic quad + gpkg.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds
from shapely.geometry import Point

from inference.tiles import TILE_SIZE_PX
from scripts.build_rgb_chips import (
    build_tile_bboxes, collect_flagged_tile_ids, context_tile_bboxes,
                                     write_rgb_chip,
)


def _write_gpkg(path: Path, tile_ids_per_row: list[str]) -> None:
    gdf = gpd.GeoDataFrame(
        {"tile_ids": tile_ids_per_row},
        geometry=[Point(0, 0)] * len(tile_ids_per_row),
        crs="EPSG:3857",
    )
    gdf.to_file(path, driver="GPKG")


def test_collect_flagged_tile_ids_dedupes_across_polygons(tmp_path):
    gpkg = tmp_path / "rts.gpkg"
    _write_gpkg(gpkg, ["tA,tB", "tB,tC"])
    assert collect_flagged_tile_ids(str(gpkg)) == {"tA", "tB", "tC"}


def test_collect_flagged_tile_ids_empty_gpkg_returns_empty_set(tmp_path):
    gpkg = tmp_path / "rts.gpkg"
    _write_gpkg(gpkg, [])
    assert collect_flagged_tile_ids(str(gpkg)) == set()


def _write_tile_list(path: Path) -> None:
    pd.DataFrame([
        dict(tile_id="t1", minx=0.0, miny=0.0, maxx=100.0, maxy=100.0),
        dict(tile_id="t2", minx=100.0, miny=0.0, maxx=200.0, maxy=100.0),
        dict(tile_id="t3", minx=200.0, miny=0.0, maxx=300.0, maxy=100.0),
        dict(tile_id="t4", minx=300.0, miny=0.0, maxx=400.0, maxy=100.0),
    ]).to_csv(path, index=False)


def test_build_tile_bboxes_returns_only_requested_ids(tmp_path):
    tile_list = tmp_path / "tiles.csv"
    _write_tile_list(tile_list)
    out = build_tile_bboxes({"t2", "t4"}, str(tile_list))
    assert set(out["tile_id"]) == {"t2", "t4"}
    row = out[out["tile_id"] == "t2"].iloc[0]
    assert (row["minx"], row["miny"], row["maxx"], row["maxy"]) == (100.0, 0.0, 200.0, 100.0)


def test_build_tile_bboxes_raises_on_missing_tile_id(tmp_path):
    tile_list = tmp_path / "tiles.csv"
    _write_tile_list(tile_list)
    with pytest.raises(ValueError, match="t99"):
        build_tile_bboxes({"t1", "t99"}, str(tile_list))


def _write_synthetic_quad(path: Path, bounds: tuple, rgb_value: tuple[int, int, int]) -> None:
    """A tiny 4-band (RGBA) quad COG standing in for a real Planet quad.

    Sized to exactly one tile (TILE_SIZE_PX px covering `bounds`) since
    read_tile does a same-resolution window read at scale=1.0 (no resampling).
    """
    h = w = TILE_SIZE_PX
    transform = transform_from_bounds(*bounds, w, h)
    data = np.zeros((4, h, w), dtype=np.uint8)
    for i, v in enumerate(rgb_value):
        data[i] = v
    data[3] = 255  # fully valid alpha
    profile = dict(driver="GTiff", height=h, width=w, count=4, dtype="uint8",
                   crs="EPSG:3857", transform=transform)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def test_write_rgb_chip_is_georeferenced_uint8_and_matches_quad_values(tmp_path):
    bounds = (0.0, 0.0, TILE_SIZE_PX * 4.77731426, TILE_SIZE_PX * 4.77731426)
    quad_path = tmp_path / "quad.tif"
    _write_synthetic_quad(quad_path, bounds, rgb_value=(10, 20, 30))
    quad_index = pd.DataFrame([dict(
        quad_id="q0", x=0, y=0, gcs_path=str(quad_path),
        minx=bounds[0], miny=bounds[1], maxx=bounds[2], maxy=bounds[3],
    )])

    out_path = tmp_path / "t0.tif"
    write_rgb_chip("t0", bounds, quad_index, str(out_path))

    with rasterio.open(out_path) as src:
        assert src.count == 3
        assert src.dtypes[0] == "uint8"
        assert src.crs.to_epsg() == 3857
        arr = src.read()
        assert tuple(src.bounds) == pytest.approx(bounds)
    assert (arr[0] == 10).all()
    assert (arr[1] == 20).all()
    assert (arr[2] == 30).all()


def test_chip_write_is_atomic(tmp_path, monkeypatch):
    """Resume skips any tile whose file exists, so a half-written chip would be
    skipped forever. A failed write must leave nothing behind."""
    import rasterio as rio

    from scripts import build_rgb_chips as brc

    real_open = rio.open

    def exploding_open(path, mode="r", **kw):
        if mode == "w":
            raise OSError("disk full")
        return real_open(path, mode, **kw)

    monkeypatch.setattr(brc.rasterio, "open", exploding_open)
    out = tmp_path / "t0_0.tif"
    monkeypatch.setattr(brc, "read_tile",
                        lambda *a, **k: (np.zeros((3, 8, 8), "uint8"),
                                         np.zeros((8, 8), bool)))
    with pytest.raises(OSError):
        brc.write_rgb_chip("t0_0", (0.0, 0.0, 10.0, 10.0), None, str(out))
    assert not out.exists()
    assert list(tmp_path.glob("*.partial")) == []


# --- review-context tile selection ----------------------------------------
def _gpkg_with(tmp_path, geoms):
    import geopandas as gpd

    gdf = gpd.GeoDataFrame({"rts_id": list(range(1, len(geoms) + 1))},
                           geometry=geoms, crs="EPSG:3857")
    path = tmp_path / "cand.gpkg"
    gdf.to_file(path, driver="GPKG")
    return str(path)


def test_context_tiles_cover_the_whole_wide_crop(tmp_path):
    """The bug this flag exists for: chipping only the tiles a polygon sits in
    leaves the review app's 1.5 km context view part-blank."""
    from shapely.geometry import box

    from review.crops import crop_bounds

    geom = box(0, 0, 100, 100)
    got = context_tile_bboxes(_gpkg_with(tmp_path, [geom]), stride_px=344)
    _, wide = crop_bounds(geom.bounds)
    # every corner of the wide crop falls inside some returned tile
    for cx, cy in ((wide[0], wide[1]), (wide[2], wide[3]),
                   (wide[0], wide[3]), (wide[2], wide[1])):
        assert ((got.minx <= cx) & (got.maxx >= cx)
                & (got.miny <= cy) & (got.maxy >= cy)).any(), (cx, cy)


def test_context_tiles_outnumber_the_polygons_own_tiles(tmp_path):
    from shapely.geometry import box

    got = context_tile_bboxes(_gpkg_with(tmp_path, [box(0, 0, 100, 100)]),
                              stride_px=344)
    assert len(got) > 4


def test_context_tiles_are_deduped_across_polygons(tmp_path):
    """Neighbouring polygons share context; each tile must be chipped once."""
    from shapely.geometry import box

    got = context_tile_bboxes(
        _gpkg_with(tmp_path, [box(0, 0, 100, 100), box(200, 200, 300, 300)]),
        stride_px=344)
    assert got.tile_id.is_unique


def test_context_tile_bounds_match_the_stride_grid(tmp_path):
    """Bounds are computed, not looked up, so they must reproduce the grid
    generate_tile_grid.py writes — same id, same square."""
    from shapely.geometry import box

    from inference.quad_index import RESOLUTION_M, WORLD_MIN
    from inference.tiles import TILE_SIZE_PX

    got = context_tile_bboxes(_gpkg_with(tmp_path, [box(0, 0, 100, 100)]),
                              stride_px=344).iloc[0]
    c, r = (int(v) for v in got.tile_id[1:].split("_"))
    stride_m, tile_m = 344 * RESOLUTION_M, TILE_SIZE_PX * RESOLUTION_M
    assert got.minx == WORLD_MIN + c * stride_m
    assert got.miny == WORLD_MIN + r * stride_m
    assert round(got.maxx - got.minx, 6) == round(tile_m, 6)
