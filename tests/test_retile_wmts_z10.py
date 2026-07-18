"""WMTS z10 re-tile grid math (scripts/retile_wmts_z10.py): the global-grid
tile bounds and shard->candidate-tile mapping that make each output COG
correspond to precisely one WebMercatorQuad z10 tile (the ADC handover
requirement). Pure math - no rasters, no GPU.
"""

from __future__ import annotations

import pytest

from scripts.retile_wmts_z10 import (
    TILE_PX, WM, Z10_TILE_M, Z15_RES, candidate_tiles, z10_tile_bounds,
)


def test_grid_constants_are_consistent():
    assert Z10_TILE_M == pytest.approx(TILE_PX * Z15_RES, abs=1e-6)
    assert 1024 * Z10_TILE_M == pytest.approx(2 * WM, abs=1e-6)


def test_corner_tiles_span_the_world():
    minx, _, _, maxy = z10_tile_bounds(0, 0)
    assert (minx, maxy) == (-WM, WM)
    _, miny, maxx, _ = z10_tile_bounds(1023, 1023)  # z10 matrix is 1024x1024
    assert maxx == pytest.approx(WM)
    assert miny == pytest.approx(-WM)


def test_adjacent_tiles_share_edges_exactly():
    b_left = z10_tile_bounds(100, 200)
    b_right = z10_tile_bounds(101, 200)
    b_below = z10_tile_bounds(100, 201)
    assert b_left[2] == b_right[0]        # maxx == neighbour minx
    assert b_left[1] == b_below[3]        # miny == neighbour maxy


def test_candidate_tiles_cover_a_bbox_and_only_it():
    # a bbox strictly inside tile (512, 300)
    minx, miny, maxx, maxy = z10_tile_bounds(512, 300)
    inner = (minx + 1, miny + 1, maxx - 1, maxy - 1)
    assert candidate_tiles(inner) == [(512, 300)]
    # a bbox spanning exactly 2x2 tiles
    span = (minx + 1, miny - 1, maxx + 1, maxy - 1)
    assert set(candidate_tiles(span)) == {(512, 300), (512, 301),
                                          (513, 300), (513, 301)}


def test_candidate_tiles_clamped_to_matrix_extent():
    world = (-WM - 10, -WM - 10, WM + 10, WM + 10)
    tiles = candidate_tiles(world)
    assert all(0 <= c <= 1023 and 0 <= r <= 1023 for c, r in tiles)
    assert len(tiles) == 1024 * 1024
