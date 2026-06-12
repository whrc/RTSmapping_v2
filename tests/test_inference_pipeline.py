"""Tests for the inference pipeline (inference/ + entry scripts).

GPU-free; synthetic RGBA quads on the real zoom-15 mosaic grid so the grid
math is exercised with production constants (just tiny 512px quads written for
2-3 grid cells).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
import yaml
from rasterio.transform import from_bounds as transform_from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.quad_index import (
    GRID_N, QUAD_SIZE_M, RESOLUTION_M, WORLD_MIN, load_quad_index, quad_bounds,
)
from inference.predictor import (
    TTA_PASSES, assert_runtime_matches_package, predict_probs,
)
from inference.tiles import TILE_SIZE_PX, InferenceTileDataset, read_tile
from inference.writer import (
    NODATA_MASK, NODATA_PROB, Manifest, write_binary_mask, write_probability_tile,
)
from scripts.generate_tile_grid import generate_tile_grid
from scripts.merge_predictions import gaussian_center_weights, merge_tiles


# ---------------------------------------------------------------------------
# Fixtures: synthetic quads on the real mosaic grid
# ---------------------------------------------------------------------------

QX, QY = 100, 1500  # arbitrary grid cell in the Arctic latitudes


def _write_quad(path: Path, x: int, y: int, fill: int = 128,
                alpha_hole: bool = False) -> None:
    """4-band RGBA quad for grid cell (x, y) at 512px (tests use small quads
    whose pixel size equals QUAD_SIZE_M/512 — read_tile only relies on bounds
    + transform, not on the production 4096px size)."""
    size = 512
    minx, miny, maxx, maxy = quad_bounds(x, y)
    arr = np.full((4, size, size), fill, dtype=np.uint8)
    arr[3] = 255
    if alpha_hole:
        arr[3, :, : size // 2] = 0  # left half NoData
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=4,
        dtype="uint8", crs="EPSG:3857",
        transform=transform_from_bounds(minx, miny, maxx, maxy, size, size),
    ) as dst:
        dst.write(arr)


@pytest.fixture
def quad_setup(tmp_path):
    """Two horizontally adjacent quads (one with an alpha hole) + index CSV.

    Quads are 512px → 8x coarser than production, so tiles here are read at
    the quads' native resolution by using tile bboxes spanning a full quad.
    """
    paths = {}
    for (x, y), hole in [((QX, QY), False), ((QX + 1, QY), True)]:
        p = tmp_path / f"{x}-{y}_quad_file_format.tif"
        _write_quad(p, x, y, fill=100 + (x - QX) * 50, alpha_hole=hole)
        paths[(x, y)] = p
    rows = []
    for (x, y), p in paths.items():
        minx, miny, maxx, maxy = quad_bounds(x, y)
        rows.append({"quad_id": f"{x}-{y}", "x": x, "y": y,
                     "gcs_path": str(p), "udm2_path": "",
                     "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy})
    index = pd.DataFrame(rows)
    csv = tmp_path / "quad_index.csv"
    index.to_csv(csv, index=False)
    return {"index": index, "csv": csv, "tmp": tmp_path}


# ---------------------------------------------------------------------------
# quad_index
# ---------------------------------------------------------------------------

def test_quad_bounds_match_observed_planet_quad():
    # Real quad 0-1515 observed at left=-20037508.34, bottom=9607828.706.
    minx, miny, _, _ = quad_bounds(0, 1515)
    assert minx == pytest.approx(-20037508.34, abs=0.01)
    assert miny == pytest.approx(9607828.706, abs=0.5)


def test_grid_constants_consistent():
    assert GRID_N * QUAD_SIZE_M == pytest.approx(2 * 20037508.34, rel=1e-9)
    assert QUAD_SIZE_M / 4096 == pytest.approx(RESOLUTION_M)


def test_quad_name_regex_handles_both_delivery_layouts():
    from inference.quad_index import _QUAD_NAME_RE
    m = _QUAD_NAME_RE.search("338-1622_quad_file_format.tif")
    assert m and (m.group(1), m.group(2)) == ("338", "1622")
    # Flat layout with mosaic name embedded (observed in 2025-Q3 column 338).
    m = _QUAD_NAME_RE.search("global_quarterly_2025q3_mosaic_338-1474_quad_file_format.tif")
    assert m and (m.group(1), m.group(2)) == ("338", "1474")
    assert _QUAD_NAME_RE.search("338-1622_ortho_udm2_file_format.tif") is None


def test_load_quad_index_validates_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"quad_id": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_quad_index(bad)


# ---------------------------------------------------------------------------
# tiles: windowed reads, quad straddling, NoData
# ---------------------------------------------------------------------------

def _tile_bbox_in_quad(x: int, y: int, frac: float = 0.25):
    """A tile-sized bbox at the quads' native resolution inside quad (x,y)."""
    minx, miny, maxx, maxy = quad_bounds(x, y)
    res = (maxx - minx) / 512  # test quads are 512px
    ox = minx + (maxx - minx) * frac
    oy = miny + (maxy - miny) * frac
    return (ox, oy, ox + TILE_SIZE_PX * res, oy + TILE_SIZE_PX * res)


def test_read_tile_interior(quad_setup):
    bbox = (lambda b: (b[0], b[1], b[2], b[3]))(quad_bounds(QX, QY))
    rgb, nodata = read_tile(bbox, quad_setup["index"])
    assert rgb.shape == (3, 512, 512)
    assert not nodata.any()
    assert (rgb == 100).all()


def test_read_tile_straddles_quads(quad_setup):
    b0 = quad_bounds(QX, QY)
    res = (b0[2] - b0[0]) / 512
    half = 256 * res
    bbox = (b0[2] - half, b0[1], b0[2] + half, b0[1] + 512 * res)
    rgb, nodata = read_tile(bbox, quad_setup["index"])
    # Left half from quad 1 (fill 100); right half overlaps quad 2's alpha
    # hole (its left half) -> NoData there.
    assert (rgb[:, :, :256] == 100).all()
    assert nodata[:, 256:].all()
    assert not nodata[:, :256].any()


def test_read_tile_outside_coverage_is_all_nodata(quad_setup):
    bbox = quad_bounds(QX + 5, QY + 5)
    _, nodata = read_tile(bbox, quad_setup["index"])
    assert nodata.all()


def test_dataset_normalizes_and_mean_substitutes(quad_setup):
    mean = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    std = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    b = quad_bounds(QX + 1, QY)  # quad with alpha hole on its left half
    tiles = pd.DataFrame([{"tile_id": "t1", "minx": b[0], "miny": b[1],
                           "maxx": b[2], "maxy": b[3]}])
    ds = InferenceTileDataset(tiles, quad_setup["index"], mean, std)
    item = ds[0]
    assert not item["all_nodata"]
    assert item["nodata_mask"][:, :256].all()
    # NoData pixels were mean-substituted -> normalize to exactly 0.
    assert np.allclose(item["image"][:, :, :256], 0.0)
    # Valid pixels: (150 - 100) / 10 = 5.
    assert np.allclose(item["image"][:, :, 256:], 5.0)


def test_dataset_flags_all_nodata(quad_setup):
    b = quad_bounds(QX + 7, QY)
    tiles = pd.DataFrame([{"tile_id": "t1", "minx": b[0], "miny": b[1],
                           "maxx": b[2], "maxy": b[3]}])
    ds = InferenceTileDataset(tiles, quad_setup["index"],
                              np.ones(3, np.float32), np.ones(3, np.float32))
    assert ds[0]["all_nodata"]


def test_dataset_rejects_missing_columns(quad_setup):
    with pytest.raises(ValueError, match="missing columns"):
        InferenceTileDataset(pd.DataFrame({"tile_id": ["a"]}),
                             quad_setup["index"],
                             np.ones(3, np.float32), np.ones(3, np.float32))


# ---------------------------------------------------------------------------
# tile grid
# ---------------------------------------------------------------------------

def test_tile_grid_counts_and_determinism(quad_setup):
    grid = generate_tile_grid(quad_setup["index"], stride_px=344)
    grid2 = generate_tile_grid(quad_setup["index"], stride_px=344)
    pd.testing.assert_frame_equal(grid, grid2)
    assert grid["tile_id"].is_unique
    # Every tile bbox intersects at least one quad.
    idx = quad_setup["index"]
    for _, t in grid.iterrows():
        assert ((idx["minx"] < t["maxx"]) & (idx["maxx"] > t["minx"])
                & (idx["miny"] < t["maxy"]) & (idx["maxy"] > t["miny"])).any()
    # Tile size is exactly 512 px at native resolution.
    assert np.allclose(grid["maxx"] - grid["minx"], TILE_SIZE_PX * RESOLUTION_M)


def test_tile_grid_aoi_filter(quad_setup):
    full = generate_tile_grid(quad_setup["index"], stride_px=344)
    b = quad_bounds(QX, QY)
    aoi = (b[0], b[1], b[0] + (b[2] - b[0]) / 4, b[1] + (b[3] - b[1]) / 4)
    sub = generate_tile_grid(quad_setup["index"], stride_px=344, aoi=aoi)
    assert 0 < len(sub) < len(full)
    # AOI grid is a subset of the full grid (same global alignment).
    assert set(sub["tile_id"]).issubset(set(full["tile_id"]))


# ---------------------------------------------------------------------------
# predictor: TTA + temperature
# ---------------------------------------------------------------------------

class _Bias(torch.nn.Module):
    """Logit = first-channel input (asymmetric -> exposes bad inverse TTA)."""
    def forward(self, x):
        return x[:, :1]


def test_tta_inverse_correctness():
    torch.manual_seed(0)
    images = torch.randn(2, 3, 16, 16)
    base = predict_probs(_Bias(), images, temperature=1.0, tta="none")
    for tta in ("minimal", "standard", "full"):
        out = predict_probs(_Bias(), images, temperature=1.0, tta=tta)
        # _Bias is equivariant under flips/rotations, so averaged inverse-
        # transformed probabilities must equal the identity pass exactly.
        assert torch.allclose(out, base, atol=1e-6), tta


def test_temperature_applied_to_logits_before_sigmoid():
    images = torch.full((1, 3, 4, 4), 2.0)
    p1 = predict_probs(_Bias(), images, temperature=1.0, tta="none")
    p2 = predict_probs(_Bias(), images, temperature=2.0, tta="none")
    assert torch.allclose(p1, torch.sigmoid(torch.tensor(2.0)).expand_as(p1))
    assert torch.allclose(p2, torch.sigmoid(torch.tensor(1.0)).expand_as(p2))


def test_tta_pass_counts():
    assert [len(TTA_PASSES[k]) for k in ("none", "minimal", "standard", "full")] \
        == [1, 2, 4, 8]


def test_predict_probs_rejects_unknown_tta():
    with pytest.raises(ValueError, match="Unknown tta"):
        predict_probs(_Bias(), torch.zeros(1, 3, 4, 4), 1.0, tta="bogus")


def test_runtime_package_mismatch_aborts():
    dep = {"precision": "bf16", "tta": "none", "torch_compile": False,
           "scales": [1.0], "temperature": 1.5, "threshold": 0.6}
    assert_runtime_matches_package({"precision": "bf16", "threshold": None}, dep)
    with pytest.raises(ValueError, match="precision"):
        assert_runtime_matches_package({"precision": "fp16"}, dep)
    with pytest.raises(ValueError, match="tta"):
        assert_runtime_matches_package({"tta": "minimal"}, dep)


# ---------------------------------------------------------------------------
# writer: COGs + manifest
# ---------------------------------------------------------------------------

def test_probability_tile_roundtrip(tmp_path):
    probs = np.random.rand(64, 64).astype(np.float32)
    probs[0, :] = NODATA_PROB
    path = str(tmp_path / "t.tif")
    write_probability_tile(path, probs, (0, 0, 64 * 3.0, 64 * 3.0))
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "float32"
        assert src.nodata == NODATA_PROB
        assert src.crs.to_string() == "EPSG:3857"
        assert np.allclose(src.read(1), probs)


def test_binary_mask_roundtrip(tmp_path):
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    mask[0, :] = NODATA_MASK
    path = str(tmp_path / "m.tif")
    write_binary_mask(path, mask, (0, 0, 32.0, 32.0))
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == NODATA_MASK
        assert (src.read(1) == mask).all()


def test_manifest_resume_skips_completed(tmp_path):
    path = str(tmp_path / "inference_log.json")
    m1 = Manifest(path, {"model_version": "test"}, checkpoint_every=1)
    m1.mark("t1", "done")
    m1.mark("t2", "all_nodata")
    m2 = Manifest(path, {"model_version": "test"})
    assert m2.is_done("t1") and m2.is_done("t2")
    assert not m2.is_done("t3")
    assert m2.counts() == {"n_tiles_processed": 1, "n_tiles_skipped_nodata": 1}
    payload = json.loads(Path(path).read_text())
    assert payload["model_version"] == "test"
    assert payload["tiles"]["t2"] == "all_nodata"


# ---------------------------------------------------------------------------
# merge: Gaussian fusion
# ---------------------------------------------------------------------------

def test_gaussian_weights_peak_center_symmetric_zero_at_edges():
    w = gaussian_center_weights(512, 128.0)
    assert w.shape == (512, 512)
    assert w.max() == pytest.approx(w[255:257, 255:257].max())
    assert np.allclose(w, w[::-1, :]) and np.allclose(w, w[:, ::-1])
    # Edge-zeroed: contributions fade in continuously across stitch seams
    # (seam-gradient artifact found by the tiny-area validation, 2026-06-12).
    assert (w[0, :] == 0).all() and (w[:, 0] == 0).all()
    assert (w[-1, :] == 0).all() and (w[:, -1] == 0).all()
    assert (w[1:-1, 1:-1] > 0).all()


def test_merge_weighted_average_and_nodata(tmp_path):
    # Two fully overlapping 512px tiles with constant probs 0.2 / 0.8 ->
    # weighted mean must be exactly 0.5 everywhere (equal weights cancel).
    res = RESOLUTION_M
    bounds = (0.0, 0.0, 512 * res, 512 * res)
    tiles = pd.DataFrame([
        {"tile_id": "a", "minx": bounds[0], "miny": bounds[1],
         "maxx": bounds[2], "maxy": bounds[3]},
        {"tile_id": "b", "minx": bounds[0], "miny": bounds[1],
         "maxx": bounds[2], "maxy": bounds[3]},
    ])
    pa = np.full((512, 512), 0.2, dtype=np.float32)
    pb = np.full((512, 512), 0.8, dtype=np.float32)
    pb[:, :10] = NODATA_PROB  # NoData strip: only tile a contributes there
    write_probability_tile(str(tmp_path / "a.tif"), pa, bounds)
    write_probability_tile(str(tmp_path / "b.tif"), pb, bounds)

    merged, mb = merge_tiles(tiles, str(tmp_path), sigma_px=128.0)
    assert mb == bounds
    # Interior: equal weights cancel -> exact mean; NoData strip -> tile a only.
    assert np.allclose(merged[1:-1, 10:-1], 0.5, atol=1e-6)
    assert np.allclose(merged[1:-1, 1:10], 0.2, atol=1e-6)
    # Outermost ring has zero weight from every tile (edge-zeroed fusion) -> NoData.
    assert (merged[0, :] == NODATA_PROB).all() and (merged[:, 0] == NODATA_PROB).all()


def test_merge_ignores_missing_tiles(tmp_path):
    res = RESOLUTION_M
    bounds = (0.0, 0.0, 512 * res, 512 * res)
    tiles = pd.DataFrame([
        {"tile_id": "present", "minx": bounds[0], "miny": bounds[1],
         "maxx": bounds[2], "maxy": bounds[3]},
        {"tile_id": "absent", "minx": bounds[0], "miny": bounds[1],
         "maxx": bounds[2], "maxy": bounds[3]},
    ])
    write_probability_tile(str(tmp_path / "present.tif"),
                           np.full((512, 512), 0.3, dtype=np.float32), bounds)
    merged, _ = merge_tiles(tiles, str(tmp_path), sigma_px=128.0)
    assert np.allclose(merged[1:-1, 1:-1], 0.3, atol=1e-6)
