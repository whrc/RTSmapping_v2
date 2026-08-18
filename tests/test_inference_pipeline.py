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
    predict_probs_ensemble,
)
from inference.s2_index import load_s2_index
from inference.tiles import (
    TILE_SIZE_PX, InferenceTileDataset, _BBoxIndex, _spatial_sort,
    read_ndvi_tile, read_tile,
)
from inference.writer import (
    NODATA_MASK, NODATA_PROB, NODATA_SCALED_U8, SCALE_U8, Manifest,
    read_probability_tile, write_binary_mask, write_probability_tile,
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


# Planet's quad filename varies by delivery: the "_file_format" infix comes from
# the order's COG `file_format` tool (not the year), and pre-rename deliveries put
# the bare "<x>-<y>_..." name under an order-UUID directory. All four observed
# regimes must index identically (Heidi Rodenhizer, PR #61 review 2026-08-17).
@pytest.mark.parametrize("name,expect", [
    # raw delivery (pre-rename), COG tool applied
    ("338-1622_quad_file_format.tif", ("338", "1622")),
    # raw delivery (pre-rename), no COG tool
    ("10-1547_quad.tif", ("10", "1547")),
    # 2025 post-rename: mosaic name flattened into the filename
    ("global_quarterly_2025q3_mosaic_338-1474_quad_file_format.tif", ("338", "1474")),
    # 2019/2021 legacy archive
    ("global_quarterly_2019q3_mosaic_10-1547_quad.tif", ("10", "1547")),
    # 2023/2024 legacy archive
    ("global_quarterly_2024q3_mosaic_0-1515_quad.tif", ("0", "1515")),
])
def test_quad_name_regex_matches_every_delivery_regime(name, expect):
    from inference.quad_index import _QUAD_NAME_RE
    m = _QUAD_NAME_RE.search(name)
    assert m and (m.group(1), m.group(2)) == expect


# Every sidecar Planet delivers alongside the quad, across all regimes. None
# carries a "quad" token, so none may be mistaken for imagery.
@pytest.mark.parametrize("name", [
    "338-1622_ortho_udm2_file_format.tif",
    "global_quarterly_2019q3_mosaic_10-1547_ortho_udm.tif",
    "global_quarterly_2023q3_mosaic_1639-1613_ortho_udm2.tif",
    "global_quarterly_2019q3_mosaic_10-1547_provenance_raster.tif",
    "global_quarterly_2025q3_mosaic_338-1474_provenance_raster_file_format.tif",
    "global_quarterly_2019q3_mosaic_10-1547_provenance_vector.zip",
    "global_quarterly_2019q3_mosaic_10-1547_metadata.json",
    "manifest.json",
])
def test_quad_name_regex_rejects_sidecars(name):
    from inference.quad_index import _QUAD_NAME_RE
    assert _QUAD_NAME_RE.search(name) is None


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


def test_is_missing_object_distinguishes_gap_from_transient():
    from rasterio.errors import RasterioIOError

    from inference.tiles import _is_missing_object
    assert _is_missing_object(RasterioIOError(
        "'/vsigs/pdg-planet-data/.../1459-1437_quad_file_format.tif' does not exist "
        "in the file system, and is not recognized as a supported dataset name."))
    assert _is_missing_object(RasterioIOError("file.tif: No such file or directory"))
    assert not _is_missing_object(RasterioIOError("HTTP response code: 503"))  # transient → retry


def test_read_tile_missing_quad_degrades_to_nodata(tmp_path):
    """A quad listed in the index but absent from the bucket must yield NoData for
    its footprint (§5.3), not crash — the pan-Arctic gap that stalled the South run."""
    minx, miny, maxx, maxy = quad_bounds(QX, QY)
    idx = pd.DataFrame([{"quad_id": f"{QX}-{QY}", "x": QX, "y": QY,
                         "gcs_path": str(tmp_path / "absent_quad_file_format.tif"),
                         "udm2_path": "", "minx": minx, "miny": miny,
                         "maxx": maxx, "maxy": maxy}])
    rgb, nodata = read_tile((minx, miny, maxx, maxy), idx)  # must not raise
    assert nodata.all() and (rgb == 0).all()


def test_read_ndvi_missing_cell_degrades_to_nan(tmp_path):
    """Absent S2 cell → NDVI stays NaN for its footprint, no crash."""
    minx, miny, maxx, maxy = quad_bounds(QX, QY)
    idx = pd.DataFrame([{"gcs_path": str(tmp_path / "absent.tif"),
                         "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}])
    ndvi = read_ndvi_tile((minx, miny, maxx, maxy), idx)  # must not raise
    assert np.isnan(ndvi).all()


def test_read_tile_scale05_expands_fov(quad_setup):
    # bbox covering the WHOLE quad at scale 0.5 with tile_size 256: the quad's
    # 512 native px decimate to 256 -> uniform fill survives bilinear.
    bbox = quad_bounds(QX, QY)
    rgb, nodata = read_tile(bbox, quad_setup["index"], tile_size_px=256, scale=0.5)
    assert rgb.shape == (3, 256, 256)
    assert not nodata.any()
    assert (rgb == 100).all()


def test_read_tile_scale05_nodata_stays_crisp(quad_setup):
    # Quad with alpha hole on its left half: nearest-resampled alpha keeps the
    # NoData boundary exact (no bilinear blending of validity).
    bbox = quad_bounds(QX + 1, QY)
    rgb, nodata = read_tile(bbox, quad_setup["index"], tile_size_px=256, scale=0.5)
    assert nodata[:, :128].all()
    assert not nodata[:, 128:].any()
    assert (rgb[:, :, 128:] == 150).all()


def _rgb_stats(mean, std):
    """Minimal normalization_stats.json dict (RGB-only) for the dataset."""
    return {"rgb": {"channel_names": ["R", "G", "B"],
                    "mean": list(map(float, mean)), "std": list(map(float, std))}}


def test_dataset_normalizes_and_mean_substitutes(quad_setup):
    stats = _rgb_stats([100.0, 100.0, 100.0], [10.0, 10.0, 10.0])
    b = quad_bounds(QX + 1, QY)  # quad with alpha hole on its left half
    tiles = pd.DataFrame([{"tile_id": "t1", "minx": b[0], "miny": b[1],
                           "maxx": b[2], "maxy": b[3]}])
    ds = InferenceTileDataset(tiles, quad_setup["index"], stats)
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
                              _rgb_stats([1, 1, 1], [1, 1, 1]))
    assert ds[0]["all_nodata"]


def test_dataset_rejects_missing_columns(quad_setup):
    with pytest.raises(ValueError, match="missing columns"):
        InferenceTileDataset(pd.DataFrame({"tile_id": ["a"]}),
                             quad_setup["index"], _rgb_stats([1, 1, 1], [1, 1, 1]))


# ---------------------------------------------------------------------------
# tiles: §11.3 quad-cache + spatial index (must be bit-identical to the mask)
# ---------------------------------------------------------------------------

def test_bbox_index_matches_boolean_mask(quad_setup):
    idx = quad_setup["index"]
    bi = _BBoxIndex(idx)
    b0 = quad_bounds(QX, QY)
    res = (b0[2] - b0[0]) / 512
    bboxes = [
        b0,                                                     # interior of quad 0
        (b0[2] - 256 * res, b0[1], b0[2] + 256 * res, b0[1] + 512 * res),  # straddle
        quad_bounds(QX + 5, QY + 5),                            # outside coverage
    ]
    for bbox in bboxes:
        minx, miny, maxx, maxy = bbox
        mask = idx[(idx["minx"] < maxx) & (idx["maxx"] > minx)
                   & (idx["miny"] < maxy) & (idx["maxy"] > miny)]
        pd.testing.assert_frame_equal(bi.hits(bbox).reset_index(drop=True),
                                      mask.reset_index(drop=True))


def test_read_tile_hits_path_identical_to_mask(quad_setup):
    # The spatial-index path (hits=) must yield byte-identical output to the
    # legacy full-scan mask path — caching/indexing changes throughput, not pixels.
    idx = quad_setup["index"]
    b0 = quad_bounds(QX, QY)
    res = (b0[2] - b0[0]) / 512
    bbox = (b0[2] - 256 * res, b0[1], b0[2] + 256 * res, b0[1] + 512 * res)  # straddle
    rgb_m, nd_m = read_tile(bbox, idx)
    rgb_h, nd_h = read_tile(bbox, idx, hits=_BBoxIndex(idx).hits(bbox))
    assert np.array_equal(rgb_m, rgb_h)
    assert np.array_equal(nd_m, nd_h)


def test_open_dataset_cache_reuses_handle(quad_setup, monkeypatch):
    from inference import tiles as tiles_mod
    tiles_mod._DATASET_CACHE.clear()
    calls: list[str] = []
    real_open = rasterio.open

    def counting_open(path, *a, **k):
        calls.append(str(path))
        return real_open(path, *a, **k)

    monkeypatch.setattr(tiles_mod.rasterio, "open", counting_open)
    bbox = quad_bounds(QX, QY)  # interior → intersects only quad 0
    rgb1, _ = read_tile(bbox, quad_setup["index"])
    rgb2, _ = read_tile(bbox, quad_setup["index"])
    quad0_path = str(quad_setup["index"].iloc[0]["gcs_path"])
    assert calls.count(quad0_path) == 1  # opened once, second read served from cache
    assert np.array_equal(rgb1, rgb2)
    tiles_mod._DATASET_CACHE.clear()


def test_spatial_sort_permutes_without_dropping_tiles(quad_setup):
    from inference.quad_index import QUAD_SIZE_M as QM, WORLD_MIN as WM
    grid = generate_tile_grid(quad_setup["index"], stride_px=344)
    srt = _spatial_sort(grid)
    assert set(srt["tile_id"]) == set(grid["tile_id"]) and len(srt) == len(grid)
    # Tiles of the same quad cell are contiguous (the cache-locality property):
    # the (row, col) cell key is non-decreasing down the sorted order.
    qy = ((srt["miny"] - WM) // QM).astype("int64").to_numpy()
    qx = ((srt["minx"] - WM) // QM).astype("int64").to_numpy()
    key = qy * (1 << 32) + qx
    assert (np.diff(key) >= 0).all()


# ---------------------------------------------------------------------------
# tiles: NDVI from S2 composites (EXTRA=ndvi)
# ---------------------------------------------------------------------------

def _write_s2_composite(path: Path, x: int, y: int, b4: float, b8: float,
                        size: int = 256, zero_left: bool = False) -> None:
    """4-band S2 composite COG (export order B4,B3,B2,B8) over quad cell (x,y).

    Coarser grid (256px over the quad cell) than the Planet tile so read_ndvi_tile
    must resample (10m→tile grid analogue). zero_left mimics a no-coverage gap
    (B4=B8=0 → NDVI div-by-zero → NaN)."""
    minx, miny, maxx, maxy = quad_bounds(x, y)
    arr = np.zeros((4, size, size), dtype=np.float32)
    arr[0] = b4   # B4 (red)
    arr[3] = b8   # B8 (NIR)
    if zero_left:
        arr[:, :, : size // 2] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", width=size, height=size, count=4,
        dtype="float32", crs="EPSG:3857",
        transform=transform_from_bounds(minx, miny, maxx, maxy, size, size),
    ) as dst:
        dst.write(arr)


@pytest.fixture
def s2_setup(tmp_path):
    """One S2 composite cell over quad (QX,QY) with B4=2000,B8=6000 (NDVI=0.5),
    left half zeroed (no-coverage) + the index CSV."""
    p = tmp_path / "E0010_N0700.tif"
    _write_s2_composite(p, QX, QY, b4=2000.0, b8=6000.0, zero_left=True)
    minx, miny, maxx, maxy = quad_bounds(QX, QY)
    index = pd.DataFrame([{"cell_id": "E0010_N0700", "gcs_path": str(p),
                           "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}])
    csv = tmp_path / "s2_index.csv"
    index.to_csv(csv, index=False)
    return {"index": index, "csv": csv, "tmp": tmp_path}


def test_load_s2_index_validates_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"cell_id": ["a"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_s2_index(bad)


def test_read_ndvi_tile_value_and_coregistration(s2_setup):
    # Tile spanning the whole cell: NDVI=(6000-2000)/(6000+2000)=0.5 on the
    # covered (right) half; left half was zeroed -> div-by-zero -> NaN.
    bbox = quad_bounds(QX, QY)
    ndvi = read_ndvi_tile(bbox, s2_setup["index"])
    assert ndvi.shape == (512, 512)
    # Buffer the ±few-px bilinear blend at the synthetic sharp zero-boundary (~col 256).
    assert np.allclose(ndvi[:, 262:], 0.5, atol=1e-4)
    assert np.isnan(ndvi[:, :250]).all()


def test_read_ndvi_tile_outside_coverage_is_nan(s2_setup):
    bbox = quad_bounds(QX + 5, QY + 5)  # no intersecting cell
    ndvi = read_ndvi_tile(bbox, s2_setup["index"])
    assert np.isnan(ndvi).all()


def test_dataset_with_ndvi_extra_stacks_and_neutralizes(quad_setup, s2_setup):
    # RGB+NDVI: image is 4-channel; NDVI z-scored on the covered half, NoData
    # (no-coverage NaN) neutralized to 0 by apply_norm.
    stats = _rgb_stats([100.0, 100.0, 100.0], [10.0, 10.0, 10.0])
    stats["extra"] = {"channel_names": ["ndvi"], "mean": [0.5], "std": [0.1]}
    b = quad_bounds(QX, QY)  # fully covered RGB quad (no alpha hole)
    tiles = pd.DataFrame([{"tile_id": "t1", "minx": b[0], "miny": b[1],
                           "maxx": b[2], "maxy": b[3]}])
    ds = InferenceTileDataset(tiles, quad_setup["index"], stats,
                              s2_index=s2_setup["index"],
                              extra_bands=[{"name": "ndvi", "band": 0}])
    item = ds[0]
    assert item["image"].shape == (4, 512, 512)
    # NDVI covered half: (0.5 - 0.5)/0.1 = 0; no-coverage half: NaN -> 0.
    assert np.allclose(item["image"][3], 0.0, atol=1e-4)


def test_dataset_extra_requires_s2_index(quad_setup):
    with pytest.raises(ValueError, match="requires an s2_index"):
        InferenceTileDataset(pd.DataFrame({"tile_id": ["a"], "minx": [0], "miny": [0],
                                           "maxx": [1], "maxy": [1]}),
                             quad_setup["index"], _rgb_stats([1, 1, 1], [1, 1, 1]),
                             extra_bands=[{"name": "ndvi", "band": 0}])


def test_dataset_rejects_non_ndvi_extra(quad_setup, s2_setup):
    with pytest.raises(NotImplementedError, match="ndvi only"):
        InferenceTileDataset(pd.DataFrame({"tile_id": ["a"], "minx": [0], "miny": [0],
                                           "maxx": [1], "maxy": [1]}),
                             quad_setup["index"], _rgb_stats([1, 1, 1], [1, 1, 1]),
                             s2_index=s2_setup["index"],
                             extra_bands=[{"name": "nbr", "band": 1}])


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


class _ConstLogit(torch.nn.Module):
    """Emits a constant logit regardless of input (known-prob ensemble math)."""
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[-2], x.shape[-1]), self.value)


def test_ensemble_single_member_equals_predict_probs():
    # 1 member at temperature T must reduce to predict_probs(model, T): the
    # fused-prob temperature inversion is the exact inverse of the per-model sigmoid.
    images = torch.randn(2, 3, 8, 8)
    for T in (1.0, 0.5, 2.0):
        solo = predict_probs(_Bias(), images, temperature=T, tta="none")
        ens = predict_probs_ensemble([_Bias()], images, temperature=T, tta="none")
        assert torch.allclose(ens, solo, atol=1e-5), T


def test_ensemble_mean_prob_then_temperature_math():
    # m1 logit=2, m2 logit=0 → per-model probs 0.8808, 0.5 → mean 0.6904
    # → fused logit 0.8023 → with T=2: sigmoid(0.4012) = 0.5990.
    images = torch.zeros(1, 3, 4, 4)
    out = predict_probs_ensemble([_ConstLogit(2.0), _ConstLogit(0.0)],
                                 images, temperature=2.0, tta="none")
    mean_p = (torch.sigmoid(torch.tensor(2.0)) + torch.sigmoid(torch.tensor(0.0))) / 2
    fused_logit = torch.log(mean_p / (1 - mean_p))
    expected = torch.sigmoid(fused_logit / 2.0)
    assert torch.allclose(out, expected.expand_as(out), atol=1e-5)


def test_ensemble_identical_members_equal_single():
    images = torch.randn(1, 3, 8, 8)
    solo = predict_probs(_Bias(), images, temperature=0.7, tta="none")
    ens = predict_probs_ensemble([_Bias(), _Bias(), _Bias()],
                                 images, temperature=0.7, tta="none")
    assert torch.allclose(ens, solo, atol=1e-5)


def test_ensemble_empty_raises():
    with pytest.raises(ValueError, match="at least one model"):
        predict_probs_ensemble([], torch.zeros(1, 3, 4, 4), temperature=1.0)


def test_gcs_package_path_is_staged(monkeypatch):
    """gs:// package dirs must route through _stage_gcs_package, not the local
    config loader (regression: every fleet worker crashed on
    `Config not found: gs:/...` — 2026-07-05 pre-launch audit)."""
    import inference.predictor as predictor_mod

    staged = {}

    def fake_stage(pkg):
        staged["pkg"] = pkg
        raise RuntimeError("staged-sentinel")

    monkeypatch.setattr(predictor_mod, "_stage_gcs_package", fake_stage)
    with pytest.raises(RuntimeError, match="staged-sentinel"):
        predictor_mod.load_deployment_package("gs://bucket/pkgs/seed42/", torch.device("cpu"))
    assert staged["pkg"] == "gs://bucket/pkgs/seed42"


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


def test_scaled_uint8_roundtrip_precision_and_nodata(tmp_path):
    # scaled_uint8 encoding: prob×250 uint8 / NoData 255. Round-trip via
    # read_probability_tile must recover probs within 1/250=0.004 and preserve NoData.
    probs = np.random.rand(64, 64).astype(np.float32)
    probs[0, :] = NODATA_PROB
    path = str(tmp_path / "u8.tif")
    write_probability_tile(path, probs, (0, 0, 64 * 3.0, 64 * 3.0),
                           dtype="scaled_uint8")
    with rasterio.open(path) as src:
        assert src.dtypes[0] == "uint8"
        assert src.nodata == NODATA_SCALED_U8
        raw = src.read(1)
    assert (raw[0, :] == NODATA_SCALED_U8).all()          # NoData row
    assert raw[1:].max() <= SCALE_U8                       # valid ≤ 250
    decoded = read_probability_tile(path)
    assert (decoded[0, :] == NODATA_PROB).all()            # NoData preserved on decode
    assert np.abs(decoded[1:] - probs[1:]).max() <= 1.0 / SCALE_U8 + 1e-6


def test_read_probability_tile_reads_float32(tmp_path):
    # read_probability_tile auto-detects the float32 encoding (returns as-is).
    probs = np.random.rand(16, 16).astype(np.float32)
    probs[0, 0] = NODATA_PROB
    path = str(tmp_path / "f32.tif")
    write_probability_tile(path, probs, (0, 0, 48.0, 48.0))  # default float32
    assert np.allclose(read_probability_tile(path), probs)


def test_merge_decodes_scaled_uint8_tiles(tmp_path):
    # merge_tiles must decode scaled_uint8 COGs identically to float32 (via
    # read_probability_tile): two overlapping constant tiles 0.2/0.8 -> mean 0.5,
    # and the NoData strip stays NoData in the merged output.
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
    pb[:, :10] = NODATA_PROB
    write_probability_tile(str(tmp_path / "a.tif"), pa, bounds, dtype="scaled_uint8")
    write_probability_tile(str(tmp_path / "b.tif"), pb, bounds, dtype="scaled_uint8")
    merged, _ = merge_tiles(tiles, str(tmp_path), sigma_px=128.0)
    assert np.allclose(merged[1:-1, 10:-1], 0.5, atol=1.0 / SCALE_U8)
    # NoData strip: only tile a (0.2) contributes there.
    assert np.allclose(merged[1:-1, 1:9], 0.2, atol=1.0 / SCALE_U8)


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


# ---------------------------------------------------------------------------
# multi-scale inference (§6.3 context reads + §7.3 arithmetic-mean fusion)
# ---------------------------------------------------------------------------
from inference.runner import (  # noqa: E402
    InferenceContext, fuse_scale_probs, run_inference, _crop_center_upsample,
)
from inference.writer import Manifest, NODATA_PROB  # noqa: E402


def test_crop_center_upsample_recovers_uniform_center():
    # 8x8 with a uniform 4x4 center; frac 0.5 crops that center and upsamples to 8.
    a = np.zeros((8, 8), np.float32); a[2:6, 2:6] = 0.9
    up = _crop_center_upsample(a, 8, 0.5)
    assert up.shape == (8, 8)
    assert np.allclose(up, 0.9)


def test_fuse_two_scales_averages_where_both_valid():
    p1 = np.full((8, 8), 0.8, np.float32); p5 = np.full((8, 8), 0.4, np.float32)
    v = np.ones((8, 8), bool)
    fused = fuse_scale_probs({1.0: p1, 0.5: p5}, {1.0: v, 0.5: v}, 8)
    assert np.allclose(fused, 0.6)  # (0.8 + 0.4) / 2


def test_fuse_falls_back_to_1x_where_05_invalid():
    p1 = np.full((8, 8), 0.8, np.float32); p5 = np.full((8, 8), 0.4, np.float32)
    fused = fuse_scale_probs({1.0: p1, 0.5: p5},
                             {1.0: np.ones((8, 8), bool), 0.5: np.zeros((8, 8), bool)}, 8)
    assert np.allclose(fused, 0.8)  # §6.3 graceful degradation to 1x-only


def test_fuse_partial_05_coverage_mixes_per_pixel():
    p1 = np.full((8, 8), 0.8, np.float32); p5 = np.full((8, 8), 0.4, np.float32)
    v5 = np.zeros((8, 8), bool); v5[:, :4] = True  # left half of the 0.5x grid valid
    fused = fuse_scale_probs({1.0: p1, 0.5: p5}, {1.0: np.ones((8, 8), bool), 0.5: v5}, 8)
    r = np.round(fused, 3)
    assert 0.6 in r and 0.8 in r  # both fused and 1x-only pixels present


def test_fuse_all_invalid_is_nan():
    p = np.full((4, 4), 0.5, np.float32); z = np.zeros((4, 4), bool)
    fused = fuse_scale_probs({1.0: p, 0.5: p}, {1.0: z, 0.5: z}, 4)
    assert np.isnan(fused).all()  # all scales NoData -> NaN (runner masks to -1.0)


def test_multiscale_dataset_yields_per_scale_images(quad_setup):
    b = quad_bounds(QX, QY)
    res = (b[2] - b[0]) / 512
    tsz = 128
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    half = tsz * res / 2  # 128px tile centered -> 0.5x reads 256px, both inside the quad
    tiles = pd.DataFrame([{"tile_id": "t", "minx": cx - half, "miny": cy - half,
                           "maxx": cx + half, "maxy": cy + half}])
    ds = InferenceTileDataset(tiles, quad_setup["index"],
                              _rgb_stats([100, 100, 100], [10, 10, 10]),
                              tile_size_px=tsz, scales=[1.0, 0.5])
    item = ds[0]
    assert set(item["images"]) == {1.0, 0.5}
    assert item["images"][1.0].shape == (3, tsz, tsz)
    assert item["images"][0.5].shape == (3, tsz, tsz)
    assert item["valid"][1.0].shape == (tsz, tsz)
    assert not item["all_nodata"]
    assert item["valid"][1.0].all() and item["valid"][0.5].all()  # uniform quad
    assert np.allclose(item["images"][1.0], 0.0)  # (100-100)/10


class _ConstLogitModel(torch.nn.Module):
    """Returns a constant logit (0 -> prob 0.5) at the input spatial size."""
    def forward(self, x):
        return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])


def test_run_inference_multiscale_writes_fused_cog(quad_setup, tmp_path):
    # Whole-quad bbox → 512 native px == the dataset's default tile_size.
    b = quad_bounds(QX, QY)
    tiles = pd.DataFrame([{"tile_id": "t", "minx": b[0], "miny": b[1],
                           "maxx": b[2], "maxy": b[3]}])
    model = _ConstLogitModel().eval()
    ctx = InferenceContext(
        models=[model], pkg={"model": model, "stats": _rgb_stats([100, 100, 100], [10, 10, 10])},
        dep_cfg={"temperature": 1.0, "tta": "none", "precision": "fp32", "scales": [1.0, 0.5]},
        run_cfg={"inference": {"batch_size": 2}}, quad_index=quad_setup["index"],
        s2_index=None, extra_bands=[], ensemble=False, package_paths=["dummy"])
    out = tmp_path / "out"
    manifest = Manifest(str(tmp_path / "log.json"), {}, checkpoint_every=1)
    counts = run_inference(ctx, tiles, str(out), manifest, torch.device("cpu"), num_workers=0)
    assert counts["n_tiles_processed"] == 1
    with rasterio.open(out / "t.tif") as src:
        arr = src.read(1)
    assert arr.shape == (512, 512)
    # const logit 0 -> prob 0.5 at both scales -> fused 0.5 everywhere valid
    valid = arr != NODATA_PROB
    assert valid.any() and np.allclose(arr[valid], 0.5, atol=1e-6)
