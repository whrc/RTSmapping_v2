"""Merge overlapping probability tiles into regional rasters (inference.md §4.3).

Gaussian distance-from-tile-center weighted average (sigma from
configs/deployment.yaml inference.fusion_sigma_px), NoData-aware: tile pixels
at the -1.0 sentinel contribute zero weight; output pixels with no valid
contribution stay -1.0. The calibrated threshold then produces the §9.2 binary
mask.

The merge extent is the union of the tile list — sized for a region/AOI chunk
(the PDG workflow partitions pan-arctic runs into chunks; merging a whole
hemisphere in one call is out of scope).

Usage:
    python scripts/merge_predictions.py \
        --config configs/deployment.yaml \
        --tile-list tiles.csv \
        --tiles-dir gs://.../inference/2025-Q3/tiles \
        --package gs://.../models/rts-v2-seed42 \
        --output-prob merged_prob.tif --output-mask merged_mask.tif
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.quad_index import RESOLUTION_M  # noqa: E402
from inference.tiles import TILE_SIZE_PX  # noqa: E402
from inference.writer import (  # noqa: E402
    NODATA_MASK, NODATA_PROB, write_binary_mask, write_probability_tile,
)
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def gaussian_center_weights(size_px: int, sigma_px: float) -> np.ndarray:
    """(size, size) weight grid peaking at the tile center, zero at edges (§4.3).

    Separable edge-zeroed Gaussian: per-axis g(i) = exp(−(i−c)²/2σ²) − g_edge,
    w = g ⊗ g. The plain radial Gaussian keeps weight exp(−2) ≈ 0.135 at the
    tile edge, so a tile's contribution appears/disappears *discontinuously*
    across stitch seams — measured as ~7× elevated probability gradients on
    seam lines (tiny-area validation, 2026-06-12). Zeroing the weight exactly
    at the edge makes contributions fade in continuously and removes the seam
    artifact while keeping the §4.3 center-trust rationale and σ unchanged.
    Consequence: pixels covered only by another tile's outermost row/column
    (e.g. the 1-px ring at an AOI boundary) have zero total weight → NoData.
    """
    c = (size_px - 1) / 2.0
    i = np.arange(size_px)
    g = np.exp(-((i - c) ** 2) / (2.0 * sigma_px ** 2))
    g = np.maximum(g - g[0], 0.0)
    return np.outer(g, g)


def merge_tiles(
    tile_list: pd.DataFrame,
    tiles_dir: str,
    sigma_px: float,
    tile_size_px: int = TILE_SIZE_PX,
    resolution_m: float = RESOLUTION_M,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Weighted-average merge of per-tile probability rasters.

    Returns (merged float32 array with -1.0 NoData, merge bounds).
    Tiles whose raster is missing (skipped all-NoData tiles) are ignored.
    `resolution_m` is the per-pixel size of the tile rasters (9.55 m at
    inference scale 0.5).
    """
    minx, miny = tile_list["minx"].min(), tile_list["miny"].min()
    maxx, maxy = tile_list["maxx"].max(), tile_list["maxy"].max()
    width = int(round((maxx - minx) / resolution_m))
    height = int(round((maxy - miny) / resolution_m))
    logger.info("Merge canvas: %d x %d px over (%.0f, %.0f, %.0f, %.0f)",
                width, height, minx, miny, maxx, maxy)

    acc = np.zeros((height, width), dtype=np.float64)
    wsum = np.zeros((height, width), dtype=np.float64)
    weights = gaussian_center_weights(tile_size_px, sigma_px)

    n_used = 0
    for _, t in tile_list.iterrows():
        path = f"{tiles_dir.rstrip('/')}/{t['tile_id']}.tif"
        try:
            with rasterio.open(path) as src:
                probs = src.read(1)
        except rasterio.errors.RasterioIOError:
            continue  # skipped tile (all-NoData) or not yet produced
        valid = probs != NODATA_PROB
        col0 = int(round((t["minx"] - minx) / resolution_m))
        row0 = int(round((maxy - t["maxy"]) / resolution_m))
        w = weights * valid
        sl = (slice(row0, row0 + tile_size_px), slice(col0, col0 + tile_size_px))
        acc[sl] += np.where(valid, probs, 0.0) * w
        wsum[sl] += w
        n_used += 1

    merged = np.full((height, width), NODATA_PROB, dtype=np.float32)
    has = wsum > 0
    merged[has] = (acc[has] / wsum[has]).astype(np.float32)
    logger.info("Merged %d tiles; %.1f%% of canvas has data",
                n_used, 100.0 * has.mean())
    return merged, (minx, miny, maxx, maxy)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/deployment.yaml")
    p.add_argument("--tile-list", required=True)
    p.add_argument("--tiles-dir", required=True)
    p.add_argument("--package", required=True,
                   help="deployment package dir (threshold source)")
    p.add_argument("--output-prob", required=True)
    p.add_argument("--output-mask", required=True)
    args = p.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    dep_cfg = load_config(f"{str(args.package).rstrip('/')}/deployment_config.yaml")
    threshold = dep_cfg["threshold"]
    if threshold is None:
        raise ValueError("deployment package threshold is null (uncalibrated)")

    tiles = pd.read_csv(args.tile_list)
    merged, bounds = merge_tiles(tiles, args.tiles_dir,
                                 sigma_px=cfg["inference"]["fusion_sigma_px"])
    write_probability_tile(args.output_prob, merged, bounds)

    mask = np.where(merged == NODATA_PROB, NODATA_MASK,
                    (merged >= threshold).astype(np.uint8)).astype(np.uint8)
    write_binary_mask(args.output_mask, mask, bounds)
    logger.info("Wrote %s and %s (threshold %.4f)",
                args.output_prob, args.output_mask, threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
