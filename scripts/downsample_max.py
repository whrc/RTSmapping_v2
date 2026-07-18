"""Exact block-max downsample of the probability canvas → browse likelihood
surface (D2 `likelihood_95m.tif`).

Each output pixel is the maximum decoded-valid value over its factor×factor
source block — definitionally "the highest RTS probability within this ~95 m
cell". Replaces `gdalwarp -r max`, whose kernel bled NoData-edge artifacts
(values 251–254, decoding to prob > 1.0) onto coverage seams. NoData (255) is
excluded from the max; an all-NoData block stays NoData; valid output is
guaranteed ≤ 250.

Usage:
    python scripts/downsample_max.py \
        --src /outputs/.../probability.vrt \
        --out /outputs/.../likelihood_95m.tif --factor 20 --workers 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path

import numpy as np
import rasterio
from rasterio import windows
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.writer import NODATA_SCALED_U8, SCALE_U8  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# per-worker window: rows/cols must be multiples of factor
WIN_ROWS = 4000
WIN_COLS = 40000


def _block_max(spec: dict):
    """Block-max one window → (out_row0, out_col0, small array)."""
    f = spec["factor"]
    with rasterio.open(spec["src"]) as src:
        win = windows.Window(*spec["window"])
        d = src.read(1, window=win)
    h, w = d.shape
    ph, pw = ceil(h / f) * f, ceil(w / f) * f
    if (ph, pw) != (h, w):  # pad partial edge blocks with NoData
        p = np.full((ph, pw), NODATA_SCALED_U8, dtype=d.dtype)
        p[:h, :w] = d
        d = p
    # NoData must never win the max: remap 255 → 0, then restore all-NoData
    valid = d != NODATA_SCALED_U8
    blocks_valid = valid.reshape(ph // f, f, pw // f, f).any(axis=(1, 3))
    m = np.where(valid, d, 0).reshape(ph // f, f, pw // f, f).max(axis=(1, 3))
    m[~blocks_valid] = NODATA_SCALED_U8
    return spec["window"][1] // f, spec["window"][0] // f, m.astype("uint8")


def _max_reduce2(a: np.ndarray) -> np.ndarray:
    """2× block-max with NoData semantics (255 never wins; all-NoData stays)."""
    h, w = a.shape
    ph, pw = ceil(h / 2) * 2, ceil(w / 2) * 2
    if (ph, pw) != (h, w):
        p = np.full((ph, pw), NODATA_SCALED_U8, dtype=a.dtype)
        p[:h, :w] = a
        a = p
    valid = a != NODATA_SCALED_U8
    bv = valid.reshape(ph // 2, 2, pw // 2, 2).any(axis=(1, 3))
    m = np.where(valid, a, 0).reshape(ph // 2, 2, pw // 2, 2).max(axis=(1, 3))
    m[~bv] = NODATA_SCALED_U8
    return m.astype("uint8")


def _write_max_overviews(out_path: str, base: np.ndarray,
                         levels: tuple = (2, 4, 8, 16, 32, 64)) -> None:
    """Exact block-max overview pyramid.

    GDAL 3.4 cannot build MAX overviews (nearest/average/… only), and nearest
    drops the sparse peaks — the original 'blank when zoomed out' defect. So:
    create the overview structure with NEAREST, then overwrite every overview
    band with successive exact 2× max-reductions of the base array.
    """
    from osgeo import gdal
    ds = gdal.Open(out_path, gdal.GA_Update)
    ds.BuildOverviews("NEAREST", list(levels))
    band = ds.GetRasterBand(1)
    a = base
    for i in range(band.GetOverviewCount()):
        a = _max_reduce2(a)
        ov = band.GetOverview(i)
        assert (ov.YSize, ov.XSize) == a.shape, (i, ov.YSize, ov.XSize, a.shape)
        ov.WriteArray(a)
    ds.FlushCache()
    ds = None


def downsample_max(src_path: str, out_path: str, factor: int = 20,
                   workers: int = 16) -> None:
    """Write the factor× block-max of ``src_path`` (scaled_uint8) to a COG-style
    tiled GTiff with matching georeferencing (pixel size × factor)."""
    with rasterio.open(src_path) as src:
        sw, sh = src.width, src.height
        t = src.transform
        crs = src.crs
    ow, oh = ceil(sw / factor), ceil(sh / factor)
    out = np.full((oh, ow), NODATA_SCALED_U8, dtype=np.uint8)

    specs = [dict(src=src_path, factor=factor,
                  window=(c0, r0, min(WIN_COLS, sw - c0), min(WIN_ROWS, sh - r0)))
             for r0 in range(0, sh, WIN_ROWS) for c0 in range(0, sw, WIN_COLS)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for orow, ocol, m in ex.map(_block_max, specs, chunksize=8):
            out[orow:orow + m.shape[0], ocol:ocol + m.shape[1]] = m

    ot = Affine(t.a * factor, t.b, t.c, t.d, t.e * factor, t.f)
    with rasterio.open(out_path, "w", driver="GTiff", width=ow, height=oh,
                       count=1, dtype="uint8", crs=crs, transform=ot,
                       nodata=float(NODATA_SCALED_U8), tiled=True,
                       compress="deflate", bigtiff="yes") as dst:
        dst.write(out, 1)
    with rasterio.open(out_path, "r+") as dst:
        # embedded color table: file opens colormapped (near-white → deep red
        # over 0..250), instead of a gray stretch that renders blank
        cmap = {v: (250 - int(70 * v / 250), 250 - int(230 * v / 250),
                    250 - int(230 * v / 250), 255) for v in range(251)}
        cmap[NODATA_SCALED_U8] = (0, 0, 0, 0)
        dst.write_colormap(1, cmap)
    _write_max_overviews(out_path, out)
    valid = out != NODATA_SCALED_U8
    logger.info("wrote %s (%dx%d, %.2f%% valid, max %d)", out_path, ow, oh,
                100 * valid.mean(), out[valid].max() if valid.any() else -1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--factor", type=int, default=20)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    setup_logging()
    downsample_max(args.src, args.out, args.factor, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
