"""Re-cut the South probability canvas into WMTS-conformant COGs.

ADC asked for rasters "arranged as COGs in a tile grid that conforms to one of
the WMTS tiling schemes" so each file corresponds to precisely one tile of a
Tile Matrix Set. The original `probability_cog_shards/` grid is canvas-anchored
(rows offset from the global grid); this script re-emits the same pixels as one
8,192-px COG per **WebMercatorQuad zoom-10 tile** (39,135.758 m), named
`{col:04d}_{row:04d}.tif` with *global* z10 indices (col 0-1023 from 180°W,
row from the matrix origin at the north edge).

The canvas grid is aligned to WebMercatorQuad to < 0.001 px (verified: the
fractional source-pixel offset of every z10 tile edge is 0.9994-1.0006), so the
integer-rounded window read is an exact pass-through - no resampling, values
and positions unchanged. All-NoData tiles are skipped.

Usage (inside the rts-train docker, on a local copy of the products dir):
    python scripts/retile_wmts_z10.py \
        --vrt /work/products/probability.vrt --out-dir /work/probability_wmts_z10 \
        --workers 32
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
from multiprocessing import Pool, get_context
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

logger = logging.getLogger(__name__)

WM = 20037508.342789244            # WebMercatorQuad half-extent (m)
Z10_TILE_M = 2 * WM / 1024         # 39135.75848201024 m
Z15_RES = 2 * WM / (256 * 2**15)   # 4.777314267823516 m/px
TILE_PX = 8192                     # Z10_TILE_M / Z15_RES
NODATA = 255


def z10_tile_bounds(col: int, row: int) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) of one global WebMercatorQuad z10 tile.

    Every edge is computed from its own index so adjacent tiles share edge
    coordinates bit-exactly (minx + Z10_TILE_M drifts by ~2e-8 m otherwise).
    """
    return (-WM + col * Z10_TILE_M, WM - (row + 1) * Z10_TILE_M,
            -WM + (col + 1) * Z10_TILE_M, WM - row * Z10_TILE_M)


def candidate_tiles(bounds: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """All (col, row) z10 tiles intersecting a bbox (e.g. one source shard).

    The z10 matrix is square: 1024 columns x 1024 rows.
    """
    minx, miny, maxx, maxy = bounds
    c0 = max(0, math.floor((minx + WM) / Z10_TILE_M))
    c1 = min(1023, math.floor((maxx + WM) / Z10_TILE_M - 1e-9))
    r0 = max(0, math.floor((WM - maxy) / Z10_TILE_M))
    r1 = min(1023, math.floor((WM - miny) / Z10_TILE_M - 1e-9))
    return [(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)]


_src = None  # per-worker VRT handle


def _init(vrt_path: str) -> None:
    global _src
    _src = rasterio.open(vrt_path)


def _write_tile(args: tuple[int, int, str]) -> str | None:
    """Read one z10 tile off the source grid and write it as a COG.

    Returns the written path, or None if the tile is entirely NoData.
    Reads are plain (non-boundless) windows clipped to the canvas — rasterio's
    boundless path wraps every read in a temporary VRT and is ~100x slower —
    pasted into a NODATA-filled tile. A 16x-decimated pre-read (overview
    tiles only) skips fully-empty tiles without decompressing full data.
    """
    col, row, out_dir = args
    minx, _, _, maxy = z10_tile_bounds(col, row)
    # source col/row of the tile's NW corner; transform.e is negative (north-up)
    px = round((minx - _src.transform.c) / _src.transform.a)
    py = round((_src.transform.f - maxy) / -_src.transform.e)
    cx0, cy0 = max(px, 0), max(py, 0)
    cx1, cy1 = min(px + TILE_PX, _src.width), min(py + TILE_PX, _src.height)
    if cx0 >= cx1 or cy0 >= cy1:
        return None
    win = Window(cx0, cy0, cx1 - cx0, cy1 - cy0)

    peek = _src.read(1, window=win,
                     out_shape=(max(1, win.height // 16), max(1, win.width // 16)))
    if (peek == NODATA).all():
        return None

    arr = np.full((TILE_PX, TILE_PX), NODATA, dtype=np.uint8)
    arr[cy0 - py:cy1 - py, cx0 - px:cx1 - px] = _src.read(1, window=win)
    if (arr == NODATA).all():
        return None
    out_path = Path(out_dir) / f"{col:04d}_{row:04d}.tif"
    profile = dict(driver="COG", height=TILE_PX, width=TILE_PX, count=1,
                   dtype="uint8", crs="EPSG:3857", nodata=NODATA,
                   transform=from_origin(minx, maxy, Z15_RES, Z15_RES),
                   compress="deflate", blocksize=512,
                   overview_resampling="nearest")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)
    return str(out_path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vrt", required=True, help="local probability.vrt")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    with rasterio.open(args.vrt) as src:
        members = [f for f in (src.files or []) if f.endswith(".tif")]
        shard_bounds = ([tuple(rasterio.open(f).bounds) for f in members]
                        or [tuple(src.bounds)])
    tiles: set[tuple[int, int]] = set()
    for b in shard_bounds:
        tiles.update(candidate_tiles(b))
    logger.info("%d candidate z10 tiles", len(tiles))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, r, str(args.out_dir)) for c, r in sorted(tiles)]
    written = 0
    with get_context("spawn").Pool(args.workers, _init, (args.vrt,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_write_tile, jobs, chunksize=8)):
            if res:
                written += 1
            if (i + 1) % 5000 == 0:
                logger.info("%d/%d checked, %d written", i + 1, len(jobs), written)
    logger.info("Done: %d non-empty tiles of %d candidates", written, len(jobs))

    list_file = args.out_dir / "tile_list.txt"
    list_file.write_text("\n".join(sorted(str(p) for p in args.out_dir.glob("*.tif"))))
    vrt_out = args.out_dir.parent / "probability_wmts_z10.vrt"
    subprocess.run(["gdalbuildvrt", "-input_file_list", str(list_file),
                    str(vrt_out)], check=True, stdout=subprocess.DEVNULL)
    list_file.unlink()
    logger.info("Wrote %s", vrt_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
