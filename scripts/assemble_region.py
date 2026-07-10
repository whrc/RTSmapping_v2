"""Assemble a large region's per-tile probability COGs into single-region
products (post-inference.md §7 region assembly).

`merge_predictions.py` merges a tile list whose union canvas fits in RAM; a
whole region such as Banks Island is 200k x 310k px, whose float64 merge
accumulators would need ~1 TB. This script tiles that *same* Gaussian
distance-from-center weighted-mean merge (inference.md §4.3) over a grid of
output blocks, so the whole-region canvas never lives in memory at once, then
mosaics the blocks into one Cloud-Optimized GeoTIFF (with internal overviews).

Each output block includes **every** tile whose footprint intersects it and
clips each tile's contribution to the block window — so a block's merge is
identical to the corresponding window of a single-shot merge, and the
non-overlapping blocks mosaic seamlessly (verified in
tests/test_assemble_region.py).

Reads prob tiles from a *local* directory: the per-tile COGs are tiny (~1 KB
scaled_uint8) but number in the hundreds of thousands, so a windowed cross-region
read per tile would take many hours — bulk-rsync them local first, then run this.

Usage:
    python scripts/assemble_region.py \
        --tile-list banks_tiles.csv \
        --probs-dir /local/banks/probs \
        --package gs://.../packages/seed42 \
        --out-dir /local/banks/out \
        --block-px 16384
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.quad_index import RESOLUTION_M  # noqa: E402
from inference.tiles import TILE_SIZE_PX  # noqa: E402
from inference.writer import (  # noqa: E402
    NODATA_MASK, NODATA_PROB, NODATA_SCALED_U8, read_probability_tile, write_binary_mask,
    write_probability_tile,
)
from scripts.merge_predictions import gaussian_center_weights  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def canvas_bounds(tiles: pd.DataFrame, resolution_m: float
                  ) -> tuple[tuple[float, float, float, float], int, int]:
    """Region bounds (minx, miny, maxx, maxy) and pixel width/height."""
    minx, miny = float(tiles["minx"].min()), float(tiles["miny"].min())
    maxx, maxy = float(tiles["maxx"].max()), float(tiles["maxy"].max())
    width = int(round((maxx - minx) / resolution_m))
    height = int(round((maxy - miny) / resolution_m))
    return (minx, miny, maxx, maxy), width, height


def iter_blocks(width: int, height: int, block_px: int):
    """Yield non-overlapping (row0, row1, col0, col1) output blocks."""
    for r0 in range(0, height, block_px):
        r1 = min(r0 + block_px, height)
        for c0 in range(0, width, block_px):
            c1 = min(c0 + block_px, width)
            yield r0, r1, c0, c1


def merge_window(tiles: pd.DataFrame, tile_paths: dict[str, str],
                 win_bounds: tuple[float, float, float, float],
                 sigma_px: float, resolution_m: float = RESOLUTION_M,
                 tile_size_px: int = TILE_SIZE_PX) -> np.ndarray:
    """Weighted-mean merge (inference.md §4.3) over one output-block window.

    `tiles` must already be the subset whose footprint intersects `win_bounds`.
    Each tile's Gaussian-weighted contribution is placed at its offset within the
    window and clipped to the window edges. Returns a float32 array (window
    height x width) with the NODATA_PROB (-1.0) sentinel where no tile contributes.
    """
    w_minx, w_miny, w_maxx, w_maxy = win_bounds
    height = int(round((w_maxy - w_miny) / resolution_m))
    width = int(round((w_maxx - w_minx) / resolution_m))
    acc = np.zeros((height, width), dtype=np.float64)
    wsum = np.zeros((height, width), dtype=np.float64)
    weights = gaussian_center_weights(tile_size_px, sigma_px)

    for _, t in tiles.iterrows():
        path = tile_paths.get(str(t["tile_id"]))
        if path is None:
            continue
        try:
            probs = read_probability_tile(path)
        except rasterio.errors.RasterioIOError:
            continue  # all-NoData tile that was skipped at inference
        valid = probs != NODATA_PROB
        # Tile top-left in window pixel coords (may fall outside the window).
        row0 = int(round((w_maxy - float(t["maxy"])) / resolution_m))
        col0 = int(round((float(t["minx"]) - w_minx) / resolution_m))
        # Clip the tile <-> window overlap.
        dr0, dc0 = max(row0, 0), max(col0, 0)
        dr1 = min(row0 + tile_size_px, height)
        dc1 = min(col0 + tile_size_px, width)
        if dr1 <= dr0 or dc1 <= dc0:
            continue
        sr0, sc0 = dr0 - row0, dc0 - col0
        sr1, sc1 = sr0 + (dr1 - dr0), sc0 + (dc1 - dc0)
        pv = probs[sr0:sr1, sc0:sc1]
        vv = valid[sr0:sr1, sc0:sc1]
        w = weights[sr0:sr1, sc0:sc1] * vv
        acc[dr0:dr1, dc0:dc1] += np.where(vv, pv, 0.0) * w
        wsum[dr0:dr1, dc0:dc1] += w

    merged = np.full((height, width), NODATA_PROB, dtype=np.float32)
    has = wsum > 0
    merged[has] = (acc[has] / wsum[has]).astype(np.float32)
    return merged


def _process_block(spec: dict):
    """Merge one output block and write its prob+mask COGs (parallel worker).

    Self-contained: `spec` carries the block window, the intersecting tile rows,
    and just those tiles' paths — so no large shared state crosses the pool.
    Returns (prob_path, mask_path) or None for an all-NoData block.
    """
    win_bounds = spec["win_bounds"]
    merged = merge_window(spec["tiles"], spec["tile_paths"], win_bounds,
                          spec["sigma_px"], spec["resolution_m"])
    if not (merged != NODATA_PROB).any():
        return None
    prob_path = f"{spec['blocks_dir']}/prob_{spec['r0']:07d}_{spec['c0']:07d}.tif"
    mask_path = f"{spec['blocks_dir']}/mask_{spec['r0']:07d}_{spec['c0']:07d}.tif"
    write_probability_tile(prob_path, merged, win_bounds,
                           dtype=spec.get("output_dtype", "float32"))
    mask = np.where(merged == NODATA_PROB, NODATA_MASK,
                    (merged >= spec["threshold"]).astype(np.uint8)).astype(np.uint8)
    write_binary_mask(mask_path, mask, win_bounds)
    return spec["r0"], spec["c0"], prob_path, mask_path


def _build_cog(vrt_inputs: list[str], vrt_path: str, cog_path: str,
               dtype_nodata: float | int) -> None:
    """gdalbuildvrt the block files then gdal_translate to a COG w/ overviews."""
    subprocess.run(["gdalbuildvrt", "-vrtnodata", str(dtype_nodata),
                    vrt_path, *vrt_inputs], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["gdal_translate", "-of", "COG",
                    "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES",
                    "-co", "OVERVIEWS=AUTO", "-co", "NUM_THREADS=ALL_CPUS",
                    vrt_path, cog_path], check=True, stdout=subprocess.DEVNULL)


def _translate_one_cog(spec: dict) -> str:
    """Pool worker: VRT a super-tile's member block COGs, then COG-translate it."""
    subprocess.run(["gdalbuildvrt", "-q", "-vrtnodata", str(spec["dtype_nodata"]),
                    spec["vrt_path"], *spec["members"]], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["gdal_translate", "-q", "-of", "COG",
                    "-co", "COMPRESS=DEFLATE", "-co", "BIGTIFF=YES",
                    "-co", "OVERVIEWS=AUTO",
                    "-co", f"NUM_THREADS={spec['num_threads']}",
                    spec["vrt_path"], spec["cog_path"]], check=True,
                   stdout=subprocess.DEVNULL)
    return spec["cog_path"]


def _build_cog_grid(entries: list[tuple[int, int, str]], out_dir: Path, name: str,
                    dtype_nodata: float | int, cog_tile_px: int, block_px: int,
                    workers: int) -> tuple[Path, list[str]]:
    """Mosaic block COGs into a grid of super-tile COGs built in parallel + a .vrt.

    A single ``gdal_translate -of COG`` over the whole-region VRT serializes on
    one process — the ~53-min step on Banks, and hopeless at South's ~40x extent
    (one ~280 GB COG). Instead the block COGs are grouped into super-tiles of
    ``cog_tile_px`` (rounded up to a multiple of ``block_px``); each super-tile is
    COG-translated in its own process, so the translate scales with ``workers``.
    The returned ``<name>.vrt`` mosaics the super-tile COGs (referenced by paths
    relative to ``out_dir``, so the out-dir is relocatable/uploadable as a unit)
    into one addressable raster for ArcGIS / vectorize_region; each super-tile
    carries its own overviews.

    ``entries`` = (block_r0, block_c0, block_cog_path). Returns (vrt_path, cog_paths).
    """
    step = max(block_px, (cog_tile_px // block_px) * block_px)  # whole blocks only
    shard_reldir = f"{name}_cog_shards"
    shard_dir = out_dir / shard_reldir
    shard_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[int, int], list[str]] = {}
    for r0, c0, path in entries:
        groups.setdefault((r0 // step, c0 // step), []).append(path)

    ncpu = os.cpu_count() or 2
    num_threads = max(1, ncpu // max(1, workers))  # avoid oversubscribing across workers
    specs = []
    for (gr, gc), members in sorted(groups.items()):
        stem = f"{name}_{gr:04d}_{gc:04d}"
        specs.append(dict(members=members, vrt_path=str(shard_dir / f"{stem}.vrt"),
                          cog_path=str(shard_dir / f"{stem}.tif"),
                          dtype_nodata=dtype_nodata, num_threads=num_threads))
    logger.info("Building %d %s super-tile COGs (%d workers x %d GDAL threads)",
                len(specs), name, workers, num_threads)

    cog_paths: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, cp in enumerate(ex.map(_translate_one_cog, specs), 1):
            cog_paths.append(cp)
            if i % 10 == 0 or i == len(specs):
                logger.info("%d/%d %s super-tile COGs done", i, len(specs), name)

    # Region VRT with paths relative to out_dir (portable alongside the shard dir).
    vrt_path = out_dir / f"{name}.vrt"
    rel_members = [f"{shard_reldir}/{Path(cp).name}" for cp in cog_paths]
    subprocess.run(["gdalbuildvrt", "-q", "-vrtnodata", str(dtype_nodata),
                    f"{name}.vrt", *rel_members], check=True,
                   stdout=subprocess.DEVNULL, cwd=str(out_dir))
    return vrt_path, cog_paths


def assemble(tiles: pd.DataFrame, tile_paths: dict[str, str], out_dir: Path,
             threshold: float, sigma_px: float, block_px: int,
             resolution_m: float = RESOLUTION_M, workers: int = 1,
             cog_tile_px: int = 0, output_dtype: str = "float32") -> dict:
    """Blocked merge → per-block prob+mask COGs → mosaicked region product.

    Block merges are independent, so they run across a process pool (`workers`);
    each task carries only its window + intersecting tiles' paths. The final
    mosaic is a single COG when ``cog_tile_px == 0`` (Banks-scale), or a grid of
    parallel super-tile COGs + a ``.vrt`` when ``cog_tile_px > 0`` (South-scale,
    where one monolithic COG-translate is the bottleneck / unwieldy).

    ``output_dtype`` sets the probability encoding of both the intermediate block
    COGs and the final product: ``"float32"`` (exact, NoData -1) or
    ``"scaled_uint8"`` (prob×250, NoData 255 — the deploy encoding; ~4× smaller,
    essential at circumpolar scale where float32 intermediates run to TBs). The
    mask is always uint8. ``read_probability_tile`` decodes either transparently.
    """
    prob_nodata = NODATA_SCALED_U8 if output_dtype == "scaled_uint8" else NODATA_PROB
    bounds, width, height = canvas_bounds(tiles, resolution_m)
    minx, miny, maxx, maxy = bounds
    logger.info("Region canvas %d x %d px over (%.0f, %.0f, %.0f, %.0f)",
                width, height, *bounds)
    blocks_dir = out_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    ty0 = tiles["miny"].to_numpy()
    ty1 = tiles["maxy"].to_numpy()

    # Row-banded pre-filtering: a naive per-block scan intersects every one of the
    # (up to tens of millions of) tiles per block — O(blocks x tiles), ~hours at the
    # circumpolar 8.4M-px canvas. Group blocks by row-band, mask tiles to the band
    # once, then intersect columns against that small subset. Same strict-inequality
    # result, orders of magnitude faster.
    from collections import OrderedDict
    rows: "OrderedDict[tuple[int, int], list[tuple[int, int]]]" = OrderedDict()
    for r0, r1, c0, c1 in iter_blocks(width, height, block_px):
        rows.setdefault((r0, r1), []).append((c0, c1))

    specs = []
    for (r0, r1), cols in rows.items():
        w_maxy = maxy - r0 * resolution_m
        w_miny = maxy - r1 * resolution_m
        band = (ty1 > w_miny) & (ty0 < w_maxy)
        if not band.any():
            continue
        b_tiles = tiles.iloc[band]
        bx0 = b_tiles["minx"].to_numpy()
        bx1 = b_tiles["maxx"].to_numpy()
        for c0, c1 in cols:
            w_minx = minx + c0 * resolution_m
            w_maxx = minx + c1 * resolution_m
            sel = (bx1 > w_minx) & (bx0 < w_maxx)  # y already satisfied by the band
            if not sel.any():
                continue
            sub = b_tiles.iloc[sel]
            specs.append(dict(
                r0=r0, c0=c0, win_bounds=(w_minx, w_miny, w_maxx, w_maxy),
                tiles=sub, tile_paths={str(t): tile_paths[str(t)]
                                       for t in sub["tile_id"] if str(t) in tile_paths},
                threshold=threshold, sigma_px=sigma_px, resolution_m=resolution_m,
                blocks_dir=str(blocks_dir), output_dtype=output_dtype))
    logger.info("%d candidate blocks; merging with %d workers", len(specs), workers)

    prob_entries, mask_entries = [], []
    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_process_block, specs):
            n_done += 1
            if n_done % 20 == 0:
                logger.info("%d/%d blocks merged", n_done, len(specs))
            if res is not None:
                r0, c0, prob_path, mask_path = res
                prob_entries.append((r0, c0, prob_path))
                mask_entries.append((r0, c0, mask_path))

    if not prob_entries:
        raise RuntimeError("no non-empty blocks — check tile-list / probs-dir")

    if cog_tile_px and cog_tile_px > 0:
        logger.info("Mosaicking %d prob + %d mask blocks into super-tile COG grids",
                    len(prob_entries), len(mask_entries))
        prob_out, prob_shards = _build_cog_grid(
            prob_entries, out_dir, "probability", prob_nodata, cog_tile_px, block_px, workers)
        mask_out, mask_shards = _build_cog_grid(
            mask_entries, out_dir, "mask", NODATA_MASK, cog_tile_px, block_px, workers)
        n_cog_shards = len(prob_shards)
    else:
        logger.info("Mosaicking %d prob + %d mask blocks into region COGs",
                    len(prob_entries), len(mask_entries))
        prob_out = out_dir / "probability.tif"
        mask_out = out_dir / "mask.tif"
        _build_cog([e[2] for e in prob_entries], str(out_dir / "probability.vrt"),
                   str(prob_out), prob_nodata)
        _build_cog([e[2] for e in mask_entries], str(out_dir / "mask.vrt"),
                   str(mask_out), NODATA_MASK)
        n_cog_shards = 1

    return {
        "region_bounds_3857": bounds,
        "canvas_px": [width, height],
        "resolution_m": resolution_m,
        "threshold": threshold,
        "fusion_sigma_px": sigma_px,
        "block_px": block_px,
        "cog_tile_px": cog_tile_px,
        "n_cog_shards": n_cog_shards,
        "n_blocks_nonempty": len(prob_entries),
        "n_tiles": int(len(tiles)),
        "probability_cog": str(prob_out),
        "mask_cog": str(mask_out),
    }


def build_tile_paths(probs_dir: str) -> dict[str, str]:
    """Map tile_id -> local prob COG path from a rsync'd probs/ tree."""
    paths = {}
    for p in glob.glob(f"{probs_dir.rstrip('/')}/**/*.tif", recursive=True):
        paths[Path(p).stem] = p
    return paths


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/deployment.yaml")
    p.add_argument("--tile-list", required=True)
    p.add_argument("--probs-dir", required=True,
                   help="local dir holding the rsync'd per-tile prob COGs")
    p.add_argument("--package", required=True,
                   help="deployment package dir (threshold source)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--block-px", type=int, default=16384)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 3))
    p.add_argument("--cog-tile-px", type=int, default=0,
                   help="0 = one monolithic region COG (Banks-scale); >0 = a grid of "
                        "super-tile COGs of this edge (rounded to a block multiple) built "
                        "in parallel + a .vrt mosaic (South-scale). e.g. 65536")
    p.add_argument("--output-dtype", default="float32",
                   choices=["float32", "scaled_uint8"],
                   help="probability encoding of the blocks + final product; scaled_uint8 "
                        "(deploy encoding, ~4x smaller) is required at circumpolar scale")
    args = p.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    dep_cfg = load_config(f"{str(args.package).rstrip('/')}/deployment_config.yaml")
    threshold = dep_cfg["threshold"]
    if threshold is None:
        raise ValueError("deployment package threshold is null (uncalibrated)")

    tiles = pd.read_csv(args.tile_list)
    tile_paths = build_tile_paths(args.probs_dir)
    logger.info("%d tiles in list, %d prob COGs found under %s",
                len(tiles), len(tile_paths), args.probs_dir)

    t0 = time.time()
    summary = assemble(tiles, tile_paths, args.out_dir, threshold,
                       sigma_px=cfg["inference"]["fusion_sigma_px"],
                       block_px=args.block_px, workers=args.workers,
                       cog_tile_px=args.cog_tile_px, output_dtype=args.output_dtype)
    summary["assemble_time_hours"] = round((time.time() - t0) / 3600, 3)
    (args.out_dir / "region_log.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s and %s (%.2f h)", summary["probability_cog"],
                summary["mask_cog"], summary["assemble_time_hours"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
