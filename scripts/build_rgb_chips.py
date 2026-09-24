"""Generate RGB context chips for the tiles a detected RTS polygon references,
for the ArcGIS Pro QC package (post-inference.md's "underlying tiles" gap: no
RGB mosaic is persisted at deployment scale — inference only windows tiles
on the fly from Planet quads).

Only the tiles that actually intersect a detection are chipped (not the whole
region — that's the same region-canvas-scale problem `assemble_region.py`
already solves for prob/mask, whether that's Banks' 200k x 310k px or the full
South run's continental canvas), reusing the exact windowed-read path real
inference uses (`inference.tiles.read_tile`) so the chip imagery matches what
the model saw (CLAUDE Rule 3 — no duplicated tile-reading logic).

`--wide-context` widens that rule to the tiles covering each polygon's **wide
review crop**, which is what the review app shows around a detection. Without it
the app's context view is part-blank for most polygons
(`post-inference/review_campaign.md` §4.1).

Usage (Banks):
    python scripts/build_rgb_chips.py \
        --gpkg banks_rts.gpkg --tile-list banks_tiles.csv \
        --quad-index quad_index_banks_usc1.csv \
        --out-dir /local/banks/products/rgb_chips

Usage (full South):
    python scripts/build_rgb_chips.py \
        --gpkg south_rts.gpkg --tile-list tiles_2025q3_domain_full.csv \
        --quad-index quad_index_2025q3.csv \
        --out-dir /local/south/products/rgb_chips

Usage (review context — no tile list needed, the grid is arithmetic):
    python scripts/build_rgb_chips.py \
        --gpkg south_rts_candidates.gpkg --wide-context \
        --quad-index quad_index_2025q3.csv \
        --out-dir /local/south/review --workers 64
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.quad_index import (RESOLUTION_M, WORLD_MIN,  # noqa: E402
                                  load_quad_index)
from inference.tiles import TILE_SIZE_PX, read_tile  # noqa: E402
from review.crops import crop_bounds  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

import logging  # noqa: E402

logger = logging.getLogger(__name__)


def collect_flagged_tile_ids(gpkg_path: str) -> set[str]:
    """Dedupe the comma-separated `tile_ids` column across every RTS polygon."""
    gdf = gpd.read_file(gpkg_path)
    ids: set[str] = set()
    for cell in gdf["tile_ids"]:
        ids.update(t for t in str(cell).split(",") if t)
    return ids


def context_tile_bboxes(gpkg_path: str, stride_px: int,
                        scale: float = 1.0) -> pd.DataFrame:
    """Tiles covering every polygon's **wide review crop**, not just the polygon.

    :func:`collect_flagged_tile_ids` answers "which tiles does this detection
    touch", which is what the ArcGIS QC package wanted. The review app asks a
    different question: what does the ground *around* the detection look like.
    The wide crop is up to 10x the feature, so it routinely reaches tiles no
    detection sits in, and those were never chipped — 502,555 of them across the
    2025q3 inventory, which is why context views rendered part-blank.

    Bounds come from the stride grid rather than the domain tile list
    (:func:`build_tile_bboxes`), because a context window legitimately runs past
    the inference AOI. Where no quad covers a tile, ``read_tile`` returns NoData
    and the crop renderer stripes it as "NO IMAGERY".

    Args:
        gpkg_path: the candidates gpkg, EPSG:3857.
        stride_px: tile-grid stride, `configs/deployment.yaml inference.stride_px`.
        scale: detection scale, as in `generate_tile_grid`.

    Returns:
        Columns ``tile_id, minx, miny, maxx, maxy`` — the same shape
        :func:`build_tile_bboxes` returns, one row per tile.
    """
    import math

    stride_m = stride_px * RESOLUTION_M / scale
    tile_m = TILE_SIZE_PX * RESOLUTION_M / scale
    gdf = gpd.read_file(gpkg_path)
    seen: set[tuple[int, int]] = set()
    for geom in gdf.geometry:
        _, wide = crop_bounds(geom.bounds)
        c0 = math.floor((wide[0] - WORLD_MIN - tile_m) / stride_m) + 1
        c1 = math.floor((wide[2] - WORLD_MIN) / stride_m)
        r0 = math.floor((wide[1] - WORLD_MIN - tile_m) / stride_m) + 1
        r1 = math.floor((wide[3] - WORLD_MIN) / stride_m)
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                seen.add((c, r))
    rows = [{"tile_id": f"t{c}_{r}",
             "minx": WORLD_MIN + c * stride_m, "miny": WORLD_MIN + r * stride_m,
             "maxx": WORLD_MIN + c * stride_m + tile_m,
             "maxy": WORLD_MIN + r * stride_m + tile_m}
            for c, r in sorted(seen)]
    logger.info("%d polygons -> %d tiles covering their wide review crops",
                len(gdf), len(rows))
    return pd.DataFrame(rows)


def build_tile_bboxes(tile_ids: set[str], tile_list_path: str) -> pd.DataFrame:
    """Look up (minx, miny, maxx, maxy) for each flagged tile_id.

    Raises if a tile_id referenced by the gpkg isn't in the tile list — that's
    a data-integrity mismatch worth surfacing, not silently dropping coverage.
    """
    tiles = pd.read_csv(tile_list_path)
    out = tiles[tiles["tile_id"].isin(tile_ids)]
    missing = tile_ids - set(out["tile_id"])
    if missing:
        raise ValueError(f"tile_ids not found in {tile_list_path}: {sorted(missing)}")
    return out


def write_rgb_chip(tile_id: str, bbox: tuple[float, float, float, float],
                   quad_index: pd.DataFrame, out_path: str) -> None:
    """Window RGB for one tile off the quads (inference.tiles.read_tile) and
    write it as a small georeferenced uint8 GeoTIFF (NoData pixels -> 0)."""
    rgb, nodata = read_tile(bbox, quad_index, TILE_SIZE_PX)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb_u8[:, nodata] = 0
    h, w = rgb_u8.shape[1:]
    transform = transform_from_bounds(*bbox, w, h)
    profile = dict(driver="GTiff", compress="deflate", tiled=True,
                   blockxsize=256, blockysize=256, crs="EPSG:3857",
                   height=h, width=w, count=3, dtype="uint8",
                   nodata=0, transform=transform)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: a run interrupted mid-write must not leave a truncated
    # chip behind, because the resume path skips any tile whose file exists.
    tmp = out.with_suffix(".tif.partial")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(rgb_u8)
    tmp.replace(out)


_QUAD_INDEX = None  # per-process quad index, loaded in the pool initializer


def _init_worker(quad_index_path: str) -> None:
    global _QUAD_INDEX
    _QUAD_INDEX = load_quad_index(quad_index_path)


def _write_one(job: tuple[str, tuple, str]) -> str | None:
    """Write one chip. Returns an error string, or None on success."""
    tile_id, bbox, out_path = job
    try:
        write_rgb_chip(tile_id, bbox, _QUAD_INDEX, out_path)
        return None
    except Exception as exc:  # noqa: BLE001 - one bad tile must not kill the run
        return f"{tile_id}: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpkg", required=True, help="region rts.gpkg (detections)")
    p.add_argument("--tile-list",
                   help="domain tile grid CSV; required unless --wide-context")
    p.add_argument("--wide-context", action="store_true",
                   help="chip the tiles covering each polygon's WIDE review "
                        "crop instead of only the tiles it overlaps, so the "
                        "review app's context view is not part-blank")
    p.add_argument("--config", default="configs/deployment.yaml",
                   help="source of inference.stride_px for --wide-context")
    p.add_argument("--quad-index", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel quad readers; the work is network-bound, so "
                        "this scales well past the core count")
    p.add_argument("--overwrite", action="store_true",
                   help="re-chip tiles that already exist (default: skip, so "
                        "an interrupted run resumes)")
    args = p.parse_args()
    setup_logging()

    if args.wide_context:
        stride_px = load_config(args.config)["inference"]["stride_px"]
        bboxes = context_tile_bboxes(args.gpkg, stride_px)
    else:
        if not args.tile_list:
            p.error("--tile-list is required unless --wide-context is given")
        tile_ids = collect_flagged_tile_ids(args.gpkg)
        logger.info("%d unique tiles referenced by %s", len(tile_ids), args.gpkg)
        bboxes = build_tile_bboxes(tile_ids, args.tile_list)

    chips_dir = args.out_dir / "rgb_chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in bboxes.itertuples():
        out_path = chips_dir / f"{row.tile_id}.tif"
        if not args.overwrite and out_path.exists():
            continue
        jobs.append((row.tile_id, (row.minx, row.miny, row.maxx, row.maxy),
                     str(out_path)))
    logger.info("%d chips to write, %d already present", len(jobs),
                len(bboxes) - len(jobs))

    errors = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(args.quad_index,)) as pool:
            for i, err in enumerate(pool.map(_write_one, jobs, chunksize=16), 1):
                if err:
                    errors.append(err)
                if i % 5000 == 0:
                    logger.info("%d/%d chips (%d errors)", i, len(jobs),
                                len(errors))
    logger.info("Wrote %d RGB chips to %s (%d errors)",
                len(jobs) - len(errors), chips_dir, len(errors))
    for err in errors[:10]:
        logger.warning("chip failed — %s", err)

    # A file list, not argv: at full-inventory scale the chip count (>100k)
    # overflows the command-line length limit.
    vrt_path = args.out_dir / "rgb_chips.vrt"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(sorted(str(c) for c in chips_dir.glob("*.tif"))))
        list_path = fh.name
    subprocess.run(["gdalbuildvrt", "-input_file_list", list_path,
                    str(vrt_path)], check=True, stdout=subprocess.DEVNULL)
    Path(list_path).unlink()
    logger.info("Wrote %s", vrt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
