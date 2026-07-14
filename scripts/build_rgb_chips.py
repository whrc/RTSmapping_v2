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
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds as transform_from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.quad_index import load_quad_index  # noqa: E402
from inference.tiles import TILE_SIZE_PX, read_tile  # noqa: E402
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(rgb_u8)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpkg", required=True, help="region rts.gpkg (detections)")
    p.add_argument("--tile-list", required=True)
    p.add_argument("--quad-index", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()
    setup_logging()

    tile_ids = collect_flagged_tile_ids(args.gpkg)
    logger.info("%d unique tiles referenced by %s", len(tile_ids), args.gpkg)
    bboxes = build_tile_bboxes(tile_ids, args.tile_list)
    quad_index = load_quad_index(args.quad_index)

    chips_dir = args.out_dir / "rgb_chips"
    for _, row in bboxes.iterrows():
        bbox = (row["minx"], row["miny"], row["maxx"], row["maxy"])
        write_rgb_chip(row["tile_id"], bbox, quad_index,
                       str(chips_dir / f"{row['tile_id']}.tif"))
    logger.info("Wrote %d RGB chips to %s", len(bboxes), chips_dir)

    vrt_path = args.out_dir / "rgb_chips.vrt"
    subprocess.run(["gdalbuildvrt", str(vrt_path), *sorted(
        str(p) for p in chips_dir.glob("*.tif"))], check=True,
                   stdout=subprocess.DEVNULL)
    logger.info("Wrote %s", vrt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
