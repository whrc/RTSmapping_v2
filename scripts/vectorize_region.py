"""Parallel region vectorization (post-inference.md §9.3 at region scale).

`vectorize_predictions.py` polygonizes one merged mask that must fit in RAM (the
GDAL polygonize scan is single-threaded); a full region's mask does not fit
(South ≈ 7.7 TB). This vectorizes the per-block mask COGs written by
`assemble_region.py` **in parallel**, then stitches them:

  - a polygon fully interior to its block is final (min_blob filter applied in
    the worker — cheap, and it can never grow by a seam merge);
  - a polygon touching a block edge is deferred; all edge polygons are then
    dissolved (`unary_union`) so a slump split across a block seam becomes one
    polygon, and min_blob is applied **after** the dissolve (a slump split across
    a seam can be sub-min_blob in each half but ≥ min_blob combined).

min_blob uses the exact pixel count `area_3857 / pixel_area` (polygonized edges
are pixel-aligned, and the projection distortion cancels since both areas are in
the same EPSG:3857 grid). Per-polygon stats match `vectorize_predictions`.

Usage:
    python scripts/vectorize_region.py \
        --blocks-dir /local/banks/out/blocks \
        --prob /local/banks/out/probability.tif \
        --tile-list banks_tiles.csv --package pkg_dir \
        --output banks_rts.gpkg --workers 32
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from math import ceil, floor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features, windows
from shapely import wkb as shp_wkb
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.writer import NODATA_SCALED_U8, SCALE_U8  # noqa: E402
from scripts.vectorize_predictions import _GEOD, _TO_WGS84  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _polygonize_block(spec: tuple[str, int]):
    """Polygonize one block mask → (interior WKB list, edge-touching WKB list).

    Interior polygons are min_blob-filtered here; edge-touching ones are deferred
    to the cross-block dissolve. Returned as WKB so they pickle across the pool.
    """
    mask_path, min_blob_px = spec
    with rasterio.open(mask_path) as src:
        m = src.read(1)
        transform = src.transform
        left, bottom, right, top = src.bounds
        xres, yres = src.res
    binm = m == 1
    if not binm.any():
        return [], []
    px_area = xres * yres
    tol = xres  # ~1 px: a polygon reaching the block boundary may continue next door
    interior, edge = [], []
    for gj, _ in features.shapes(binm.astype("uint8"), mask=binm, transform=transform):
        geom = shape(gj)
        b = geom.bounds
        on_edge = (b[0] <= left + tol or b[1] <= bottom + tol
                   or b[2] >= right - tol or b[3] >= top - tol)
        if on_edge:
            edge.append(geom.wkb)
        elif geom.area >= min_blob_px * px_area:
            interior.append(geom.wkb)
    return interior, edge


def _record(rts_id: int, geom, prb, tiles: pd.DataFrame, scales: list[float]) -> dict:
    """Per-polygon §9.3 attributes (mirrors vectorize_predictions.vectorize)."""
    transform = prb.transform
    fwin = windows.from_bounds(*geom.bounds, transform=transform)
    c0, r0 = floor(fwin.col_off), floor(fwin.row_off)
    c1, r1 = ceil(fwin.col_off + fwin.width), ceil(fwin.row_off + fwin.height)
    win = windows.Window(c0, r0, c1 - c0, r1 - r0)
    local = features.rasterize([mapping(geom)], out_shape=(r1 - r0, c1 - c0),
                               transform=windows.transform(win, transform))
    pvals = prb.read(1, window=win)[local == 1]
    if prb.dtypes[0] == "uint8":  # scaled_uint8 product: prob×250, NoData 255
        pvals = pvals[pvals != NODATA_SCALED_U8].astype(np.float32) / SCALE_U8
    else:  # float32 product: prob in [0,1], NoData -1
        pvals = pvals[pvals >= 0]
    geom_wgs = gpd.GeoSeries([geom], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
    area, perim = _GEOD.geometry_area_perimeter(geom_wgs)
    lon, lat = _TO_WGS84.transform(geom.centroid.x, geom.centroid.y)
    b = geom.bounds
    hit = tiles[(tiles["minx"] < b[2]) & (tiles["maxx"] > b[0])
                & (tiles["miny"] < b[3]) & (tiles["maxy"] > b[1])]
    return {
        "rts_id": rts_id,
        "area_m2": abs(area),
        "perimeter_m": perim,
        "centroid_lat": lat,
        "centroid_lon": lon,
        "mean_prob": float(pvals.mean()) if pvals.size else float("nan"),
        "max_prob": float(pvals.max()) if pvals.size else float("nan"),
        "detection_scale": ",".join(str(s) for s in scales),
        "tile_ids": ",".join(hit["tile_id"].astype(str)),
    }


def dissolve_edges(edge_wkb: list[bytes], min_blob_px: int,
                   px_area: float) -> list:
    """Union edge-touching polygons across seams, then min_blob-filter."""
    if not edge_wkb:
        return []
    merged = unary_union([shp_wkb.loads(w) for w in edge_wkb])
    polys = list(merged.geoms) if merged.geom_type.startswith("Multi") else [merged]
    return [g for g in polys if g.area >= min_blob_px * px_area]


def vectorize_region(blocks_dir: str, prob_path: str, tile_list: str,
                     scales: list[float], min_blob_px: int,
                     workers: int) -> gpd.GeoDataFrame:
    """Parallel polygonize of block masks → dissolved, min_blob-filtered polygons."""
    mask_blocks = sorted(glob.glob(f"{blocks_dir.rstrip('/')}/mask_*.tif"))
    if not mask_blocks:
        raise RuntimeError(f"no mask_*.tif under {blocks_dir}")
    specs = [(mb, min_blob_px) for mb in mask_blocks]

    interior_wkb, edge_wkb = [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for inter, edge in ex.map(_polygonize_block, specs):
            interior_wkb += inter
            edge_wkb += edge
    logger.info("polygonized %d blocks: %d interior + %d edge-touching",
                len(specs), len(interior_wkb), len(edge_wkb))

    prb = rasterio.open(prob_path)
    px_area = abs(prb.res[0] * prb.res[1])
    dissolved = dissolve_edges(edge_wkb, min_blob_px, px_area)
    final = [shp_wkb.loads(w) for w in interior_wkb] + dissolved
    logger.info("%d final polygons (%d interior + %d dissolved-edge)",
                len(final), len(interior_wkb), len(dissolved))

    tiles = pd.read_csv(tile_list)
    records = [_record(i, g, prb, tiles, scales) for i, g in enumerate(final, 1)]
    prb.close()
    gdf = gpd.GeoDataFrame(records, geometry=final, crs="EPSG:3857")
    logger.info("Vectorized %d polygons, total %.2f km2", len(gdf),
                gdf["area_m2"].sum() / 1e6 if len(gdf) else 0.0)
    return gdf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blocks-dir", required=True,
                   help="dir of assemble_region.py mask_*.tif block COGs")
    p.add_argument("--prob", required=True, help="region probability COG")
    p.add_argument("--tile-list", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    setup_logging()

    dep_cfg = load_config(f"{str(args.package).rstrip('/')}/deployment_config.yaml")
    gdf = vectorize_region(args.blocks_dir, args.prob, args.tile_list,
                           scales=dep_cfg.get("scales", [1.0]),
                           min_blob_px=int(dep_cfg.get("min_blob_size_px", 0)),
                           workers=args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(args.output, driver="GPKG")
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
