"""Vectorize a merged binary mask into RTS polygons (inference.md §9.3).

Polygonizes mask==1 regions and attaches the spec'd attributes (geodesic
area/perimeter, WGS84 centroid, mean/max probability from the merged
probability raster, detecting scales, intersecting tile ids).

Usage:
    python scripts/vectorize_predictions.py \
        --mask merged_mask.tif --prob merged_prob.tif \
        --tile-list tiles.csv --package gs://.../rts-v2-seed42 \
        --output rts_predictions.gpkg
"""

from __future__ import annotations

import argparse
import logging
import sys
from math import ceil, floor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Geod, Transformer
from rasterio import features
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config, vectorize_min_blob_px  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
_GEOD = Geod(ellps="WGS84")


def vectorize(mask_path: str, prob_path: str, tile_list_path: str,
              scales: list[float], min_blob_px: int = 0) -> gpd.GeoDataFrame:
    """Build the §9.3 polygon layer from mask + probability rasters.

    Blobs smaller than ``min_blob_px`` pixels are dropped (the deployment
    ``vectorize_min_blob_px`` object-decision filter — a vectorization-stage
    param in PIXELS, see configs/deployment.yaml; distinct from the eval-stage
    ``metrics.min_blob_size_px`` and from the shipped geodesic ``--min-area-m2``
    MMU in m²). Probability pixel-stats are read windowed from
    the COG per polygon, so this scales to region rasters far larger than RAM.
    """
    import pandas as pd

    with rasterio.open(mask_path) as msk:
        mask = msk.read(1)
        transform = msk.transform

    tiles = pd.read_csv(tile_list_path)
    records = []
    geoms = []
    binmask = mask == 1
    shapes = features.shapes(binmask.astype(np.uint8), mask=binmask,
                             transform=transform)
    prb = rasterio.open(prob_path)
    rts_id = 0
    for geom_json, _ in shapes:
        geom = shape(geom_json)
        # Pixel stats: rasterize this polygon's window only. Round to an integer
        # window that fully covers the float window (floor the near edge, ceil
        # the far edge) — version-robust vs rasterio's Window.round_* API.
        fwin = rasterio.windows.from_bounds(*geom.bounds, transform=transform)
        c0, r0 = floor(fwin.col_off), floor(fwin.row_off)
        c1 = ceil(fwin.col_off + fwin.width)
        r1 = ceil(fwin.row_off + fwin.height)
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        local = features.rasterize([geom_json], out_shape=(int(win.height), int(win.width)),
                                   transform=rasterio.windows.transform(win, transform))
        blob_px = int((local == 1).sum())
        if blob_px < min_blob_px:
            continue  # sub-min_blob_size object (speckle FP)
        rts_id += 1
        pvals = prb.read(1, window=win)[local == 1]
        pvals = pvals[pvals >= 0]  # drop NoData

        geom_wgs = gpd.GeoSeries([geom], crs="EPSG:3857").to_crs("EPSG:4326").iloc[0]
        area, perim = _GEOD.geometry_area_perimeter(geom_wgs)
        lon, lat = _TO_WGS84.transform(geom.centroid.x, geom.centroid.y)
        b = geom.bounds
        hit = tiles[(tiles["minx"] < b[2]) & (tiles["maxx"] > b[0])
                    & (tiles["miny"] < b[3]) & (tiles["maxy"] > b[1])]
        records.append({
            "rts_id": rts_id,
            "area_m2": abs(area),
            "perimeter_m": perim,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "mean_prob": float(pvals.mean()) if pvals.size else float("nan"),
            "max_prob": float(pvals.max()) if pvals.size else float("nan"),
            "detection_scale": ",".join(str(s) for s in scales),
            "tile_ids": ",".join(hit["tile_id"].astype(str)),
        })
        geoms.append(geom)
    prb.close()

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:3857")
    logger.info("Vectorized %d polygons, total %.2f km2",
                len(gdf), gdf["area_m2"].sum() / 1e6 if len(gdf) else 0.0)
    return gdf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mask", required=True)
    p.add_argument("--prob", required=True)
    p.add_argument("--tile-list", required=True)
    p.add_argument("--package", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    setup_logging()

    dep_cfg = load_config(f"{str(args.package).rstrip('/')}/deployment_config.yaml")
    gdf = vectorize(args.mask, args.prob, args.tile_list,
                    scales=dep_cfg.get("scales", [1.0]),
                    min_blob_px=vectorize_min_blob_px(dep_cfg))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(args.output, driver="GPKG")
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
