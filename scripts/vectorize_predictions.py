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
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Geod, Transformer
from rasterio import features
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

_TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
_GEOD = Geod(ellps="WGS84")


def vectorize(mask_path: str, prob_path: str, tile_list_path: str,
              scales: list[float]) -> gpd.GeoDataFrame:
    """Build the §9.3 polygon layer from mask + probability rasters."""
    import pandas as pd

    with rasterio.open(mask_path) as msk:
        mask = msk.read(1)
        transform = msk.transform
    with rasterio.open(prob_path) as prb:
        probs = prb.read(1)

    tiles = pd.read_csv(tile_list_path)
    records = []
    geoms = []
    shapes = features.shapes((mask == 1).astype(np.uint8), mask=(mask == 1),
                             transform=transform)
    for rts_id, (geom_json, _) in enumerate(shapes, start=1):
        geom = shape(geom_json)
        # Pixel stats: rasterize this polygon's window only.
        win = rasterio.windows.from_bounds(*geom.bounds, transform=transform)
        win = win.round_offsets("floor").round_lengths("ceil")
        local = features.rasterize([geom_json], out_shape=(int(win.height), int(win.width)),
                                   transform=rasterio.windows.transform(win, transform))
        r0, c0 = int(win.row_off), int(win.col_off)
        pvals = probs[r0:r0 + int(win.height),
                      c0:c0 + int(win.width)][local == 1]
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
                    scales=dep_cfg.get("scales", [1.0]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(args.output, driver="GPKG")
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
