"""Mask the 2025 tile grid to the inference domain (circumpolar_south_domain.geojson).

Keeps tiles whose centroid (EPSG:3857) falls inside the domain polygon. Streams the
33M-row tile CSV in chunks to bound memory.
"""
import sys, time
import numpy as np, pandas as pd, geopandas as gpd, shapely

DOMAIN = "/app/domain/circumpolar_south_domain.geojson"
TILES = sys.argv[1] if len(sys.argv) > 1 else "/outputs/inference/tiles_2025q3.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/outputs/inference/tiles_2025q3_domain.csv"
CHUNK = 2_000_000

t0 = time.time()
dom = gpd.read_file(DOMAIN).to_crs(3857)
geom = dom.union_all() if hasattr(dom, "union_all") else dom.unary_union
print(f"domain loaded + reprojected to 3857, area_frac bbox: {geom.bounds}", flush=True)

kept = 0
total = 0
first = True
for chunk in pd.read_csv(TILES, chunksize=CHUNK):
    cx = (chunk["minx"].to_numpy() + chunk["maxx"].to_numpy()) / 2.0
    cy = (chunk["miny"].to_numpy() + chunk["maxy"].to_numpy()) / 2.0
    inside = shapely.contains_xy(geom, cx, cy)
    sub = chunk[inside]
    sub.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    kept += len(sub); total += len(chunk)
    print(f"  ... {total:,} scanned, {kept:,} kept ({time.time()-t0:.0f}s)", flush=True)

print(f"DONE: {kept:,} / {total:,} tiles inside domain ({100*kept/total:.1f}%)  -> {OUT}")
