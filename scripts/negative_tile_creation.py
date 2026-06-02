"""
Negative Training Tile Creation Script

Outputs: - RGB tiles in gs://{BUCKET}/{RGB_PREFIX} as 3-band GeoTIFFs
         - Label tiles in gs://{BUCKET}/{LABELS_PREFIX} as single-band GeoTIFFs
         - metadata.csv in gs://{BUCKET}/{METADATA_PREFIX} with columns: "Tile_ID, centroid_lat, centroid_lon, TrainClass, RegionName, UIDs

Output CRS is 3857 from the original tiles

Geohashing is used to create compact, reversible UIDs for each tile based on the centroid lat/lon. This allows for
a more flexible addition of more data without UID mapping file.
"""

import os
import sys
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
import geopandas as gpd
from shapely.geometry import Point, box
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree
from google.colab import auth
from google.cloud import storage
import concurrent.futures
from tqdm import tqdm
import pandas as pd
import pyproj


def require_env(name):
    val = os.environ.get(name)
    if val is None:
        print(f"ERROR: Required environment variable '{name}' is not set.")
        sys.exit(1)
    return val

BUCKET               = require_env("BUCKET")
POLYGON_GEOJSON_BLOB = require_env("POLYGON_GEOJSON_BLOB")
GRID_GEOJSON_BLOB    = require_env("GRID_GEOJSON_BLOB")
DATA_ROOT            = require_env("DATA_ROOT")
METADATA_SUBREGIONS  = require_env("METADATA_SUBREGIONS")
WORK_DIR             = require_env("WORK_DIR")
MAX_WORKERS          = int(require_env("MAX_WORKERS"))

_target      = os.environ.get("TARGET_TILES")
TARGET_TILES = int(_target) if _target else None

TILE_SIZE        = 512
WORKING_CRS      = "EPSG:6933" # This is for more accurate centriod placement for ecoregions
RGB_PREFIX       = f"{DATA_ROOT}/PLANET-RGB/"
METADATA_PREFIX  = f"{DATA_ROOT}/"
METADATA_COLUMNS = ["Tile_ID", "centroid_lat", "centroid_lon", "TrainClass", "RegionName", "UIDs"]

print(f"BUCKET: {BUCKET} | workers: {MAX_WORKERS} | target: {TARGET_TILES or 'all'}")

auth.authenticate_user()

os.makedirs(f"{WORK_DIR}/input",  exist_ok=True)
os.makedirs(f"{WORK_DIR}/output", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)


# Load datasets --------------------------------------

polygon_local = f"{WORK_DIR}/input/polygons.geojson"
regions_local = f"{WORK_DIR}/input/regions.geojson"
grid_local    = f"{WORK_DIR}/input/grid.geojson"

bucket.blob(POLYGON_GEOJSON_BLOB).download_to_filename(polygon_local)
bucket.blob(METADATA_SUBREGIONS).download_to_filename(regions_local)
bucket.blob(GRID_GEOJSON_BLOB).download_to_filename(grid_local)

gdf_polygons = gpd.read_file(polygon_local)
gdf_polygons = gdf_polygons[gdf_polygons["TrainClass"] == "Negative"].copy()
gdf_polygons = gdf_polygons.to_crs(WORKING_CRS)
gdf_polygons["geometry"] = gdf_polygons["geometry"].buffer(0)

gdf_regions      = gpd.read_file(regions_local)
gdf_regions_work = gdf_regions.to_crs(WORKING_CRS)

if "ECO_NAME" not in gdf_regions.columns:
    print(f"ERROR: 'ECO_NAME' not found. Available: {list(gdf_regions.columns)}")
    sys.exit(1)

gdf_grid = gpd.read_file(grid_local).to_crs(WORKING_CRS)

print(f"Loaded — polygons: {len(gdf_polygons)}, regions: {len(gdf_regions)}, grid: {len(gdf_grid)}")


# Assign ecoregions --------------------------------------------

poly_centroids = gdf_polygons.copy()
poly_centroids["geometry"] = gdf_polygons.geometry.centroid

joined = gpd.sjoin(
    poly_centroids[["geometry"]],
    gdf_regions_work[["geometry", "ECO_NAME"]],
    how="left", predicate="within",
)
gdf_polygons["ECO_NAME"] = joined["ECO_NAME"].values

missing = gdf_polygons["ECO_NAME"].isna()
if missing.any():
    region_tree = STRtree(gdf_regions_work.geometry.centroid.values)
    for idx in gdf_polygons[missing].index:
        nearest_i = region_tree.nearest(gdf_polygons.loc[idx, "geometry"].centroid)
        gdf_polygons.at[idx, "ECO_NAME"] = gdf_regions_work.iloc[nearest_i]["ECO_NAME"]
    print(f"  {missing.sum()} polygons assigned via nearest centroid")


# Stratified sampling -----------------------------------------

eco_groups   = gdf_polygons.groupby("ECO_NAME")
n_ecoregions = len(eco_groups)
per_region   = max(1, TARGET_TILES // n_ecoregions) if TARGET_TILES else None

sampled_parts = []
for _, group in eco_groups:
    if per_region is None or len(group) <= per_region:
        sampled_parts.append(group)
    else:
        sampled_parts.append(group.sample(n=per_region, random_state=42))

gdf_sampled = pd.concat(sampled_parts).reset_index(drop=True)

if TARGET_TILES and len(gdf_sampled) > TARGET_TILES:
    gdf_sampled = gdf_sampled.sample(n=TARGET_TILES, random_state=42).reset_index(drop=True)

print(f"Sampled {len(gdf_sampled)} polygons across {n_ecoregions} ecoregions")

sampled_local = f"{WORK_DIR}/input/polygons_sampled.geojson"
gdf_sampled.to_file(sampled_local, driver="GeoJSON")


# Build tasks -------------------------------------

def grid_row_to_blob(row) -> str:
    delivery = row["delivery_location"].rstrip("/")
    filename = f"{row['basemap_name']}_{row['grid_column']}-{row['grid_row']}_quad.tif"
    return f"{delivery}/{filename}"

grid_sindex   = gdf_grid.sindex
tasks         = []
no_grid_count = 0

for i, poly_row in gdf_sampled.iterrows():
    centroid = poly_row.geometry.centroid

    candidate_idxs = list(grid_sindex.intersection(centroid.bounds))
    covering = [j for j in candidate_idxs if gdf_grid.iloc[j].geometry.contains(centroid)]

    if not covering and candidate_idxs:
        covering = [min(candidate_idxs, key=lambda j: gdf_grid.iloc[j].geometry.distance(centroid))] 

    if not covering:
        no_grid_count += 1
        continue

    grid_row  = gdf_grid.iloc[covering[0]]
    tasks.append({
        "polygon_geom": poly_row.geometry,
        "blob_path":    grid_row_to_blob(grid_row),
        "eco_name":     poly_row.get("ECO_NAME", ""),
    })

print(f"Tasks: {len(tasks)} built, {no_grid_count} polygons had no covering grid cell")


# UID derivation -----------------------
# geohashing the centriod
# Example: lat=39.47, lon=-105.21  ->  9xj0tck3mm3c

_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def make_tile_uid(lat: float, lon: float, precision: int = 12) -> str:
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    bits    = [16, 8, 4, 2, 1]
    bit_idx = 0
    even    = True
    ch      = 0
    result  = []
    while len(result) < precision:
        if even:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                ch |= bits[bit_idx]
                lon_range[0] = mid
            else:
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                ch |= bits[bit_idx]
                lat_range[0] = mid
            else:
                lat_range[1] = mid
        even = not even
        if bit_idx < 4:
            bit_idx += 1
        else:
            result.append(_GEOHASH_BASE32[ch])
            ch      = 0
            bit_idx = 0
    return "".join(result)

# Resume support ---------------------------

metadata_blob_path = f"{METADATA_PREFIX}metadata.csv"
metadata_blob      = bucket.blob(metadata_blob_path)
existing_df        = None
done_centroids     = set()

if metadata_blob.exists():
    existing_local = f"{WORK_DIR}/input/metadata_existing.csv"
    metadata_blob.download_to_filename(existing_local)
    existing_df = pd.read_csv(existing_local, dtype={"Tile_ID": str})

    # Each tile centroid is unique - use (lat, lon) rounded to 4dp as the
    # duplicate key, matching the precision encoded in the UID.
    done_centroids = set(
        zip(
            existing_df["centroid_lat"].round(6),
            existing_df["centroid_lon"].round(6),
        )
    )
    print(f"Found existing metadata: {len(existing_df)} rows, {len(done_centroids)} centroids done")
else:
    print("No existing metadata found - starting fresh")

# Centroids are not known until a tile is opened, so all tasks are queued and te duplicate check happens in the result loop once we have the actual coords.
tasks_to_run = tasks

print(f"{len(done_centroids)} tiles already in metadata, {len(tasks_to_run)} tasks queued")

if not tasks_to_run:
    print("Nothing to do.")
    sys.exit(0)


# Windowing -------------------------------------------------

def get_containing_window(src, poly_bounds_native, tile_size=512):
    minx, miny, maxx, maxy = poly_bounds_native

    tl_row, tl_col = rowcol(src.transform, minx, maxy)
    br_row, br_col = rowcol(src.transform, maxx, miny)
    if abs(br_col - tl_col) > tile_size or abs(br_row - tl_row) > tile_size:
        return None
    if src.width < tile_size or src.height < tile_size:
        return None

    cx, cy           = (minx + maxx) / 2, (miny + maxy) / 2
    center_row, center_col = rowcol(src.transform, cx, cy)

    half    = tile_size // 2
    col_off = max(0, min(int(center_col) - half, src.width  - tile_size))
    row_off = max(0, min(int(center_row) - half, src.height - tile_size))

    return Window(col_off, row_off, tile_size, tile_size)


# Worker ----------------------------------------

def worker_init(sampled_polygon_path: str):
    global _gcs_bucket, _project_to_wgs84

    _gcs_client  = storage.Client()
    _gcs_bucket  = _gcs_client.bucket(BUCKET)


def process_single_tile(task: dict, work_dir: str):
    poly_geom = task["polygon_geom"]
    blob_path = task["blob_path"]

    base_name     = blob_path.split("/")[-1].replace(".tif", "")
    local_input   = f"{work_dir}/input/{base_name}.tif"
    local_rgb_out = f"{work_dir}/output/rgb_{base_name}.tif"

    try:
        _gcs_bucket.blob(blob_path).download_to_filename(local_input)

        with rasterio.open(local_input) as src:
            project_to_native = pyproj.Transformer.from_crs(
                WORKING_CRS, src.crs.to_epsg(), always_xy=True
            ).transform
            poly_native = shapely_transform(project_to_native, poly_geom)

            win = get_containing_window(src, poly_native.bounds, TILE_SIZE)
            if win is None:
                return None

            try:
                rgb_data = src.read(indexes=[1, 2, 3], window=win)
            except Exception:
                return None

            if src.nodata is not None and (rgb_data == src.nodata).all():
                return None

            chip_tf   = rasterio.windows.transform(win, src.transform)
            tile_bbox = box(*rasterio.transform.array_bounds(win.height, win.width, chip_tf))

            # Build transformer from actual raster CRS, not assumed 3857
            project_to_wgs84 = pyproj.Transformer.from_crs(
                src.crs.to_epsg(), 4326, always_xy=True
            ).transform
            centroid  = shapely_transform(project_to_wgs84, tile_bbox.centroid)
            native_crs = src.crs

        profile = dict(
            driver="GTiff", 
            count=3, 
            dtype=rgb_data.dtype,
            width=TILE_SIZE, height=TILE_SIZE,
            crs=native_crs, 
            transform=chip_tf, 
            compress="LZW",
            photometric="RGB",
        )
        with rasterio.open(local_rgb_out, "w", **profile) as dst:
            dst.write(rgb_data)
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")

        return (local_rgb_out, centroid.y, centroid.x)

    finally:
        if os.path.exists(local_input):
            os.remove(local_input)


# Ecoregion lookup (main process) -----------------------------

_region_tree = STRtree(gdf_regions_work.geometry.centroid.values)

def find_nearest_region(centroid_lon: float, centroid_lat: float) -> str:
    projector = pyproj.Transformer.from_crs(4326, WORKING_CRS, always_xy=True).transform
    pt_work   = shapely_transform(projector, Point(centroid_lon, centroid_lat))
    return gdf_regions_work.iloc[_region_tree.nearest(pt_work)]["ECO_NAME"]


# Run ---------------------------------------------------

print(f"\nProcessing {len(tasks_to_run)} polygons with {MAX_WORKERS} workers...\n")

success_count = 0
skip_count    = 0
error_count   = 0
metadata_rows = []

with concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init,
    initargs=(sampled_local,),
) as executor:

    future_to_task = {
        executor.submit(process_single_tile, task, WORK_DIR): task
        for task in tasks_to_run
    }

    with tqdm(total=len(tasks_to_run), desc="tiles", unit="tile") as pbar:
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result is not None:
                    local_rgb, centroid_lat, centroid_lon = result
                    centroid_key = (round(centroid_lat, 6), round(centroid_lon, 6))

                    if centroid_key in done_centroids:
                        skip_count += 1
                        tqdm.write(f" already done  {task['blob_path'].split("/")[-1]}  (centroid match)")
                        os.remove(local_rgb)
                    else:
                        uid_str = make_tile_uid(centroid_key[0], centroid_key[1])

                        bucket.blob(f"{RGB_PREFIX}{uid_str}.tif").upload_from_filename(local_rgb)
                        os.remove(local_rgb)

                        metadata_rows.append({
                            "Tile_ID":      uid_str,
                            "centroid_lat": centroid_key[0],
                            "centroid_lon": centroid_key[1],
                            "TrainClass":   "negative",
                            "RegionName":   find_nearest_region(centroid_lon, centroid_lat),
                            "UIDs":         9999,
                        })

                        done_centroids.add(centroid_key)
                        success_count += 1
                        tqdm.write(f"{uid_str}  {task['blob_path'].split("/")[-1]}")
                else:
                    skip_count += 1
                    tqdm.write(f" skipped  {task['blob_path'].split("/")[-1]}")

            except Exception as exc:
                error_count += 1
                tqdm.write(f"{task['blob_path'].split("/")[-1]} — {exc}")

            pbar.update(1)


# Write metadata -------------------------------------------------

if metadata_rows:
    new_df    = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
    local_csv = f"{WORK_DIR}/output/metadata.csv"

    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.reindex(columns=METADATA_COLUMNS)
    combined.to_csv(local_csv, index=False)

    bucket.blob(metadata_blob_path).upload_from_filename(local_csv)
    os.remove(local_csv)
    print(f"Metadata: {len(combined)} rows > gs://{BUCKET}/{metadata_blob_path}")
else:
    print("No successful tiles — metadata not written.")

print(f"\n{success_count} written | {skip_count} skipped | {error_count} errors")