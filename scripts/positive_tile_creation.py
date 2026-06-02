"""
Positive Training Tile Creation Script

Outputs: - RGB tiles in gs://{BUCKET}/{RGB_PREFIX} as 3-band GeoTIFFs
         - Label tiles in gs://{BUCKET}/{LABELS_PREFIX} as single-band GeoTIFFs
         - metadata.csv in gs://{BUCKET}/{METADATA_PREFIX} with columns: Tile_ID, centroid_lat, centroid_lon, TrainClass, RegionName, UIDs

More info on metadata columns and downstream use can be foudn in the data.md documentation file.

Output CRS is 3857 but centrioids in metadata are in WGS84 (4326) for easier use with geohash UIDs and region lookup.       

Geohashing is used to create compact, reversible UIDs for each tile based on the centroid lat/lon. This allows for
a more flexible addition of more data without UID mapping file.
"""


import os
import sys
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import box, Point
from shapely.ops import transform as shapely_transform
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

BUCKET                = require_env("BUCKET")
INPUT_PREFIX          = require_env("INPUT_PREFIX")
DATA_ROOT             = require_env("DATA_ROOT")
POSITIVE_GEOJSON_BLOB = require_env("POSITIVE_GEOJSON")
IGNORE_GEOJSON_BLOB   = require_env("IGNORE_GEOJSON")
METADATA_SUBREGIONS   = require_env("METADATA_SUBREGIONS")
WORK_DIR              = require_env("WORK_DIR")
MAX_WORKERS           = int(require_env("MAX_WORKERS"))

_test_limit = os.environ.get("TEST_LIMIT")
TEST_LIMIT = int(_test_limit) if _test_limit else None

RGB_PREFIX       = f"{DATA_ROOT}PLANET-RGB/"
LABELS_PREFIX    = f"{DATA_ROOT}labels/"
METADATA_PREFIX  = f"{DATA_ROOT}"
METADATA_COLUMNS = ["Tile_ID", "centroid_lat", "centroid_lon", "TrainClass", "RegionName", "UIDs"]

EQUAL_AREA_CRS = "EPSG:6933"

auth.authenticate_user()

os.makedirs(f"{WORK_DIR}/input",  exist_ok=True)
os.makedirs(f"{WORK_DIR}/output", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)

positive_local = f"{WORK_DIR}/input/positive.geojson"
ignore_local   = f"{WORK_DIR}/input/ignore.geojson"
regions_local  = f"{WORK_DIR}/input/regions.geojson"

bucket.blob(POSITIVE_GEOJSON_BLOB).download_to_filename(positive_local)
bucket.blob(IGNORE_GEOJSON_BLOB).download_to_filename(ignore_local)
bucket.blob(METADATA_SUBREGIONS).download_to_filename(regions_local)

gdf_positive = gpd.read_file(positive_local)
gdf_ignore   = gpd.read_file(ignore_local)
gdf_regions  = gpd.read_file(regions_local)

if "ECO_NAME" not in gdf_regions.columns:
    print(f"ERROR: 'ECO_NAME' column not found in regions GeoJSON. Available: {list(gdf_regions.columns)}")
    sys.exit(1)

gdf_regions_ea   = gdf_regions.to_crs(EQUAL_AREA_CRS)
region_centroids = gdf_regions_ea.geometry.centroid


def find_nearest_region(tile_centroid_wgs84):
    tile_centroid_ea = gpd.GeoSeries([tile_centroid_wgs84], crs="EPSG:4326").to_crs(EQUAL_AREA_CRS).iloc[0]
    distances        = region_centroids.distance(tile_centroid_ea)
    return gdf_regions_ea.loc[distances.idxmin(), "ECO_NAME"]


all_tile_blobs = [
    blob.name
    for blob in bucket.list_blobs(prefix=INPUT_PREFIX)
    if blob.name.endswith(".tif")
][:TEST_LIMIT]

print(f"Found {len(all_tile_blobs)} tiles in input prefix")

if not all_tile_blobs:
    print("No tiles found — check INPUT_PREFIX and BUCKET")
    sys.exit(0)


# UID derivation ------------------------------------------------------
# Tile_ID is a 12-character geohash of the tile centroid (lat, lon).
# Geohash is a compact base-32 encoding that gives ~37 mm precision at
# 12 characters and is fully reversible to lat/lon without the CSV.
# Standard alphabet: 0-9 b-z (excluding a, i, l, o to avoid confusion).
#
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

# Resume support -----------------------------

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

# All blobs are queued; the centroid check in the result loop handles skipping.
tiles_to_process = all_tile_blobs

print(f"{len(done_centroids)} tiles already in metadata, {len(tiles_to_process)} tasks queued")

if not tiles_to_process:
    print("All tiles already processed")
    sys.exit(0)


# -- Worker -------------------------------------------------

def worker_init(positive_path, ignore_path):
    global gdf_positive, gdf_ignore
    gdf_positive = gpd.read_file(positive_path)
    gdf_ignore   = gpd.read_file(ignore_path)


def process_single_tile(blob_path, bucket_name, work_dir):
    base_name       = blob_path.replace("/", "_").replace(".tif", "")
    local_input     = f"{work_dir}/input/{base_name}.tif"
    local_rgb_out   = f"{work_dir}/output/rgb_{base_name}.tif"
    local_label_out = f"{work_dir}/output/label_{base_name}.tif"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    try:
        bucket.blob(blob_path).download_to_filename(local_input)

        with rasterio.open(local_input) as src:
            tile_width     = src.width
            tile_height    = src.height
            tile_crs       = src.crs
            tile_transform = src.transform
            tile_nodata    = src.nodata
            rgb_data = src.read(indexes=[1, 2, 3])

            tile_bounds = rasterio.transform.array_bounds(tile_height, tile_width, tile_transform)
            tile_bbox   = box(*tile_bounds)

            def subset_to_tile(gdf):
                if gdf.crs != tile_crs:
                    gdf = gdf.to_crs(tile_crs)
                gdf = gdf.copy()
                gdf["geometry"] = gdf["geometry"].buffer(0)
                return gdf[gdf.intersects(tile_bbox)]

            raster_kwargs = dict(out_shape=(tile_height, tile_width), transform=tile_transform, fill=0, dtype=np.uint8)

            new_mask = np.zeros((tile_height, tile_width), dtype=np.uint8)

            pos_subset    = subset_to_tile(gdf_positive)
            ignore_subset = subset_to_tile(gdf_ignore)

            if len(ignore_subset) > 0:
                new_mask = rasterize([(geom, 255) for geom in ignore_subset.geometry], **raster_kwargs)
            if len(pos_subset) > 0:
                pos_raster = rasterize([(geom, 1) for geom in pos_subset.geometry], **raster_kwargs)
                new_mask[pos_raster == 1] = 1

            if new_mask.max() == 0:
                return None

            project = pyproj.Transformer.from_crs(tile_crs.to_epsg(), 4326, always_xy=True).transform
            tile_centroid_wgs84 = shapely_transform(project, tile_bbox.centroid)
            centroid_lon, centroid_lat = tile_centroid_wgs84.x, tile_centroid_wgs84.y

        base_profile = dict(
            driver="GTiff", width=tile_width, 
            height=tile_height,
            crs=tile_crs, 
            transform=tile_transform, 
            compress="LZW",
            photometric="RGB"
        )
        if tile_nodata is not None:
            base_profile["nodata"] = tile_nodata

        with rasterio.open(local_rgb_out, "w", **{**base_profile, "dtype": rgb_data.dtype, "count": 3}) as dst:
            dst.write(rgb_data)
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")

        with rasterio.open(local_label_out, "w", **{**base_profile, "dtype": np.uint8, "count": 1}) as dst:
            dst.write(new_mask[np.newaxis, :, :])
            dst.set_band_description(1, "Mask: 0=background 1=positive 255=ignore")

        return (local_rgb_out, local_label_out, centroid_lat, centroid_lon)

    finally:
        if os.path.exists(local_input):
            os.remove(local_input)


# Run --------------------------------------------

print(f"Processing {len(tiles_to_process)} tiles with {MAX_WORKERS} workers\n")

success_count = 0
skip_count    = 0
error_count   = 0
metadata_rows = []

with concurrent.futures.ProcessPoolExecutor(
    max_workers=MAX_WORKERS,
    initializer=worker_init,
    initargs=(positive_local, ignore_local)
) as executor:

    future_to_blob = {
        executor.submit(process_single_tile, blob_path, BUCKET, WORK_DIR): blob_path
        for blob_path in tiles_to_process
    }

    with tqdm(total=len(tiles_to_process), desc="processing tiles", unit="tile") as pbar:
        for future in concurrent.futures.as_completed(future_to_blob):
            blob_path = future_to_blob[future]
            try:
                result = future.result()
                if result is not None:
                    local_rgb, local_label, centroid_lat, centroid_lon = result
                    centroid_key = (round(centroid_lat, 6), round(centroid_lon, 6))

                    if centroid_key in done_centroids:
                        skip_count += 1
                        tqdm.write(f" already done  {blob_path.split('/')[-1]}  (centroid match)")
                        os.remove(local_rgb)
                        os.remove(local_label)
                    else:
                        uid_str = make_tile_uid(centroid_key[0], centroid_key[1])

                        bucket.blob(f"{RGB_PREFIX}{uid_str}.tif").upload_from_filename(local_rgb)
                        bucket.blob(f"{LABELS_PREFIX}{uid_str}.tif").upload_from_filename(local_label)
                        os.remove(local_rgb)
                        os.remove(local_label)

                        region_name = find_nearest_region(Point(centroid_lon, centroid_lat))

                        metadata_rows.append({
                            "Tile_ID":      uid_str,
                            "centroid_lat": centroid_key[0],
                            "centroid_lon": centroid_key[1],
                            "TrainClass":   "positive",
                            "RegionName":   region_name,
                            "UIDs":         9999,
                        })

                        done_centroids.add(centroid_key)
                        success_count += 1
                else:
                    skip_count += 1
            except Exception as exc:
                error_count += 1
                tqdm.write(f"ERROR: {blob_path.split('/')[-1]} — {exc}")
            pbar.update(1)


# Write metadata ----------------------------------------------------------

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
    print("No successful tiles — metadata CSV not written")

print(f"\n{success_count} written, {skip_count} skipped, {error_count} errors")
print(f"RGB:      gs://{BUCKET}/{RGB_PREFIX}")
print(f"Labels:   gs://{BUCKET}/{LABELS_PREFIX}")