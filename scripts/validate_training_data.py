"""
Validates output of positive_tile_creation.py and negative_tile_creation.py.

Checks:
  1.  CRS consistency across all RGB and label tiles
  2.  Band order and count (RGB = 3 bands, label = 1 band)
  3.  Band descriptions match expected (Red/Green/Blue)
  4.  Centroid lat/lon are valid WGS84 coordinates
  5.  Centroid lat/lon precision consistency (rounded to 6dp)
  6.  Duplicate Tile_IDs in metadata
  7.  Duplicate centroid (lat, lon) pairs
  8.  UID (Tile_ID) re-derivable from centroid coordinates
  9.  RGB tile pixel values are non-zero / non-nodata
  10. Label tiles contain only valid values (0, 1, 255)
  11. RGB and label tile existence parity (every positive tile should have a label)
  12. Metadata TrainClass values are in expected set
  13. Tile spatial dimensions match expected TILE_SIZE

Environment variables:
  BUCKET      GCS bucket name (required)
  DATA_ROOT   Path to training data root (required)
  WORK_DIR    Local working directory (required)
  SAMPLE_TILES  Validate a random sample of N tiles instead of all (optional)
  SKIP_PIXEL_CHECK  Skip tile download checks entirely (optional)
"""

import os
import sys
import random
import traceback
from collections import defaultdict

import pandas as pd
import numpy as np
import rasterio
from google.cloud import storage

try:
    from google.colab import auth
    auth.authenticate_user()
    print("Authenticated via Colab.")
except ImportError:
    print("Not running in Colab — using default GCS credentials (ADC).")


#Constants 

def require_env(name):
    val = os.environ.get(name)
    if val is None:
        print(f"ERROR: Required environment variable '{name}' is not set.")
        sys.exit(1)
    return val

BUCKET        = require_env("BUCKET")
DATA_ROOT     = require_env("DATA_ROOT").rstrip("/")
WORK_DIR      = require_env("WORK_DIR")

RGB_PREFIX    = f"{DATA_ROOT}/PLANET-RGB/"
LABELS_PREFIX = f"{DATA_ROOT}/labels/"
METADATA_PATH = f"{DATA_ROOT}/metadata.csv"

TILE_SIZE          = 512
EXPECTED_CRS       = "EPSG:3857"
CENTROID_CRS       = "EPSG:4326"
VALID_CLASSES      = {"positive", "negative"}
VALID_LABEL_VALS   = {0, 1, 255}

_sample_env  = os.environ.get("SAMPLE_TILES")
SAMPLE_TILES = int(_sample_env) if _sample_env else None
SKIP_PIXEL   = bool(os.environ.get("SKIP_PIXEL_CHECK"))

os.makedirs(f"{WORK_DIR}/val_tmp", exist_ok=True)

client = storage.Client()
bucket = client.bucket(BUCKET)

issues   = []
warnings = []


def fail(check, msg):
    issues.append(("FAIL", check, msg))
    print(f"  FAIL  [{check}] {msg}")

def warn(check, msg):
    warnings.append(("WARN", check, msg))
    print(f"  WARN  [{check}] {msg}")

def ok(check, msg):
    print(f"  OK    [{check}] {msg}")

def section(title):
    print(f"\n  {title}")


# Geohash (copied verbatim from pipeline) 

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


# Check 1: Metadata file 

section("1. Metadata file")

meta_blob = bucket.blob(METADATA_PATH)
if not meta_blob.exists():
    fail("metadata_exists", f"gs://{BUCKET}/{METADATA_PATH} not found — cannot continue.")
    sys.exit(1)

local_meta = f"{WORK_DIR}/val_tmp/metadata.csv"
meta_blob.download_to_filename(local_meta)
df = pd.read_csv(local_meta, dtype={"Tile_ID": str})

ok("metadata_exists", f"{len(df)} rows loaded from gs://{BUCKET}/{METADATA_PATH}")

required_cols = ["Tile_ID", "centroid_lat", "centroid_lon", "TrainClass", "RegionName", "UIDs"]
missing_cols  = [c for c in required_cols if c not in df.columns]
if missing_cols:
    fail("metadata_columns", f"Missing columns: {missing_cols}")
else:
    ok("metadata_columns", f"All required columns present: {required_cols}")

print(f"\n  Column dtypes:\n{df.dtypes.to_string()}")
print(f"\n  TrainClass counts:\n{df['TrainClass'].value_counts().to_string()}")
print(f"\n  UIDs value counts (top 5):\n{df['UIDs'].value_counts().head().to_string()}")


# Check 2: Duplicate Tile_IDs

section("2. Duplicate Tile_IDs")

dup_ids = df[df.duplicated("Tile_ID", keep=False)]
if len(dup_ids):
    fail("no_dup_tile_ids", f"{len(dup_ids)} rows share a duplicate Tile_ID")
    print(dup_ids[["Tile_ID", "centroid_lat", "centroid_lon", "TrainClass"]].to_string())
else:
    ok("no_dup_tile_ids", "No duplicate Tile_IDs")


# Check 3: Duplicate centroids 

section("3. Duplicate centroid (lat, lon)")

df["_lat6"] = df["centroid_lat"].round(6)
df["_lon6"] = df["centroid_lon"].round(6)
dup_centroids = df[df.duplicated(["_lat6", "_lon6"], keep=False)]

if len(dup_centroids):
    fail("no_dup_centroids", f"{len(dup_centroids)} rows share a duplicate centroid (rounded to 6dp)")
    print(dup_centroids[["Tile_ID", "_lat6", "_lon6", "TrainClass"]].to_string())
else:
    ok("no_dup_centroids", "No duplicate centroids")


#Check 4: Centroid coordinate validit

section(f"4. Centroid coordinate validity (expected CRS: {CENTROID_CRS})")

lat_bad = df[(df["centroid_lat"] < -90) | (df["centroid_lat"] > 90)]
lon_bad = df[(df["centroid_lon"] < -180) | (df["centroid_lon"] > 180)]

if len(lat_bad):
    fail("centroid_lat_range", f"{len(lat_bad)} rows have centroid_lat outside [-90, 90]")
    print(lat_bad[["Tile_ID", "centroid_lat"]].to_string())
else:
    ok("centroid_lat_range", f"All centroid_lat in [-90, 90] — consistent with {CENTROID_CRS}")

if len(lon_bad):
    fail("centroid_lon_range", f"{len(lon_bad)} rows have centroid_lon outside [-180, 180]")
    print(lon_bad[["Tile_ID", "centroid_lon"]].to_string())
else:
    ok("centroid_lon_range", f"All centroid_lon in [-180, 180] — consistent with {CENTROID_CRS}")

# RTS sites are typically > 50°N
south_of_50 = df[df["centroid_lat"] < 50]
if len(south_of_50):
    warn("centroid_lat_plausibility",
         f"{len(south_of_50)} tiles have centroid_lat < 50°N — unexpected for RTS data. "
         f"Lat range: {south_of_50['centroid_lat'].min():.4f} – {south_of_50['centroid_lat'].max():.4f}")
    print(south_of_50[["Tile_ID", "centroid_lat", "centroid_lon", "TrainClass"]].head(10).to_string())
else:
    ok("centroid_lat_plausibility", "All centroids north of 50°N — plausible for RTS")


# Check 5: UID re-derivation 

section("5. UID (Tile_ID) re-derivable from centroid")

uid_mismatches = []
for _, row in df.iterrows():
    expected = make_tile_uid(round(row["centroid_lat"], 6), round(row["centroid_lon"], 6))
    if expected != row["Tile_ID"]:
        uid_mismatches.append({
            "Tile_ID":  row["Tile_ID"],
            "expected": expected,
            "lat":      row["centroid_lat"],
            "lon":      row["centroid_lon"],
        })

if uid_mismatches:
    fail("uid_rederivation", f"{len(uid_mismatches)} Tile_IDs do not match geohash of their centroid")
    print(pd.DataFrame(uid_mismatches).to_string())
else:
    ok("uid_rederivation", "All Tile_IDs match geohash(centroid_lat, centroid_lon)")


# Check 6: TrainClass values

section("6. TrainClass values")

unexpected_classes = set(df["TrainClass"].unique()) - VALID_CLASSES
if unexpected_classes:
    fail("train_class_values", f"Unexpected TrainClass values: {unexpected_classes}")
else:
    ok("train_class_values", f"All TrainClass values in {VALID_CLASSES}")


# Check 7: GCS tile existence and label matching

section("7. GCS tile existence and RGB / label parity")

rgb_blobs   = {b.name.split("/")[-1].replace(".tif", "")
               for b in bucket.list_blobs(prefix=RGB_PREFIX) if b.name.endswith(".tif")}
label_blobs = {b.name.split("/")[-1].replace(".tif", "")
               for b in bucket.list_blobs(prefix=LABELS_PREFIX) if b.name.endswith(".tif")}

meta_ids = set(df["Tile_ID"].astype(str))
pos_ids  = set(df[df["TrainClass"] == "positive"]["Tile_ID"].astype(str))
neg_ids  = set(df[df["TrainClass"] == "negative"]["Tile_ID"].astype(str))

print(f"  RGB tiles in GCS:   {len(rgb_blobs)}")
print(f"  Label tiles in GCS: {len(label_blobs)}")
print(f"  Metadata rows:      {len(meta_ids)}  (pos={len(pos_ids)}, neg={len(neg_ids)})")

in_meta_not_rgb = meta_ids - rgb_blobs
in_rgb_not_meta = rgb_blobs - meta_ids

if in_meta_not_rgb:
    fail("rgb_exists", f"{len(in_meta_not_rgb)} Tile_IDs in metadata have no RGB tile in GCS")
    print("  Examples:", list(in_meta_not_rgb)[:10])
else:
    ok("rgb_exists", "All metadata Tile_IDs have a corresponding RGB tile")

if in_rgb_not_meta:
    warn("rgb_orphan", f"{len(in_rgb_not_meta)} RGB tiles in GCS have no metadata row")
    print("  Examples:", list(in_rgb_not_meta)[:10])
else:
    ok("rgb_orphan", "No orphan RGB tiles")

# Labels required for positives only
pos_missing_label = pos_ids - label_blobs
neg_with_label    = neg_ids & label_blobs

if pos_missing_label:
    fail("label_exists_positive", f"{len(pos_missing_label)} positive tiles have no label in GCS")
    print("  Examples:", list(pos_missing_label)[:10])
else:
    ok("label_exists_positive", "All positive tiles have a corresponding label tile")

if neg_with_label:
    warn("label_unexpected_negative", f"{len(neg_with_label)} negative tiles unexpectedly have a label tile")
else:
    ok("label_unexpected_negative", "No negative tiles have label tiles (expected)")


# Check 8: Tile CRS, bands, dimensions, pixel values 

section(f"8. Tile CRS, bands, dimensions, pixel values  [SKIP_PIXEL={SKIP_PIXEL}]")

if SKIP_PIXEL:
    warn("pixel_checks", "SKIP_PIXEL_CHECK=1 — skipping tile download checks")
else:
    ids_to_check = list(meta_ids & rgb_blobs)
    if SAMPLE_TILES:
        ids_to_check = random.sample(ids_to_check, min(SAMPLE_TILES, len(ids_to_check)))
        print(f"  Sampling {len(ids_to_check)} tiles (set SAMPLE_TILES env var to change)")
    else:
        print(f"  Checking ALL {len(ids_to_check)} tiles — set SAMPLE_TILES=N to sample")

    crs_seen_rgb         = defaultdict(int)
    crs_seen_label       = defaultdict(int)
    band_errors          = []
    dim_errors           = []
    nodata_errors        = []
    label_val_errors     = []
    label_crs_mismatches = []

    for tile_id in ids_to_check:
        rgb_blob_path   = f"{RGB_PREFIX}{tile_id}.tif"
        label_blob_path = f"{LABELS_PREFIX}{tile_id}.tif"
        local_rgb       = f"{WORK_DIR}/val_tmp/rgb_{tile_id}.tif"
        local_label     = f"{WORK_DIR}/val_tmp/lbl_{tile_id}.tif"

        try:
            bucket.blob(rgb_blob_path).download_to_filename(local_rgb)

            with rasterio.open(local_rgb) as src:
                crs_str = src.crs.to_string() if src.crs else "None"
                crs_seen_rgb[crs_str] += 1

                if src.count != 3:
                    band_errors.append((tile_id, "rgb", f"count={src.count}, expected 3"))

                descs = [src.descriptions[i] or "" for i in range(src.count)]
                if descs != ["Red", "Green", "Blue"]:
                    band_errors.append((tile_id, "rgb_desc",
                                        f"descriptions={descs}, expected ['Red', 'Green', 'Blue']"))

                if src.width != TILE_SIZE or src.height != TILE_SIZE:
                    dim_errors.append((tile_id, f"{src.width}x{src.height}"))

                data = src.read()
                if src.nodata is not None and (data == src.nodata).all():
                    nodata_errors.append((tile_id, "rgb", "all nodata"))
                elif (data == 0).all():
                    nodata_errors.append((tile_id, "rgb", "all zeros"))

                rgb_crs = src.crs

            os.remove(local_rgb)

        except Exception as e:
            fail("tile_read", f"RGB tile {tile_id}: {e}")
            traceback.print_exc()

        # Label checks (positives only)
        train_class = df.loc[df["Tile_ID"] == tile_id, "TrainClass"].values
        if len(train_class) and train_class[0] == "positive" and tile_id in label_blobs:
            try:
                bucket.blob(label_blob_path).download_to_filename(local_label)

                with rasterio.open(local_label) as lsrc:
                    lcrs_str = lsrc.crs.to_string() if lsrc.crs else "None"
                    crs_seen_label[lcrs_str] += 1

                    if lsrc.count != 1:
                        band_errors.append((tile_id, "label", f"count={lsrc.count}, expected 1"))

                    unique_vals  = set(np.unique(lsrc.read(1)).tolist())
                    invalid_vals = unique_vals - VALID_LABEL_VALS
                    if invalid_vals:
                        label_val_errors.append((tile_id, invalid_vals))

                    if lsrc.crs != rgb_crs:
                        label_crs_mismatches.append((tile_id, f"rgb={rgb_crs}, label={lsrc.crs}"))

                os.remove(local_label)

            except Exception as e:
                fail("label_read", f"Label tile {tile_id}: {e}")

    # Report 

    print(f"\n  CRS distribution — RGB tiles:")
    for crs_str, cnt in sorted(crs_seen_rgb.items(), key=lambda x: -x[1]):
        marker = "ok" if EXPECTED_CRS in crs_str else "!!"
        print(f"    [{marker}] {crs_str}: {cnt} tiles")

    if len(crs_seen_rgb) == 1 and EXPECTED_CRS in next(iter(crs_seen_rgb)):
        ok("rgb_crs_consistent", f"All RGB tiles use {EXPECTED_CRS}")
    elif len(crs_seen_rgb) == 1:
        warn("rgb_crs_expected", f"All RGB tiles use a single CRS ({next(iter(crs_seen_rgb))}) "
             f"but expected {EXPECTED_CRS}")
    else:
        fail("rgb_crs_consistent", f"RGB tiles have {len(crs_seen_rgb)} different CRS values — "
             "mixed CRS will break downstream training")

    if crs_seen_label:
        print(f"\n  CRS distribution — label tiles:")
        for crs_str, cnt in sorted(crs_seen_label.items(), key=lambda x: -x[1]):
            marker = "ok" if EXPECTED_CRS in crs_str else "!!"
            print(f"    [{marker}] {crs_str}: {cnt} tiles")

        if len(crs_seen_label) == 1 and EXPECTED_CRS in next(iter(crs_seen_label)):
            ok("label_crs_consistent", f"All label tiles use {EXPECTED_CRS}")
        elif len(crs_seen_label) == 1:
            warn("label_crs_expected", f"All label tiles use a single CRS ({next(iter(crs_seen_label))}) "
                 f"but expected {EXPECTED_CRS}")
        else:
            fail("label_crs_consistent", f"Label tiles have {len(crs_seen_label)} different CRS values")

    if label_crs_mismatches:
        fail("rgb_label_crs_match",
             f"{len(label_crs_mismatches)} tile pairs have mismatched RGB/label CRS")
        for tid, detail in label_crs_mismatches[:10]:
            print(f"    {tid}: {detail}")
    elif crs_seen_label:
        ok("rgb_label_crs_match", "RGB and label CRS match for all checked tile pairs")

    if band_errors:
        fail("band_check", f"{len(band_errors)} band errors found")
        for tid, kind, detail in band_errors[:10]:
            print(f"    {tid} [{kind}]: {detail}")
    else:
        ok("band_check", "All checked tiles have correct band count and descriptions")

    if dim_errors:
        fail("tile_dimensions", f"{len(dim_errors)} tiles are not {TILE_SIZE}×{TILE_SIZE}")
        for tid, dims in dim_errors[:10]:
            print(f"    {tid}: {dims}")
    else:
        ok("tile_dimensions", f"All checked tiles are {TILE_SIZE}×{TILE_SIZE}")

    if nodata_errors:
        fail("nodata_check", f"{len(nodata_errors)} tiles are entirely nodata or zero")
        for tid, kind, detail in nodata_errors[:10]:
            print(f"    {tid} [{kind}]: {detail}")
    else:
        ok("nodata_check", "No tiles are entirely nodata or zero")

    if label_val_errors:
        fail("label_values", f"{len(label_val_errors)} label tiles contain unexpected pixel values")
        for tid, vals in label_val_errors[:10]:
            print(f"    {tid}: invalid values = {vals}")
    else:
        ok("label_values", f"All label tiles contain only valid values {VALID_LABEL_VALS}")


# Summary

section("SUMMARY")

n_fail = len(issues)
n_warn = len(warnings)

if issues:
    print(f"\n  {n_fail} FAILURE(S):\n")
    for _, check, msg in issues:
        print(f"    FAIL [{check}] {msg}")
else:
    print("\n  No failures.")

if warnings:
    print(f"\n  {n_warn} WARNING(S):\n")
    for _, check, msg in warnings:
        print(f"    WARN [{check}] {msg}")
else:
    print("  No warnings.")

print()
if n_fail == 0:
    print("  All checks passed.")
elif n_fail <= 2:
    print(f"  {n_fail} check(s) failed -- review before training.")
else:
    print(f"  {n_fail} check(s) failed —-  pipeline output needs attention.")

print()