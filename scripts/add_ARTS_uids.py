# Adds the UIDs to new polygons and match format of ARTS data set
# Follow with manual reassignment of UIDs to match the relationship columns

import uuid
import warnings
import re
import tempfile
import os

import pandas as pd
import geopandas as gpd
from pathlib import Path
from google.cloud import storage
from google.colab import auth

from ARTS import dataformatting

# If running in colab authenticate outside of 
auth.authenticate_user()

# config setup

def _parse_fields(raw: str) -> list:
    return [f.strip() for f in raw.split(",") if f.strip()]
 
GCS_BUCKET      = input("GCS bucket name: ").strip()
GCS_BASE_PREFIX = input("GCS base prefix [ARTS]: ").strip() or "ARTS"
your_file       = input("Input filename (.geojson or .shp): ").strip()
dataset_version = input("Dataset version [v.1.0.0]: ").strip() or "v.1.0.0"
 
_new_fields_raw = input("New metadata fields, comma-separated (leave blank if none): ").strip()
new_fields      = _parse_fields(_new_fields_raw)
 
new_fields_abbreviated = []
if your_file.endswith(".shp") and new_fields:
    _abbr_raw          = input("Abbreviated field names for shapefile, comma-separated: ").strip()
    new_fields_abbreviated = _parse_fields(_abbr_raw)
 
calculate_centroid = input("Generate centroid columns? (y/n) [n]: ").strip().lower() == "y"

#GCS helpers

def gcs_path(*parts: str) -> str:
    """Join GCS path components, stripping leading/trailing slashes."""
    return "/".join(p.strip("/") for p in parts if p)


def gcs_read_csv(bucket_name: str, blob_path: str) -> pd.DataFrame:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    return pd.read_csv(blob.open("rt"))


def gcs_read_geojson(bucket_name: str, blob_path: str) -> gpd.GeoDataFrame:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    return gpd.read_file(blob.open("rb"))


def gcs_write_geojson(gdf: gpd.GeoDataFrame, bucket_name: str, blob_path: str) -> None:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    with blob.open("wt") as f:
        f.write(gdf.to_json())
    print(f"Saved: gs://{bucket_name}/{blob_path}")


def gcs_download_tmp(bucket_name: str, blob_path: str, suffix: str = "") -> str:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    _, tmp_path = tempfile.mkstemp(suffix=suffix)
    blob.download_to_filename(tmp_path)
    return tmp_path

# GCS path setups

file_stem = your_file.rsplit(".", 1)[0]
file_ext  = your_file.rsplit(".", 1)[-1]

rts_blob          = gcs_path(GCS_BASE_PREFIX, "input_data", your_file)
arts_main_blob    = gcs_path(GCS_BASE_PREFIX, "ARTS_main_dataset", dataset_version,
                             f"ARTS_main_dataset_{dataset_version}.geojson")
metadata_blob     = gcs_path(GCS_BASE_PREFIX, "Metadata_Format_Summary.csv")
output_prefix     = gcs_path(GCS_BASE_PREFIX, "output")

# The two files this script writes to GCS:
uids_out_blob          = gcs_path(output_prefix, f"{file_stem}_with_uids.geojson")
intersections_out_blob = gcs_path(output_prefix, f"{file_stem}_overlapping.geojson")

# Load metadata fromat

print("Loading Metadata Format Summary …")
metadata_format_summary = gcs_read_csv(GCS_BUCKET, metadata_blob)

required_fields  = list(metadata_format_summary[metadata_format_summary.Required == "True"].FieldName.values)
generated_fields = list(metadata_format_summary[metadata_format_summary.Required == "Generated"].FieldName.values)
optional_fields  = list(metadata_format_summary[metadata_format_summary.Required == "False"].FieldName.values)

print(metadata_format_summary)

# load ARTS dataset for comp

print("\nLoading ARTS main dataset …")
ARTS_main_dataset = gcs_read_geojson(GCS_BUCKET, arts_main_blob).filter(
    items=required_fields + generated_fields + optional_fields + ["geometry"]
)
ARTS_main_dataset.ContributionDate = pd.to_datetime(ARTS_main_dataset.ContributionDate)

for field in required_fields:
    if field not in ARTS_main_dataset.columns:
        raise ValueError(
            f"{field!r} is missing from the ARTS main dataset. "
            "Has the dataset been modified since download?"
        )

print(ARTS_main_dataset)

# Load and preprocess new polygons dataset

print("\nLoading new RTS dataset …")
rts_local = gcs_download_tmp(GCS_BUCKET, rts_blob, suffix=f".{file_ext}")

if re.search(r"\.geojson$", your_file):
    new_dataset = dataformatting.preprocessing(
        rts_local,
        required_fields,
        generated_fields,
        optional_fields,
        new_fields,
        None,
        calculate_centroid=calculate_centroid,
    )
elif re.search(r"\.shp$", your_file):
    new_dataset = dataformatting.preprocessing(
        rts_local,
        required_fields,
        generated_fields,
        optional_fields,
        new_fields,
        new_fields_abbreviated,
        calculate_centroid,
    )
else:
    raise ValueError(f"Unsupported file type: {your_file}")

os.unlink(rts_local)
print(new_dataset)

# Metadata check

print("\nRunning formatting checks …")
dataformatting.run_formatting_checks(new_dataset)

# generation of uids

print("\nGenerating UIDs …")
dataformatting.seed_gen(new_dataset)

new_dataset["UID"] = [
    str(uuid.uuid5(uuid.NAMESPACE_DNS, name=seed)) for seed in new_dataset.seed
]
new_dataset.drop("BaseMapResolutionStr", inplace=True, axis=1)
print(new_dataset.UID)

# check for intersections

print("\nChecking for intersections …")
_, intersections_tmp = tempfile.mkstemp(suffix=".geojson")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
    new_dataset = dataformatting.check_intersections(
        new_dataset, ARTS_main_dataset, intersections_tmp, False
    )

gcs_write_geojson(gpd.read_file(intersections_tmp), GCS_BUCKET, intersections_out_blob)
os.unlink(intersections_tmp)

# write new uids to GCS

print("\nWriting UID-stamped dataset to GCS …")
gcs_write_geojson(new_dataset, GCS_BUCKET, uids_out_blob)

print(new_dataset)

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 1 complete.

  Files written to GCS:
    gs://{GCS_BUCKET}/{uids_out_blob}
        └─ New data with UIDs and formatted columns.
           Use this as your working file for manual editing.

    gs://{GCS_BUCKET}/{intersections_out_blob}
        └─ Overlapping polygons for GIS inspection.

  Next steps:
    1. Download both files and open in your GIS software.
    2. Classify each intersection into the appropriate
       relationship column:
         RepeatRTS, StabilizedRTS, NewRTS, MergedRTS,
         SplitRTS, AccidentalOverlap, UnknownRelationship
    3. Save your edits and upload the edited file back
       to GCS at:
         gs://{GCS_BUCKET}/{gcs_path(output_prefix, f"{file_stem}_with_uids_edited.geojson")}
    4. Run ARTS_step2_validate_and_output.py.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")