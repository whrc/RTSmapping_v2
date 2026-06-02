import re
import tempfile
import os

import pandas as pd
import geopandas as gpd
from pathlib import Path
from google.cloud import storage
from google.colab import auth

from ARTS import dataformatting

# Authenticate once at the top — comment out if running outside Colab
auth.authenticate_user()

# Config
def _parse_fields(raw: str) -> list:
    return [f.strip() for f in raw.split(",") if f.strip()]
 
GCS_BUCKET      = input("GCS bucket name: ").strip()
GCS_BASE_PREFIX = input("GCS base prefix [ARTS]: ").strip() or "ARTS"
your_file       = input("Input filename (.geojson or .shp): ").strip()
dataset_version = input("Dataset version [v.1.0.0]: ").strip() or "v.1.0.0"
 
_new_fields_raw = input("New metadata fields, comma-separated (leave blank if none): ").strip()
new_fields      = _parse_fields(_new_fields_raw)
 
separate_file   = input("Write to separate file instead of appending? (y/n) [n]: ").strip().lower() == "y"
 
# GCS helpers
def gcs_path(*parts: str) -> str:
    return "/".join(p.strip("/") for p in parts if p)


def gcs_read_csv(bucket_name: str, blob_path: str) -> pd.DataFrame:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    return pd.read_csv(blob.open("rt"))


def gcs_read_geojson(bucket_name: str, blob_path: str) -> gpd.GeoDataFrame:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    return gpd.read_file(blob.open("rb"))


def gcs_download_tmp(bucket_name: str, blob_path: str, suffix: str = "") -> str:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    _, tmp_path = tempfile.mkstemp(suffix=suffix)
    blob.download_to_filename(tmp_path)
    return tmp_path


def gcs_write_geojson(gdf: gpd.GeoDataFrame, bucket_name: str, blob_path: str) -> None:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    with blob.open("wt") as f:
        f.write(gdf.to_json())
    print(f"Saved: gs://{bucket_name}/{blob_path}")


def upload_directory_to_gcs(local_dir: Path, bucket_name: str, gcs_prefix: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for local_file in local_dir.rglob("*"):
        if local_file.is_file():
            blob_name = gcs_path(gcs_prefix, str(local_file.relative_to(local_dir)))
            bucket.blob(blob_name).upload_from_filename(str(local_file))
            print(f"Uploaded: gs://{bucket_name}/{blob_name}")

# Get GCS paths

file_stem = your_file.rsplit(".", 1)[0]

output_prefix  = gcs_path(GCS_BASE_PREFIX, "output")
arts_main_blob = gcs_path(GCS_BASE_PREFIX, "ARTS_main_dataset", dataset_version,
                          f"ARTS_main_dataset_{dataset_version}.geojson")
metadata_blob  = gcs_path(GCS_BASE_PREFIX, "Metadata_Format_Summary.csv")

# Step 1 outputs
uids_blob   = gcs_path(output_prefix, f"{file_stem}_with_uids.geojson")

# Your manually edited file (must be uploaded to GCS before running this script)
edited_blob = gcs_path(output_prefix, f"{file_stem}_with_uids_edited.geojson")

# ── Validate edited file is present ──────────────────────────────────────────

print("Checking that the edited file exists in GCS …")
client = storage.Client()
if not client.bucket(GCS_BUCKET).blob(edited_blob).exists():
    raise FileNotFoundError(
        f"Edited file not found:\n  gs://{GCS_BUCKET}/{edited_blob}\n\n"
        "Please upload your GIS-edited file to that path and re-run this script."
    )
print("  Found.")

# ── Load Metadata Format ──────────────────────────────────────────────────────

print("\nLoading Metadata Format Summary …")
metadata_format_summary = gcs_read_csv(GCS_BUCKET, metadata_blob)

required_fields  = list(metadata_format_summary[metadata_format_summary.Required == "True"].FieldName.values)
generated_fields = list(metadata_format_summary[metadata_format_summary.Required == "Generated"].FieldName.values)
optional_fields  = list(metadata_format_summary[metadata_format_summary.Required == "False"].FieldName.values)
all_fields       = required_fields + generated_fields + optional_fields + new_fields

print(metadata_format_summary)

# ── Load ARTS Main Dataset ────────────────────────────────────────────────────

print("\nLoading ARTS main dataset …")
ARTS_main_dataset = gcs_read_geojson(GCS_BUCKET, arts_main_blob).filter(
    items=required_fields + generated_fields + optional_fields + ["geometry"]
)
ARTS_main_dataset.ContributionDate = pd.to_datetime(ARTS_main_dataset.ContributionDate)

# ── Load UID-stamped Dataset from Step 1 ─────────────────────────────────────

print("\nLoading UID-stamped dataset from Step 1 …")
uids_local = gcs_download_tmp(GCS_BUCKET, uids_blob, suffix=".geojson")
new_dataset = gpd.read_file(uids_local)
os.unlink(uids_local)
print(new_dataset)

# ── Load Manually Edited File and Merge ──────────────────────────────────────

print("\nLoading manually edited file and merging …")
edited_local = gcs_download_tmp(GCS_BUCKET, edited_blob, suffix=".geojson")
merged_data = dataformatting.merge_data(new_dataset, edited_local)
os.unlink(edited_local)
print(merged_data)

# ── Remove False Negatives ────────────────────────────────────────────────────

print("\nRemoving false negatives …")
merged_data = dataformatting.remove_new_false_negatives(merged_data)

if merged_data[
    (merged_data.TrainClass == "Positive") & (merged_data.FalseNegative.str.len() > 0)
].shape[0] > 0:
    ARTS_main_dataset = dataformatting.remove_old_false_negatives(ARTS_main_dataset, merged_data)
    updated_main = True
else:
    updated_main = False

# ── Check Completeness of Intersection Information ────────────────────────────

print("\nChecking intersection info completeness …")
tmp_base = Path(tempfile.mkdtemp())
dataformatting.check_intersection_info(merged_data, your_file, tmp_base, False)

# ── Confirm UIDs ──────────────────────────────────────────────────────────────

print("\nConfirming UIDs …")
dataformatting.check_uids(merged_data.UID)

# ── Final Column Selection ────────────────────────────────────────────────────

print("\nFinalising columns …")
formatted_data = dataformatting.add_empty_columns(merged_data, list(optional_fields))
formatted_data = formatted_data[all_fields + ["geometry"]]
print(formatted_data)

# ── Output ────────────────────────────────────────────────────────────────────

print("\nWriting output …")

new_increment      = int(re.split("-", re.split(r"\.", dataset_version)[1])[0]) + 1
new_version        = ".".join(re.split(r"\.", dataset_version)[:1] + [str(new_increment), "0", "0"])
updated_ARTS_local = Path(tempfile.mkdtemp()) / new_version

dataformatting.output(
    formatted_data,
    ARTS_main_dataset,
    new_fields,
    all_fields,
    tmp_base,
    your_file,
    updated_ARTS_local,
    separate_file,
    False,
    updated_main,
)

upload_directory_to_gcs(tmp_base, GCS_BUCKET, output_prefix)
upload_directory_to_gcs(
    updated_ARTS_local.parent,
    GCS_BUCKET,
    gcs_path(GCS_BASE_PREFIX, "ARTS_main_dataset"),
)

print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 2 complete.

  Updated ARTS main dataset version: {new_version}
  Output written to:
    gs://{GCS_BUCKET}/{gcs_path(GCS_BASE_PREFIX, "ARTS_main_dataset")}
    gs://{GCS_BUCKET}/{output_prefix}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")