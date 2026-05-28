#!/usr/bin/env python3
"""Validate RTS dataset in a GCS bucket against data/data.md spec.

Streams files from GCS to check layout, tile correspondence, metadata schema,
splits consistency, raster integrity, and label semantics. Prints a report and
writes a log file.

Usage:
    python scripts/check_data_bucket.py --bucket gs://abrupt_thaw/RTS_MODEL_V2/DATA
"""

import argparse
import io
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import rasterio
import yaml
from google.cloud import storage
from tqdm import tqdm

# Ensure GDAL /vsigs/ can authenticate via Application Default Credentials.
# On a GCE VM the compute SA is the default, but if the user has run
# `gcloud auth application-default login`, the ADC file supersedes it for GDAL.
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    _adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if _adc.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_adc)

logger = logging.getLogger("check_data_bucket")

EXPECTED_CRS = "EPSG:3857"
EXPECTED_TILE_SIZE = 512
EXPECTED_RGB_BANDS = 3
EXPECTED_EXTRA_BANDS = 4
EXPECTED_METADATA_COLUMNS = [
    "Tile_ID", "centroid_lat", "centroid_lon",
    "TrainClass", "RegionName", "UIDs",
]
VALID_TRAIN_CLASSES = {"positive", "negative"}
VALID_LABEL_VALUES = {0, 1, 255}
NEGATIVE_SAMPLE_SIZE = 200
SEED = 42


class CheckResult(NamedTuple):
    """Result of a single validation check."""

    name: str
    passed: bool
    messages: list[str]


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def parse_bucket_url(url: str) -> tuple[str, str]:
    """Parse gs://bucket/prefix into (bucket_name, prefix).

    Args:
        url: Full GCS URL like gs://bucket_name/some/prefix.

    Returns:
        Tuple of (bucket_name, prefix). Prefix has no trailing slash.
    """
    stripped = url.removeprefix("gs://")
    parts = stripped.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
    return bucket_name, prefix


def read_blob_text(client: storage.Client, bucket_name: str,
                   blob_path: str) -> str:
    """Read a small text blob from GCS.

    Args:
        client: GCS client.
        bucket_name: Name of the bucket.
        blob_path: Full blob path within the bucket.

    Returns:
        Text content of the blob.
    """
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_text()


def list_top_level_entries(client: storage.Client, bucket_name: str,
                           prefix: str) -> tuple[set[str], set[str]]:
    """List top-level files and directories under a GCS prefix.

    Args:
        client: GCS client.
        bucket_name: Name of the bucket.
        prefix: Prefix path (no trailing slash).

    Returns:
        Tuple of (file_names, dir_names). Dir names include trailing slash.
    """
    search_prefix = f"{prefix}/" if prefix else ""
    iterator = client.list_blobs(
        bucket_name, prefix=search_prefix, delimiter="/",
    )
    files = set()
    for blob in iterator:
        name = blob.name.removeprefix(search_prefix)
        if name and "/" not in name:
            files.add(name)
    dirs = {p.removeprefix(search_prefix) for p in iterator.prefixes}
    return files, dirs


def list_tif_tile_ids(client: storage.Client, bucket_name: str,
                      prefix: str, subdir: str) -> set[str]:
    """List .tif file stems in a GCS subdirectory.

    Args:
        client: GCS client.
        bucket_name: Name of the bucket.
        prefix: Root prefix (no trailing slash).
        subdir: Subdirectory name (e.g. "PLANET-RGB").

    Returns:
        Set of tile ID strings (file stems without .tif extension).
    """
    search_prefix = f"{prefix}/{subdir}/"
    tile_ids: set[str] = set()
    for blob in client.list_blobs(bucket_name, prefix=search_prefix):
        name = blob.name.removeprefix(search_prefix)
        if name.endswith(".tif") and "/" not in name:
            tile_ids.add(name.removesuffix(".tif"))
    return tile_ids


# ---------------------------------------------------------------------------
# Check 1: Layout
# ---------------------------------------------------------------------------

def check_layout(client: storage.Client, bucket_name: str,
                 prefix: str) -> tuple[CheckResult, bool, bool]:
    """Check that the bucket root has the expected directories and files.

    Args:
        client: GCS client.
        bucket_name: Name of the bucket.
        prefix: Root prefix.

    Returns:
        Tuple of (CheckResult, has_extra, has_version).
    """
    messages: list[str] = []
    passed = True

    files, dirs = list_top_level_entries(client, bucket_name, prefix)

    required_dirs = {"PLANET-RGB/", "labels/"}
    optional_dirs = {"EXTRA/"}
    required_files = {"metadata.csv", "splits.yaml"}
    optional_files = {"version.json"}

    # Check required directories
    for d in required_dirs:
        if d in dirs:
            messages.append(f"  Found required directory: {d}")
        else:
            messages.append(f"  MISSING required directory: {d}")
            passed = False

    # Check optional directories
    has_extra = "EXTRA/" in dirs
    if has_extra:
        messages.append("  Found optional directory: EXTRA/")
    else:
        messages.append("  Optional directory not present: EXTRA/")

    # Check required files
    for f in required_files:
        if f in files:
            messages.append(f"  Found required file: {f}")
        else:
            messages.append(f"  MISSING required file: {f}")
            passed = False

    # Check optional files
    has_version = "version.json" in files
    if has_version:
        messages.append("  Found optional file: version.json")
    else:
        messages.append("  Optional file not present: version.json")

    return CheckResult("Layout", passed, messages), has_extra, has_version


# ---------------------------------------------------------------------------
# Check 2: Tile correspondence
# ---------------------------------------------------------------------------

def check_tile_correspondence(client: storage.Client, bucket_name: str,
                              prefix: str,
                              metadata_df: pd.DataFrame | None,
                              has_extra: bool,
                              rgb_ids: set[str] | None = None,
                              ) -> CheckResult:
    """Check that tile ID sets match across directories and metadata.

    Args:
        client: GCS client.
        bucket_name: Name of the bucket.
        prefix: Root prefix.
        metadata_df: Loaded metadata DataFrame, or None if missing.
        has_extra: Whether EXTRA/ directory exists.
        rgb_ids: Pre-listed RGB tile IDs to avoid re-listing.

    Returns:
        CheckResult with mismatch details.
    """
    messages: list[str] = []
    passed = True

    if rgb_ids is None:
        rgb_ids = list_tif_tile_ids(client, bucket_name, prefix, "PLANET-RGB")
    label_ids = list_tif_tile_ids(client, bucket_name, prefix, "labels")

    sources: dict[str, set[str]] = {
        "PLANET-RGB": rgb_ids,
        "labels": label_ids,
    }

    if metadata_df is not None and "Tile_ID" in metadata_df.columns:
        meta_ids = set(metadata_df["Tile_ID"].astype(str))
        sources["metadata.csv"] = meta_ids
    else:
        messages.append("  metadata.csv not available — skipping metadata tile comparison")

    if has_extra:
        extra_ids = list_tif_tile_ids(client, bucket_name, prefix, "EXTRA")
        sources["EXTRA"] = extra_ids

    # Only positive tiles are expected to have label files; negative tiles are
    # all-background by definition and don't need a label raster on disk.
    positive_ids: set[str] = set()
    if metadata_df is not None and "Tile_ID" in metadata_df.columns and "TrainClass" in metadata_df.columns:
        positive_ids = set(metadata_df.loc[metadata_df["TrainClass"] == "positive", "Tile_ID"].astype(str))
    label_reference = positive_ids if positive_ids else rgb_ids  # fall back to all RGB if no metadata

    messages.append(f"  PLANET-RGB tile count: {len(rgb_ids)}")
    messages.append(f"  labels tile count: {len(label_ids)}")
    messages.append(f"  positive tile count (metadata): {len(positive_ids)}")
    if has_extra:
        messages.append(f"  EXTRA tile count: {len(sources.get('EXTRA', set()))}")
    if "metadata.csv" in sources:
        messages.append(f"  metadata.csv tile count: {len(sources['metadata.csv'])}")

    # Labels must exist for every positive tile; extra label files are an error.
    missing_labels = label_reference - label_ids
    extra_labels = label_ids - label_reference
    if missing_labels:
        passed = False
        sample = sorted(missing_labels)[:5]
        messages.append(
            f"  Positive tiles missing label file ({len(missing_labels)}): "
            f"{sample}{'...' if len(missing_labels) > 5 else ''}"
        )
    if extra_labels:
        passed = False
        sample = sorted(extra_labels)[:5]
        messages.append(
            f"  Label files with no matching positive tile ({len(extra_labels)}): "
            f"{sample}{'...' if len(extra_labels) > 5 else ''}"
        )

    # metadata.csv must cover every PLANET-RGB tile
    if "metadata.csv" in sources:
        meta_ids = sources["metadata.csv"]
        only_in_rgb = rgb_ids - meta_ids
        only_in_meta = meta_ids - rgb_ids
        if only_in_rgb:
            passed = False
            sample = sorted(only_in_rgb)[:5]
            messages.append(
                f"  In PLANET-RGB but not in metadata.csv ({len(only_in_rgb)}): "
                f"{sample}{'...' if len(only_in_rgb) > 5 else ''}"
            )
        if only_in_meta:
            passed = False
            sample = sorted(only_in_meta)[:5]
            messages.append(
                f"  In metadata.csv but not in PLANET-RGB ({len(only_in_meta)}): "
                f"{sample}{'...' if len(only_in_meta) > 5 else ''}"
            )

    if has_extra:
        extra_ids = sources.get("EXTRA", set())
        only_in_rgb = rgb_ids - extra_ids
        only_in_extra = extra_ids - rgb_ids
        if only_in_rgb:
            passed = False
            sample = sorted(only_in_rgb)[:5]
            messages.append(
                f"  In PLANET-RGB but not in EXTRA ({len(only_in_rgb)}): "
                f"{sample}{'...' if len(only_in_rgb) > 5 else ''}"
            )
        if only_in_extra:
            passed = False
            sample = sorted(only_in_extra)[:5]
            messages.append(
                f"  In EXTRA but not in PLANET-RGB ({len(only_in_extra)}): "
                f"{sample}{'...' if len(only_in_extra) > 5 else ''}"
            )

    if passed:
        messages.append(
            f"  Tile correspondence OK: {len(rgb_ids)} RGB, "
            f"{len(label_ids)}/{len(positive_ids)} positive labels present"
        )

    return CheckResult("Tile Correspondence", passed, messages)


# ---------------------------------------------------------------------------
# Check 3: Metadata schema
# ---------------------------------------------------------------------------

def check_metadata_schema(metadata_df: pd.DataFrame | None) -> CheckResult:
    """Validate metadata.csv schema and value constraints.

    Args:
        metadata_df: Loaded metadata DataFrame, or None if missing.

    Returns:
        CheckResult with constraint violation details.
    """
    if metadata_df is None:
        return CheckResult(
            "Metadata Schema", False,
            ["  SKIPPED — metadata.csv not available"],
        )

    messages: list[str] = []
    passed = True

    # Column check
    actual_cols = list(metadata_df.columns)
    if actual_cols != EXPECTED_METADATA_COLUMNS:
        passed = False
        missing = set(EXPECTED_METADATA_COLUMNS) - set(actual_cols)
        extra = set(actual_cols) - set(EXPECTED_METADATA_COLUMNS)
        if missing:
            messages.append(f"  MISSING columns: {sorted(missing)}")
        if extra:
            messages.append(f"  Unexpected columns: {sorted(extra)}")
        if not missing and not extra:
            messages.append(
                f"  Column order mismatch: expected {EXPECTED_METADATA_COLUMNS}, "
                f"got {actual_cols}"
            )
    else:
        messages.append("  All expected columns present in correct order")

    # Tile_ID uniqueness
    if "Tile_ID" in metadata_df.columns:
        n_dupes = metadata_df["Tile_ID"].duplicated().sum()
        if n_dupes > 0:
            passed = False
            messages.append(f"  Tile_ID has {n_dupes} duplicate values")
        else:
            messages.append(f"  Tile_ID unique ({len(metadata_df)} entries)")

    # TrainClass values
    if "TrainClass" in metadata_df.columns:
        invalid_classes = set(metadata_df["TrainClass"].unique()) - VALID_TRAIN_CLASSES
        if invalid_classes:
            passed = False
            messages.append(f"  Invalid TrainClass values: {invalid_classes}")
        else:
            messages.append("  TrainClass values valid (positive/negative)")

    # UIDs: ideally empty for negatives, non-empty for positives.
    # The tile creation scripts use '9999' as a placeholder UID when object-level
    # IDs are not yet assigned — treat this as a warning, not a hard failure.
    if "UIDs" in metadata_df.columns and "TrainClass" in metadata_df.columns:
        neg_mask = metadata_df["TrainClass"] == "negative"
        pos_mask = metadata_df["TrainClass"] == "positive"

        # Negative tiles having non-empty UIDs is a data-quality note, not a blocker.
        neg_with_uids = metadata_df[neg_mask & metadata_df["UIDs"].notna()
                                     & (metadata_df["UIDs"].astype(str).str.strip() != "")]
        if len(neg_with_uids) > 0:
            sample = neg_with_uids["Tile_ID"].head(5).tolist()
            messages.append(
                f"  NOTE: {len(neg_with_uids)} negative tiles have non-empty UIDs "
                f"(placeholder '9999'?): {sample}"
            )

        # Positive tiles should have non-empty UIDs.
        pos_without_uids = metadata_df[pos_mask & (metadata_df["UIDs"].isna()
                                       | (metadata_df["UIDs"].astype(str).str.strip() == ""))]
        if len(pos_without_uids) > 0:
            passed = False
            sample = pos_without_uids["Tile_ID"].head(5).tolist()
            messages.append(
                f"  {len(pos_without_uids)} positive tiles have empty UIDs: {sample}"
            )

        if len(pos_without_uids) == 0:
            messages.append("  UIDs non-empty for all positive tiles")

    # RegionName non-empty
    if "RegionName" in metadata_df.columns:
        empty_regions = metadata_df[
            metadata_df["RegionName"].isna()
            | (metadata_df["RegionName"].astype(str).str.strip() == "")
        ]
        if len(empty_regions) > 0:
            passed = False
            sample = empty_regions["Tile_ID"].head(5).tolist()
            messages.append(
                f"  {len(empty_regions)} tiles have empty RegionName: {sample}"
            )
        else:
            messages.append("  RegionName non-empty for all tiles")

    return CheckResult("Metadata Schema", passed, messages)


# ---------------------------------------------------------------------------
# Check 4: Splits
# ---------------------------------------------------------------------------

def check_splits(splits: dict | None,
                 metadata_df: pd.DataFrame | None) -> CheckResult:
    """Validate splits.yaml against metadata regions.

    Args:
        splits: Parsed splits.yaml dict, or None if missing.
        metadata_df: Loaded metadata DataFrame, or None if missing.

    Returns:
        CheckResult with region coverage details.
    """
    if splits is None and metadata_df is None:
        return CheckResult(
            "Splits", False,
            ["  SKIPPED — both splits.yaml and metadata.csv not available"],
        )
    if splits is None:
        return CheckResult(
            "Splits", False,
            ["  SKIPPED — splits.yaml not available"],
        )
    if metadata_df is None:
        return CheckResult(
            "Splits", False,
            ["  SKIPPED — metadata.csv not available (cannot cross-check regions)"],
        )

    messages: list[str] = []
    passed = True

    metadata_regions = set(metadata_df["RegionName"].unique())

    # val_balanced and val_realistic intentionally share the same regions —
    # they differ only in sampling ratio at eval time, not in geography.
    # Skip the disjointness check for val_balanced.
    ALLOWED_OVERLAP_PAIRS = {("val_realistic", "val_balanced"), ("val_balanced", "val_realistic")}

    # Collect all regions from splits and check for duplicates
    all_split_regions: list[str] = []
    region_to_split: dict[str, str] = {}

    for split_name, regions in splits.items():
        if not isinstance(regions, list):
            passed = False
            messages.append(f"  Split '{split_name}' value is not a list")
            continue
        for region in regions:
            all_split_regions.append(region)
            if region in region_to_split:
                existing = region_to_split[region]
                if (existing, split_name) not in ALLOWED_OVERLAP_PAIRS:
                    passed = False
                    messages.append(
                        f"  Region '{region}' in multiple splits: "
                        f"'{existing}' and '{split_name}'"
                    )
                # val_balanced uses val_realistic geography — don't overwrite the primary label
            else:
                region_to_split[region] = split_name

    split_region_set = set(all_split_regions)

    # Every region in splits must exist in metadata
    missing_from_meta = split_region_set - metadata_regions
    if missing_from_meta:
        passed = False
        messages.append(
            f"  Regions in splits.yaml but not in metadata: "
            f"{sorted(missing_from_meta)}"
        )

    # Flag metadata regions not assigned to any split
    unassigned = metadata_regions - split_region_set
    if unassigned:
        passed = False
        messages.append(
            f"  Metadata regions not in any split: {sorted(unassigned)}"
        )

    if passed:
        messages.append(
            f"  {len(splits)} splits, {len(split_region_set)} regions assigned, "
            f"0 unassigned"
        )

    return CheckResult("Splits", passed, messages)


# ---------------------------------------------------------------------------
# Check 5: Rasters
# ---------------------------------------------------------------------------

def _validate_tile(bucket_name: str, prefix: str, tile_id: str,
                   has_extra: bool,
                   train_class: str = "positive") -> tuple[list[str], dict]:
    """Validate raster files for a single tile.

    Args:
        bucket_name: GCS bucket name.
        prefix: Root prefix.
        tile_id: Tile identifier.
        has_extra: Whether to check EXTRA channel.
        train_class: 'positive' or 'negative'. Negative tiles have no label file.

    Returns:
        Tuple of (error_messages, detail_dict).
    """
    errors: list[str] = []
    detail: dict = {"tile_id": tile_id, "train_class": train_class,
                    "n_rts_pixels": 0, "n_total_pixels": 0}

    rgb_path = f"/vsigs/{bucket_name}/{prefix}/PLANET-RGB/{tile_id}.tif"
    lbl_path = f"/vsigs/{bucket_name}/{prefix}/labels/{tile_id}.tif"

    rgb_crs = None
    rgb_bounds = None
    rgb_transform = None

    # --- RGB ---
    try:
        with rasterio.open(rgb_path) as ds:
            rgb_crs = ds.crs
            rgb_bounds = ds.bounds
            rgb_transform = ds.transform
            data = ds.read()
            if data.shape != (EXPECTED_RGB_BANDS, EXPECTED_TILE_SIZE, EXPECTED_TILE_SIZE):
                errors.append(
                    f"RGB shape {data.shape}, expected "
                    f"({EXPECTED_RGB_BANDS}, {EXPECTED_TILE_SIZE}, {EXPECTED_TILE_SIZE})"
                )
            if data.dtype != np.uint8:
                errors.append(f"RGB dtype {data.dtype}, expected uint8")
            if np.any(np.isnan(data.astype(float))):
                errors.append("RGB contains NaN values")
    except Exception as e:
        errors.append(f"RGB open/read error: {e}")

    # --- Label (positive tiles only) ---
    if train_class == "positive":
        try:
            with rasterio.open(lbl_path) as ds:
                lbl_crs = ds.crs
                lbl_bounds = ds.bounds
                lbl_transform = ds.transform
                data = ds.read(1)  # single band
                if data.shape != (EXPECTED_TILE_SIZE, EXPECTED_TILE_SIZE):
                    errors.append(
                        f"Label shape {data.shape}, expected "
                        f"({EXPECTED_TILE_SIZE}, {EXPECTED_TILE_SIZE})"
                    )
                if data.dtype != np.uint8:
                    errors.append(f"Label dtype {data.dtype}, expected uint8")
                unique_vals = set(np.unique(data))
                if not unique_vals.issubset(VALID_LABEL_VALUES):
                    invalid = unique_vals - VALID_LABEL_VALUES
                    errors.append(f"Label has invalid values: {invalid}")
                detail["n_rts_pixels"] = int(np.sum(data == 1))
                detail["n_total_pixels"] = int(data.size)

                # CRS/bounds/transform match RGB
                if rgb_crs is not None:
                    if str(lbl_crs) != str(rgb_crs):
                        errors.append(
                            f"Label CRS {lbl_crs} != RGB CRS {rgb_crs}"
                        )
                    if lbl_bounds != rgb_bounds:
                        errors.append("Label bounds != RGB bounds")
                    if lbl_transform != rgb_transform:
                        errors.append("Label transform != RGB transform")
        except Exception as e:
            errors.append(f"Label open/read error: {e}")

    # --- EXTRA ---
    if has_extra:
        extra_path = f"/vsigs/{bucket_name}/{prefix}/EXTRA/{tile_id}.tif"
        try:
            with rasterio.open(extra_path) as ds:
                ext_crs = ds.crs
                ext_bounds = ds.bounds
                ext_transform = ds.transform
                data = ds.read()
                if data.shape != (EXPECTED_EXTRA_BANDS, EXPECTED_TILE_SIZE, EXPECTED_TILE_SIZE):
                    errors.append(
                        f"EXTRA shape {data.shape}, expected "
                        f"({EXPECTED_EXTRA_BANDS}, {EXPECTED_TILE_SIZE}, {EXPECTED_TILE_SIZE})"
                    )
                if not np.all(np.isfinite(data)):
                    n_bad = int(np.sum(~np.isfinite(data)))
                    errors.append(f"EXTRA has {n_bad} non-finite values")

                # CRS/bounds/transform match RGB
                if rgb_crs is not None:
                    if str(ext_crs) != str(rgb_crs):
                        errors.append(
                            f"EXTRA CRS {ext_crs} != RGB CRS {rgb_crs}"
                        )
                    if ext_bounds != rgb_bounds:
                        errors.append("EXTRA bounds != RGB bounds")
                    if ext_transform != rgb_transform:
                        errors.append("EXTRA transform != RGB transform")
        except Exception as e:
            errors.append(f"EXTRA open/read error: {e}")

    # --- CRS check against expected ---
    if rgb_crs is not None and str(rgb_crs) != EXPECTED_CRS:
        errors.append(f"CRS is {rgb_crs}, expected {EXPECTED_CRS}")

    return errors, detail


def check_rasters(bucket_name: str, prefix: str,
                  metadata_df: pd.DataFrame | None,
                  has_extra: bool,
                  rgb_tile_ids: set[str] | None = None,
                  ) -> tuple[CheckResult, list[dict]]:
    """Validate raster files on a sample of tiles.

    With metadata: samples all positive tiles + up to 200 random negatives.
    Without metadata: samples up to 200 random tiles from PLANET-RGB.

    Args:
        bucket_name: GCS bucket name.
        prefix: Root prefix.
        metadata_df: Loaded metadata DataFrame, or None if missing.
        has_extra: Whether EXTRA/ directory exists.
        rgb_tile_ids: Pre-listed RGB tile IDs (used when metadata is missing).

    Returns:
        Tuple of (CheckResult, per-tile detail list).
    """
    messages: list[str] = []
    all_details: list[dict] = []
    error_count = 0

    rng = random.Random(SEED)
    class_lookup: dict[str, str] = {}

    if metadata_df is not None and "TrainClass" in metadata_df.columns:
        # Build sample from metadata
        positive_ids = metadata_df[
            metadata_df["TrainClass"] == "positive"
        ]["Tile_ID"].astype(str).tolist()
        negative_ids = metadata_df[
            metadata_df["TrainClass"] == "negative"
        ]["Tile_ID"].astype(str).tolist()

        sampled_negatives = rng.sample(
            negative_ids, min(NEGATIVE_SAMPLE_SIZE, len(negative_ids)),
        )
        sampled_ids = positive_ids + sampled_negatives
        messages.append(
            f"  Sampling {len(positive_ids)} positives + "
            f"{len(sampled_negatives)} negatives = {len(sampled_ids)} tiles"
        )
        class_lookup = dict(zip(
            metadata_df["Tile_ID"].astype(str),
            metadata_df["TrainClass"],
        ))
    else:
        # No metadata — sample random tiles from RGB directory
        if rgb_tile_ids is None or len(rgb_tile_ids) == 0:
            return CheckResult(
                "Rasters", False,
                ["  SKIPPED — no metadata and no RGB tiles found"],
            ), []
        all_rgb = sorted(rgb_tile_ids)
        n_sample = min(NEGATIVE_SAMPLE_SIZE, len(all_rgb))
        sampled_ids = rng.sample(all_rgb, n_sample)
        messages.append(
            f"  metadata.csv not available — sampling {n_sample} random "
            f"tiles from {len(all_rgb)} in PLANET-RGB"
        )

    # Set GDAL env for cloud reads
    os.environ["GDAL_HTTP_TIMEOUT"] = "30"
    os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
    os.environ["GDAL_HTTP_RETRY_DELAY"] = "5"
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

    failed_tiles: list[str] = []
    error_categories: dict[str, int] = {}

    for tile_id in tqdm(sampled_ids, desc="Checking rasters", unit="tile"):
        try:
            tile_class = class_lookup.get(tile_id, "positive")
            errors, detail = _validate_tile(
                bucket_name, prefix, tile_id, has_extra,
                train_class=tile_class,
            )
            detail["train_class"] = tile_class
            all_details.append(detail)
            if errors:
                error_count += len(errors)
                failed_tiles.append(tile_id)
                for err in errors:
                    # Categorise by first word(s) before the detail
                    cat = err.split(":")[0].split("!=")[0].strip()
                    error_categories[cat] = error_categories.get(cat, 0) + 1
                    logger.debug("Tile %s: %s", tile_id, err)
        except Exception as e:
            error_count += 1
            failed_tiles.append(tile_id)
            error_categories["unexpected error"] = (
                error_categories.get("unexpected error", 0) + 1
            )
            logger.debug("Tile %s: unexpected error: %s", tile_id, e)

    passed = error_count == 0
    if passed:
        messages.append(f"  All {len(sampled_ids)} sampled tiles passed")
    else:
        messages.append(
            f"  {len(failed_tiles)}/{len(sampled_ids)} tiles had errors "
            f"({error_count} total)"
        )
        for cat, cnt in sorted(error_categories.items(), key=lambda x: -x[1]):
            messages.append(f"    {cnt}x {cat}")
        sample = failed_tiles[:5]
        messages.append(
            f"  Example tile IDs: {sample}"
            f"{'...' if len(failed_tiles) > 5 else ''}"
        )

    return CheckResult("Rasters", passed, messages), all_details


# ---------------------------------------------------------------------------
# Check 6: Label semantics
# ---------------------------------------------------------------------------

def check_label_semantics(raster_details: list[dict]) -> CheckResult:
    """Check that label pixel counts match TrainClass.

    Positive tiles must have >= 1 pixel == 1.
    Negative tiles must have 0 pixels == 1.

    Args:
        raster_details: Per-tile detail dicts from check_rasters.

    Returns:
        CheckResult with mismatches.
    """
    # Check if we have TrainClass info at all
    has_class_info = any(
        d.get("train_class") in ("positive", "negative")
        for d in raster_details
    )
    if not raster_details:
        return CheckResult(
            "Label Semantics", False,
            ["  SKIPPED — no raster details available"],
        )
    if not has_class_info:
        # No metadata — report what we can observe
        n_with_rts = sum(1 for d in raster_details if d.get("n_rts_pixels", 0) > 0)
        n_without = len(raster_details) - n_with_rts
        return CheckResult(
            "Label Semantics", False,
            [
                "  SKIPPED — metadata.csv not available, cannot verify "
                "TrainClass consistency",
                f"  Observed: {n_with_rts} tiles with RTS pixels, "
                f"{n_without} tiles without (out of {len(raster_details)} sampled)",
            ],
        )

    messages: list[str] = []
    passed = True

    pos_no_rts: list[str] = []
    neg_with_rts: list[str] = []

    for detail in raster_details:
        tile_id = detail["tile_id"]
        train_class = detail.get("train_class", "Unknown")
        n_rts = detail.get("n_rts_pixels", 0)

        if train_class == "positive" and n_rts == 0:
            pos_no_rts.append(tile_id)
        elif train_class == "negative" and n_rts > 0:
            neg_with_rts.append(tile_id)

    if pos_no_rts:
        passed = False
        messages.append(
            f"  {len(pos_no_rts)} positive tiles have 0 RTS pixels: "
            f"{pos_no_rts[:10]}{'...' if len(pos_no_rts) > 10 else ''}"
        )
    if neg_with_rts:
        passed = False
        messages.append(
            f"  {len(neg_with_rts)} negative tiles have >0 RTS pixels: "
            f"{neg_with_rts[:10]}{'...' if len(neg_with_rts) > 10 else ''}"
        )
    if passed:
        n_pos = sum(1 for d in raster_details if d.get("train_class") == "positive")
        n_neg = sum(1 for d in raster_details if d.get("train_class") == "negative")
        messages.append(
            f"  All {n_pos} positive tiles have >=1 RTS pixel, "
            f"all {n_neg} negative tiles have 0"
        )

    return CheckResult("Label Semantics", passed, messages)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_statistics(metadata_df: pd.DataFrame | None,
                       splits: dict | None,
                       raster_details: list[dict],
                       version_info: dict | None) -> str:
    """Compute and format dataset statistics.

    Args:
        metadata_df: Loaded metadata DataFrame, or None if missing.
        splits: Parsed splits.yaml dict, or None if missing.
        raster_details: Per-tile detail dicts from raster check.
        version_info: Parsed version.json or None.

    Returns:
        Formatted statistics string.
    """
    lines: list[str] = []

    if metadata_df is not None and "TrainClass" in metadata_df.columns:
        # --- Tile counts ---
        total = len(metadata_df)
        n_pos = int((metadata_df["TrainClass"] == "positive").sum())
        n_neg = int((metadata_df["TrainClass"] == "negative").sum())
        ratio = f"{n_pos}:{n_neg}" if n_neg > 0 else "N/A"
        ratio_float = f"1:{n_neg / n_pos:.1f}" if n_pos > 0 else "N/A"

        lines.append("Tile Counts")
        lines.append(f"  Total tiles:    {total}")
        lines.append(f"  Positive:       {n_pos} ({100 * n_pos / total:.1f}%)")
        lines.append(f"  Negative:       {n_neg} ({100 * n_neg / total:.1f}%)")
        lines.append(f"  Pos:Neg ratio:  {ratio} ({ratio_float})")
        lines.append("")

        # --- Per-split breakdown ---
        if splits is not None:
            # val_balanced shares regions with val_realistic by design.
            # Give priority to val_realistic so its tile count is visible in stats.
            _PRIORITY = {"train": 0, "val_realistic": 1, "test_realistic": 2, "val_balanced": 3}
            region_to_split: dict[str, str] = {}
            for split_name, regions in sorted(splits.items(), key=lambda kv: _PRIORITY.get(kv[0], 99)):
                if isinstance(regions, list):
                    for r in regions:
                        if r not in region_to_split:  # first-seen wins (lower priority number)
                            region_to_split[r] = split_name

            df = metadata_df.copy()
            df["split"] = df["RegionName"].map(region_to_split)

            lines.append("Per-Split Breakdown")
            for split_name in splits:
                split_df = df[df["split"] == split_name]
                s_pos = int((split_df["TrainClass"] == "positive").sum())
                s_neg = int((split_df["TrainClass"] == "negative").sum())
                s_total = len(split_df)
                lines.append(
                    f"  {split_name:10s}: {s_pos:>5d} pos / {s_neg:>5d} neg = "
                    f"{s_total:>5d} total"
                )
            unassigned_df = df[df["split"].isna()]
            if len(unassigned_df) > 0:
                lines.append(f"  unassigned: {len(unassigned_df)} tiles")
            lines.append("")
        else:
            lines.append("Per-Split Breakdown")
            lines.append("  splits.yaml not available")
            lines.append("")

        # --- Per-region breakdown ---
        if "RegionName" in metadata_df.columns:
            lines.append("Per-Region Breakdown")
            region_stats = (
                metadata_df.groupby("RegionName")["TrainClass"]
                .value_counts()
                .unstack(fill_value=0)
            )
            for region in sorted(region_stats.index):
                r_pos = int(region_stats.loc[region].get("positive", 0))
                r_neg = int(region_stats.loc[region].get("negative", 0))
                r_total = r_pos + r_neg
                split_label = ""
                if splits is not None:
                    split_label = f" [{region_to_split.get(region, 'unassigned')}]"
                lines.append(
                    f"  {region:40s}: {r_total:>5d} tiles "
                    f"({r_pos:>4d} pos, {r_neg:>4d} neg){split_label}"
                )
            lines.append("")

        # --- Unique RTS UIDs ---
        if "UIDs" in metadata_df.columns:
            all_uids: set[str] = set()
            pos_mask = metadata_df["TrainClass"] == "positive"
            for uids_str in metadata_df.loc[pos_mask, "UIDs"].dropna():
                uids_str = str(uids_str).strip()
                if uids_str:
                    for uid in uids_str.split(","):
                        uid = uid.strip()
                        if uid:
                            all_uids.add(uid)
            lines.append(f"Unique RTS UIDs: {len(all_uids)}")
            lines.append("")
    else:
        lines.append("Tile Counts")
        lines.append("  metadata.csv not available — cannot compute tile statistics")
        lines.append("")

    # --- Label pixel coverage (sampled rasters) ---
    if raster_details:
        pos_details = [
            d for d in raster_details
            if d.get("train_class") == "positive" and d.get("n_total_pixels", 0) > 0
        ]
        if pos_details:
            lines.append("Label Pixel Coverage (sampled positives)")
            bucket_1_10 = 0
            bucket_10_50 = 0
            bucket_50_plus = 0
            for d in pos_details:
                coverage = d["n_rts_pixels"] / d["n_total_pixels"] * 100
                if coverage <= 10:
                    bucket_1_10 += 1
                elif coverage <= 50:
                    bucket_10_50 += 1
                else:
                    bucket_50_plus += 1
            lines.append(f"  1-10%  RTS coverage: {bucket_1_10} tiles")
            lines.append(f"  10-50% RTS coverage: {bucket_10_50} tiles")
            lines.append(f"  >50%   RTS coverage: {bucket_50_plus} tiles")
        else:
            # No metadata — just report observed RTS pixel distribution
            n_with_rts = sum(
                1 for d in raster_details if d.get("n_rts_pixels", 0) > 0
            )
            lines.append("Label Pixel Observations (sampled rasters)")
            lines.append(
                f"  {n_with_rts}/{len(raster_details)} sampled tiles "
                f"contain RTS pixels"
            )
        lines.append("")

    # --- Data version ---
    lines.append("Data Version")
    if version_info:
        lines.append(f"  {json.dumps(version_info, indent=2)}")
    else:
        lines.append("  version.json not found")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> None:
    """Configure logging to console (INFO) and file (DEBUG).

    Args:
        log_path: Path to the log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
    ))

    logger.addHandler(console)
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(results: list[CheckResult], stats: str,
                  bucket_url: str, version_str: str) -> str:
    """Format the full validation report.

    Args:
        results: List of CheckResults from all checks.
        stats: Formatted statistics string.
        bucket_url: Original bucket URL.
        version_str: Data version string.

    Returns:
        Complete report as a string.
    """
    sep = "=" * 60
    lines: list[str] = []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(sep)
    lines.append("  RTS DATA BUCKET VALIDATION REPORT")
    lines.append(f"  Bucket:  {bucket_url}")
    lines.append(f"  Version: {version_str}")
    lines.append(f"  Date:    {now}")
    lines.append(sep)
    lines.append("")

    for i, result in enumerate(results, 1):
        status = "PASSED" if result.passed else "FAILED"
        lines.append(f"--- Check {i}/{len(results)}: {result.name} ---")
        lines.append(f"  {status}")
        for msg in result.messages:
            lines.append(msg)
        lines.append("")

    lines.append(sep)
    lines.append("  STATISTICS")
    lines.append(sep)
    lines.append("")
    lines.append(stats)
    lines.append("")

    n_passed = sum(1 for r in results if r.passed)
    n_failed = len(results) - n_passed
    lines.append(sep)
    lines.append(f"  RESULT: {n_passed}/{len(results)} passed, {n_failed}/{len(results)} failed")
    lines.append(f"  Exit code: {0 if n_failed == 0 else 1}")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate RTS dataset in a GCS bucket.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket URL (e.g. gs://abrupt_thaw/RTS_MODEL_V2/DATA)",
    )
    return parser.parse_args()


def main() -> None:
    """Run all validation checks and print report."""
    args = parse_args()
    bucket_url = args.bucket
    bucket_name, prefix = parse_bucket_url(bucket_url)

    # Create GCS client
    try:
        client = storage.Client()
        # Quick connectivity test
        client.get_bucket(bucket_name)
    except Exception as e:
        print(f"ERROR: Cannot access bucket '{bucket_name}': {e}")
        sys.exit(1)

    # Read version.json (optional)
    version_info: dict | None = None
    version_str = "unknown"
    try:
        version_text = read_blob_text(
            client, bucket_name, f"{prefix}/version.json",
        )
        version_info = json.loads(version_text)
        version_str = str(version_info.get("version", "unknown"))
    except Exception:
        pass

    # Set up logging
    project_root = Path(__file__).resolve().parent.parent
    log_path = project_root / "docs" / f"data_check_v{version_str}.log"
    setup_logging(log_path)
    logger.info("Starting data bucket validation: %s", bucket_url)

    # Read metadata.csv (may not exist yet)
    metadata_df: pd.DataFrame | None = None
    try:
        meta_text = read_blob_text(
            client, bucket_name, f"{prefix}/metadata.csv",
        )
        metadata_df = pd.read_csv(io.StringIO(meta_text))
        logger.info("Loaded metadata.csv: %d rows", len(metadata_df))
    except Exception as e:
        logger.warning("Cannot read metadata.csv: %s", e)

    # Read splits.yaml (may not exist yet)
    splits: dict | None = None
    try:
        splits_text = read_blob_text(
            client, bucket_name, f"{prefix}/splits.yaml",
        )
        splits = yaml.safe_load(splits_text)
        logger.info("Loaded splits.yaml: %d splits", len(splits))
    except Exception as e:
        logger.warning("Cannot read splits.yaml: %s", e)

    # Run all checks
    results: list[CheckResult] = []

    # Check 1: Layout
    logger.info("Running Check 1/6: Layout")
    layout_result, has_extra, has_version = check_layout(
        client, bucket_name, prefix,
    )
    results.append(layout_result)

    # Check 2: Tile correspondence (also collects RGB tile IDs for later)
    logger.info("Running Check 2/6: Tile Correspondence")
    rgb_tile_ids = list_tif_tile_ids(client, bucket_name, prefix, "PLANET-RGB")
    results.append(check_tile_correspondence(
        client, bucket_name, prefix, metadata_df, has_extra,
        rgb_ids=rgb_tile_ids,
    ))

    # Check 3: Metadata schema
    logger.info("Running Check 3/6: Metadata Schema")
    results.append(check_metadata_schema(metadata_df))

    # Check 4: Splits
    logger.info("Running Check 4/6: Splits")
    results.append(check_splits(splits, metadata_df))

    # Check 5: Rasters
    logger.info("Running Check 5/6: Rasters")
    raster_result, raster_details = check_rasters(
        bucket_name, prefix, metadata_df, has_extra,
        rgb_tile_ids=rgb_tile_ids,
    )
    results.append(raster_result)

    # Check 6: Label semantics
    logger.info("Running Check 6/6: Label Semantics")
    results.append(check_label_semantics(raster_details))

    # Compute statistics
    stats = compute_statistics(metadata_df, splits, raster_details, version_info)

    # Format and output report
    report = format_report(results, stats, bucket_url, version_str)
    logger.info("\n%s", report)
    print(report)

    # Exit
    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
