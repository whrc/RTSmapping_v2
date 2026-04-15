#!/usr/bin/env python3
"""Validate RTS dataset format in a GCS bucket.

Format check only — verifies file presence, naming, and folder structure
per data/data.md §3. Does NOT read raster contents.

See data/datacheck.md §1 for the full specification.

Usage:
    python scripts/check_data_format.py --bucket gs://abrupt_thaw/RTS_MODEL_V2/DATA
"""

import argparse
import io
import logging
import sys
from datetime import datetime, timezone
from typing import NamedTuple

import pandas as pd
import yaml
from google.cloud import storage

logger = logging.getLogger("check_data_format")

EXPECTED_METADATA_COLUMNS = [
    "Tile_id", "centroid_lat", "centroid_lon",
    "TrainClass", "RegionName", "UIDs",
]


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
# Check 1: Folder structure
# ---------------------------------------------------------------------------

def check_folder_structure(client: storage.Client, bucket_name: str,
                           prefix: str) -> tuple[CheckResult, bool]:
    """Check that the bucket root has the expected directories and files.

    Args:
        client: GCS client.
        bucket_name: Name of the bucket.
        prefix: Root prefix.

    Returns:
        Tuple of (CheckResult, has_extra).
    """
    messages: list[str] = []
    passed = True

    files, dirs = list_top_level_entries(client, bucket_name, prefix)

    required_dirs = {"PLANET-RGB/", "labels/"}
    required_files = {"metadata.csv", "splits.yaml"}
    optional_files = {"version.json"}

    for d in sorted(required_dirs):
        if d in dirs:
            messages.append(f"  Found required directory: {d}")
        else:
            messages.append(f"  MISSING required directory: {d}")
            passed = False

    has_extra = "EXTRA/" in dirs
    if has_extra:
        messages.append("  Found optional directory: EXTRA/")
    else:
        messages.append("  Optional directory not present: EXTRA/")

    for f in sorted(required_files):
        if f in files:
            messages.append(f"  Found required file: {f}")
        else:
            messages.append(f"  MISSING required file: {f}")
            passed = False

    for f in sorted(optional_files):
        if f in files:
            messages.append(f"  Found optional file: {f}")

    return CheckResult("Folder Structure", passed, messages), has_extra


# ---------------------------------------------------------------------------
# Check 2: File presence
# ---------------------------------------------------------------------------

def check_file_presence(client: storage.Client, bucket_name: str,
                        prefix: str, has_extra: bool) -> CheckResult:
    """Check that image directories contain .tif files only, no nesting.

    Args:
        client: GCS client.
        bucket_name: Name of the bucket.
        prefix: Root prefix.
        has_extra: Whether EXTRA/ directory exists.

    Returns:
        CheckResult with file presence details.
    """
    messages: list[str] = []
    passed = True

    dirs_to_check = ["PLANET-RGB", "labels"]
    if has_extra:
        dirs_to_check.append("EXTRA")

    for subdir in dirs_to_check:
        files, nested_dirs = list_top_level_entries(
            client, bucket_name, f"{prefix}/{subdir}",
        )

        if nested_dirs:
            passed = False
            sample = sorted(nested_dirs)[:5]
            messages.append(
                f"  {subdir}/: unexpected nested directories: {sample}"
            )

        non_tif = {f for f in files if not f.endswith(".tif")}
        if non_tif:
            passed = False
            sample = sorted(non_tif)[:5]
            messages.append(
                f"  {subdir}/: {len(non_tif)} non-.tif files: "
                f"{sample}{'...' if len(non_tif) > 5 else ''}"
            )

        tif_count = len(files) - len(non_tif)
        if tif_count == 0:
            passed = False
            messages.append(f"  {subdir}/: no .tif files found")
        else:
            messages.append(f"  {subdir}/: {tif_count} .tif files")

    return CheckResult("File Presence", passed, messages)


# ---------------------------------------------------------------------------
# Check 3: Naming convention
# ---------------------------------------------------------------------------

def check_naming_convention(tile_ids_by_dir: dict[str, set[str]]) -> CheckResult:
    """Check that tile IDs are numeric with consistent zero-padding.

    Args:
        tile_ids_by_dir: Dict mapping directory name to set of tile ID strings.

    Returns:
        CheckResult with naming details.
    """
    messages: list[str] = []
    passed = True

    all_ids: set[str] = set()
    for dir_name, tile_ids in tile_ids_by_dir.items():
        if not tile_ids:
            continue
        all_ids.update(tile_ids)

        non_numeric = {tid for tid in tile_ids if not tid.isdigit()}
        if non_numeric:
            passed = False
            sample = sorted(non_numeric)[:5]
            messages.append(
                f"  {dir_name}/: {len(non_numeric)} non-numeric tile IDs: "
                f"{sample}{'...' if len(non_numeric) > 5 else ''}"
            )

    numeric_ids = {tid for tid in all_ids if tid.isdigit()}
    if numeric_ids:
        widths = {len(tid) for tid in numeric_ids}
        if len(widths) > 1:
            passed = False
            messages.append(
                f"  Inconsistent zero-padding widths: {sorted(widths)}"
            )
        else:
            width = widths.pop()
            messages.append(
                f"  All {len(numeric_ids)} tile IDs numeric, "
                f"{width}-digit zero-padded"
            )
    elif not all_ids:
        messages.append("  No tile IDs found")

    return CheckResult("Naming Convention", passed, messages)


# ---------------------------------------------------------------------------
# Check 4: Tile ID correspondence
# ---------------------------------------------------------------------------

def check_tile_correspondence(tile_ids_by_dir: dict[str, set[str]],
                              metadata_df: pd.DataFrame | None,
                              ) -> CheckResult:
    """Check that tile ID sets match across directories and metadata.

    Args:
        tile_ids_by_dir: Dict mapping directory name to set of tile ID strings.
        metadata_df: Loaded metadata DataFrame, or None if file missing.

    Returns:
        CheckResult with mismatch details.
    """
    messages: list[str] = []
    passed = True

    sources: dict[str, set[str]] = dict(tile_ids_by_dir)

    if metadata_df is None:
        messages.append(
            "  metadata.csv not available — skipping metadata tile comparison"
        )
    elif "Tile_id" not in metadata_df.columns:
        passed = False
        messages.append(
            "  metadata.csv loaded but missing Tile_id column — "
            "cannot compare tile IDs"
        )
    else:
        meta_ids = set(metadata_df["Tile_id"].astype(str))
        sources["metadata.csv"] = meta_ids

    rgb_ids = sources.get("PLANET-RGB", set())
    for name, ids in sources.items():
        messages.append(f"  {name} tile count: {len(ids)}")

    for name, ids in sources.items():
        if name == "PLANET-RGB":
            continue
        only_in_rgb = rgb_ids - ids
        only_in_other = ids - rgb_ids
        if only_in_rgb:
            passed = False
            sample = sorted(only_in_rgb)[:5]
            messages.append(
                f"  In PLANET-RGB but not in {name} ({len(only_in_rgb)}): "
                f"{sample}{'...' if len(only_in_rgb) > 5 else ''}"
            )
        if only_in_other:
            passed = False
            sample = sorted(only_in_other)[:5]
            messages.append(
                f"  In {name} but not in PLANET-RGB ({len(only_in_other)}): "
                f"{sample}{'...' if len(only_in_other) > 5 else ''}"
            )

    if passed and len(sources) > 1:
        messages.append(
            f"  All {len(rgb_ids)} tile IDs consistent across "
            f"{len(sources)} sources"
        )

    return CheckResult("Tile ID Correspondence", passed, messages)


# ---------------------------------------------------------------------------
# Check 5: metadata.csv schema
# ---------------------------------------------------------------------------

def check_metadata_schema(metadata_df: pd.DataFrame | None) -> CheckResult:
    """Validate metadata.csv has required columns in correct order.

    Args:
        metadata_df: Loaded metadata DataFrame, or None if file missing.

    Returns:
        CheckResult with column validation details.
    """
    if metadata_df is None:
        return CheckResult(
            "Metadata Schema", False,
            ["  SKIPPED — metadata.csv not available"],
        )

    messages: list[str] = []
    passed = True

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
                f"  Column order mismatch: expected "
                f"{EXPECTED_METADATA_COLUMNS}, got {actual_cols}"
            )
    else:
        messages.append(
            f"  All expected columns present in correct order "
            f"({len(metadata_df)} rows)"
        )

    return CheckResult("Metadata Schema", passed, messages)


# ---------------------------------------------------------------------------
# Check 6: splits.yaml structure
# ---------------------------------------------------------------------------

def check_splits_structure(splits: dict | None) -> CheckResult:
    """Validate splits.yaml is valid YAML with list-of-string values.

    Args:
        splits: Parsed splits.yaml dict, or None if file missing.

    Returns:
        CheckResult with structure validation details.
    """
    if splits is None:
        return CheckResult(
            "Splits Structure", False,
            ["  SKIPPED — splits.yaml not available"],
        )

    messages: list[str] = []
    passed = True

    if not isinstance(splits, dict):
        return CheckResult(
            "Splits Structure", False,
            [f"  splits.yaml root is {type(splits).__name__}, expected dict"],
        )

    for split_name, regions in splits.items():
        if not isinstance(regions, list):
            passed = False
            messages.append(
                f"  Split '{split_name}' value is "
                f"{type(regions).__name__}, expected list"
            )
            continue
        non_str = [r for r in regions if not isinstance(r, str)]
        if non_str:
            passed = False
            messages.append(
                f"  Split '{split_name}' contains non-string entries: "
                f"{non_str[:3]}"
            )
        else:
            messages.append(
                f"  Split '{split_name}': {len(regions)} regions"
            )

    if passed:
        total_regions = sum(
            len(r) for r in splits.values() if isinstance(r, list)
        )
        messages.append(
            f"  {len(splits)} splits, {total_regions} total region assignments"
        )

    return CheckResult("Splits Structure", passed, messages)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(metadata_df: pd.DataFrame | None,
                    splits: dict | None) -> str:
    """Compute dataset summary statistics for the report.

    Args:
        metadata_df: Loaded metadata DataFrame, or None.
        splits: Parsed splits.yaml dict, or None.

    Returns:
        Formatted summary string.
    """
    lines: list[str] = []

    if metadata_df is None or "TrainClass" not in metadata_df.columns:
        lines.append("  metadata.csv not available — cannot compute summary")
        return "\n".join(lines)

    total = len(metadata_df)
    n_pos = int((metadata_df["TrainClass"] == "Positive").sum())
    n_neg = int((metadata_df["TrainClass"] == "Negative").sum())
    ratio_str = f"1:{n_neg / n_pos:.1f}" if n_pos > 0 else "N/A"

    lines.append("Tile Counts")
    lines.append(f"  Total:     {total}")
    lines.append(f"  Positive:  {n_pos}")
    lines.append(f"  Negative:  {n_neg}")
    lines.append(f"  Ratio:     {ratio_str}")
    lines.append("")

    if splits is not None and "RegionName" in metadata_df.columns:
        region_to_split: dict[str, str] = {}
        for split_name, regions in splits.items():
            if isinstance(regions, list):
                for r in regions:
                    region_to_split[r] = split_name

        df = metadata_df.copy()
        df["split"] = df["RegionName"].map(region_to_split)

        lines.append("Per-Split Breakdown")
        for split_name in splits:
            split_df = df[df["split"] == split_name]
            s_pos = int((split_df["TrainClass"] == "Positive").sum())
            s_neg = int((split_df["TrainClass"] == "Negative").sum())
            s_total = len(split_df)
            n_regions = len(splits.get(split_name, []))
            lines.append(
                f"  {split_name:10s}: {s_total:>5d} tiles "
                f"({s_pos:>4d} pos, {s_neg:>4d} neg), "
                f"{n_regions} regions"
            )
        unassigned = df[df["split"].isna()]
        if len(unassigned) > 0:
            lines.append(f"  unassigned: {len(unassigned)} tiles")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(results: list[CheckResult], summary: str,
                  bucket_url: str) -> str:
    """Format the full validation report.

    Args:
        results: List of CheckResults from all checks.
        summary: Formatted summary string.
        bucket_url: Original bucket URL.

    Returns:
        Complete report as a string.
    """
    sep = "=" * 60
    lines: list[str] = []

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(sep)
    lines.append("  RTS DATA FORMAT CHECK")
    lines.append(f"  Bucket:  {bucket_url}")
    lines.append(f"  Date:    {now}")
    lines.append(sep)
    lines.append("")

    for i, result in enumerate(results, 1):
        status = "PASSED" if result.passed else "FAILED"
        lines.append(
            f"--- Check {i}/{len(results)}: {result.name} [{status}] ---"
        )
        for msg in result.messages:
            lines.append(msg)
        lines.append("")

    lines.append(sep)
    lines.append("  SUMMARY")
    lines.append(sep)
    lines.append("")
    lines.append(summary)
    lines.append("")

    n_passed = sum(1 for r in results if r.passed)
    n_failed = len(results) - n_passed
    lines.append(sep)
    lines.append(
        f"  RESULT: {n_passed}/{len(results)} passed, "
        f"{n_failed}/{len(results)} failed"
    )
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all format validation checks and print report."""
    parser = argparse.ArgumentParser(
        description="Validate RTS dataset format in a GCS bucket.",
    )
    parser.add_argument(
        "--bucket",
        required=True,
        help="GCS bucket URL (e.g. gs://abrupt_thaw/RTS_MODEL_V2/DATA)",
    )
    args = parser.parse_args()

    bucket_url = args.bucket
    bucket_name, prefix = parse_bucket_url(bucket_url)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        client = storage.Client()
        client.get_bucket(bucket_name)
    except Exception as e:
        logger.error("Cannot access bucket '%s': %s", bucket_name, e)
        sys.exit(1)

    # Load metadata.csv (format check proceeds without it)
    metadata_df: pd.DataFrame | None = None
    try:
        meta_text = read_blob_text(
            client, bucket_name, f"{prefix}/metadata.csv",
        )
        metadata_df = pd.read_csv(io.StringIO(meta_text))
        logger.info("Loaded metadata.csv: %d rows", len(metadata_df))
    except Exception as e:
        logger.warning("Cannot read metadata.csv: %s", e)

    # Load splits.yaml (format check proceeds without it)
    splits: dict | None = None
    try:
        splits_text = read_blob_text(
            client, bucket_name, f"{prefix}/splits.yaml",
        )
        splits = yaml.safe_load(splits_text)
        logger.info("Loaded splits.yaml: %d splits", len(splits))
    except Exception as e:
        logger.warning("Cannot read splits.yaml: %s", e)

    results: list[CheckResult] = []

    # Check 1: Folder structure
    logger.info("Running Check 1/6: Folder Structure")
    folder_result, has_extra = check_folder_structure(
        client, bucket_name, prefix,
    )
    results.append(folder_result)

    # Check 2: File presence
    logger.info("Running Check 2/6: File Presence")
    results.append(check_file_presence(client, bucket_name, prefix, has_extra))

    # Collect tile IDs for checks 3 and 4
    tile_ids_by_dir: dict[str, set[str]] = {
        "PLANET-RGB": list_tif_tile_ids(
            client, bucket_name, prefix, "PLANET-RGB",
        ),
        "labels": list_tif_tile_ids(
            client, bucket_name, prefix, "labels",
        ),
    }
    if has_extra:
        tile_ids_by_dir["EXTRA"] = list_tif_tile_ids(
            client, bucket_name, prefix, "EXTRA",
        )

    # Check 3: Naming convention
    logger.info("Running Check 3/6: Naming Convention")
    results.append(check_naming_convention(tile_ids_by_dir))

    # Check 4: Tile ID correspondence
    logger.info("Running Check 4/6: Tile ID Correspondence")
    results.append(check_tile_correspondence(tile_ids_by_dir, metadata_df))

    # Check 5: metadata.csv schema
    logger.info("Running Check 5/6: Metadata Schema")
    results.append(check_metadata_schema(metadata_df))

    # Check 6: splits.yaml structure
    logger.info("Running Check 6/6: Splits Structure")
    results.append(check_splits_structure(splits))

    # Summary and report
    summary = compute_summary(metadata_df, splits)
    report = format_report(results, summary, bucket_url)
    print(report)

    all_passed = all(r.passed for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
