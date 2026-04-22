"""Content-level validation of the v2.0 dataset (raster values, shapes, CRS, semantics).

Per data/datacheck.md §2. Runs on a random 5% sample of tiles.

Usage:
  python scripts/check_data_content.py --bucket gs://abruptthawmapping/training/v2.0
  python scripts/check_data_content.py --bucket /data/training/v2.0  # gcsfuse mount
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.splits import load_metadata, load_splits_yaml, assert_no_region_leakage  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _path(root: str, subdir: str, tid: str) -> str:
    return f"{root.rstrip('/')}/{subdir}/{tid}.tif"


def check_tile(
    tid: str, root: str, rgb_dir: str, labels_dir: str, extra_dir: str | None,
    expected_crs: str, expected_size: int, is_positive: bool,
) -> list[str]:
    """Return a list of human-readable error messages; empty = pass."""
    errors: list[str] = []

    # RGB
    try:
        with rasterio.open(_path(root, rgb_dir, tid)) as src:
            rgb = src.read(out_dtype="uint8")
            rgb_bounds = src.bounds
            rgb_transform = src.transform
            rgb_crs = src.crs.to_string() if src.crs else None
    except Exception as e:
        errors.append(f"{tid}: failed to open RGB: {e}")
        return errors

    if rgb.shape != (3, expected_size, expected_size):
        errors.append(f"{tid}: RGB shape {rgb.shape} != (3, {expected_size}, {expected_size})")
    if rgb.dtype != np.uint8:
        errors.append(f"{tid}: RGB dtype {rgb.dtype} != uint8")
    if np.isnan(rgb).any():
        errors.append(f"{tid}: RGB contains NaN")
    if rgb_crs != expected_crs:
        errors.append(f"{tid}: RGB CRS {rgb_crs} != {expected_crs}")

    # Label
    try:
        with rasterio.open(_path(root, labels_dir, tid)) as src:
            label = src.read(1, out_dtype="uint8")
            lab_bounds = src.bounds
            lab_transform = src.transform
            lab_crs = src.crs.to_string() if src.crs else None
    except Exception as e:
        errors.append(f"{tid}: failed to open label: {e}")
        return errors

    if label.shape != (expected_size, expected_size):
        errors.append(f"{tid}: label shape {label.shape} != ({expected_size}, {expected_size})")
    unique = set(np.unique(label).tolist())
    if not unique.issubset({0, 1, 255}):
        errors.append(f"{tid}: label has values {unique} outside {{0,1,255}}")
    if is_positive and not (label == 1).any():
        errors.append(f"{tid}: labeled Positive but no pixel == 1")
    if (not is_positive) and (label == 1).any():
        errors.append(f"{tid}: labeled Negative but contains pixel == 1")
    if lab_crs != expected_crs:
        errors.append(f"{tid}: label CRS {lab_crs} != {expected_crs}")
    if rgb_bounds != lab_bounds:
        errors.append(f"{tid}: RGB vs label bounds mismatch: {rgb_bounds} vs {lab_bounds}")
    if tuple(rgb_transform) != tuple(lab_transform):
        errors.append(f"{tid}: RGB vs label transform mismatch")

    # EXTRA (optional)
    if extra_dir is not None:
        try:
            with rasterio.open(_path(root, extra_dir, tid)) as src:
                extra = src.read(out_dtype="float32")
                ex_bounds = src.bounds
                ex_crs = src.crs.to_string() if src.crs else None
        except FileNotFoundError:
            return errors  # EXTRA is optional per dataset
        except Exception as e:
            errors.append(f"{tid}: failed to open EXTRA: {e}")
            return errors

        if extra.ndim != 3 or extra.shape[1:] != (expected_size, expected_size):
            errors.append(f"{tid}: EXTRA shape {extra.shape} not (*, {expected_size}, {expected_size})")
        if not np.isfinite(extra).all():
            errors.append(f"{tid}: EXTRA contains non-finite values")
        if ex_crs != expected_crs:
            errors.append(f"{tid}: EXTRA CRS {ex_crs} != {expected_crs}")
        if rgb_bounds != ex_bounds:
            errors.append(f"{tid}: RGB vs EXTRA bounds mismatch")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True,
                        help="Data root (gs:// URI or local path)")
    parser.add_argument("--rgb-dir", default="PLANET-RGB")
    parser.add_argument("--extra-dir", default="EXTRA")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--splits", default="splits.yaml")
    parser.add_argument("--crs", default="EPSG:3857")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--skip-extra", action="store_true",
                        help="Don't check the EXTRA directory even if present")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    random.seed(args.seed)
    root = args.bucket

    metadata = load_metadata(f"{root.rstrip('/')}/{args.metadata}")
    splits = load_splits_yaml(f"{root.rstrip('/')}/{args.splits}")
    assert_no_region_leakage(splits)

    all_regions = {r for rs in splits.values() for r in rs}
    unassigned = set(metadata["RegionName"]) - all_regions
    if unassigned:
        logger.warning("Regions present in metadata but absent from splits: %s", sorted(unassigned))
    missing = all_regions - set(metadata["RegionName"])
    if missing:
        logger.error("Splits reference regions not in metadata: %s", sorted(missing))
        return 1

    n = max(1, int(len(metadata) * args.sample_fraction))
    sample = metadata.sample(n=n, random_state=args.seed).to_dict("records")
    logger.info("Checking %d / %d tiles (%.1f%% sample)", n, len(metadata),
                100 * args.sample_fraction)

    extra_dir = None if args.skip_extra else args.extra_dir
    errors_total: list[str] = []
    for row in tqdm(sample, desc="content check"):
        errs = check_tile(
            tid=row["Tile_id"],
            root=root, rgb_dir=args.rgb_dir, labels_dir=args.labels_dir,
            extra_dir=extra_dir,
            expected_crs=args.crs, expected_size=args.tile_size,
            is_positive=(row["TrainClass"] == "Positive"),
        )
        errors_total.extend(errs)

    if errors_total:
        logger.error("FAILED with %d error(s):", len(errors_total))
        for e in errors_total[:50]:
            logger.error("  %s", e)
        if len(errors_total) > 50:
            logger.error("  ... %d more", len(errors_total) - 50)
        return 1

    logger.info("All %d sampled tiles passed.", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
