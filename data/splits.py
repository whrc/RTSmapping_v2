"""Split resolution: metadata.csv + splits.yaml → per-split tile lists.

See data/data.md §6 for the spatial-blocking split spec.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REQUIRED_METADATA_COLUMNS = [
    "Tile_id",
    "centroid_lat",
    "centroid_lon",
    "TrainClass",
    "RegionName",
    "UIDs",
]

VALID_SPLITS = ["train", "val_balanced", "val_realistic", "test_realistic"]


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load metadata.csv; validate required columns and TrainClass values."""
    df = pd.read_csv(path, dtype={"Tile_id": str, "UIDs": str})
    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")
    bad_class = set(df["TrainClass"].unique()) - {"Positive", "Negative"}
    if bad_class:
        raise ValueError(f"TrainClass must be Positive or Negative; saw {bad_class}")
    df["UIDs"] = df["UIDs"].fillna("")
    return df


def load_splits_yaml(path: str | Path) -> dict[str, list[str]]:
    """Load splits.yaml. Returns {split_name: [region_name, ...]}."""
    with Path(path).open("r") as f:
        splits = yaml.safe_load(f)
    if not isinstance(splits, dict):
        raise ValueError(f"splits.yaml root must be a mapping: {path}")
    for split_name, regions in splits.items():
        if split_name not in VALID_SPLITS:
            raise ValueError(f"Unknown split '{split_name}'; expected one of {VALID_SPLITS}")
        if not isinstance(regions, list) or not all(isinstance(r, str) for r in regions):
            raise ValueError(f"splits.yaml[{split_name}] must be a list of region names")
    return splits


def assert_no_region_leakage(splits: dict[str, list[str]]) -> None:
    """Raise if any region appears in more than one split."""
    seen: dict[str, str] = {}
    for split_name, regions in splits.items():
        for r in regions:
            if r in seen:
                raise ValueError(
                    f"Region '{r}' appears in both '{seen[r]}' and '{split_name}' "
                    "— splits must be spatially disjoint"
                )
            seen[r] = split_name


def get_tile_ids(
    split_name: str,
    metadata: pd.DataFrame,
    splits: dict[str, list[str]],
    class_filter: str | None = None,
) -> list[str]:
    """Return tile IDs belonging to a split.

    Args:
        split_name: one of VALID_SPLITS.
        metadata: output of load_metadata().
        splits: output of load_splits_yaml().
        class_filter: optional "Positive" or "Negative" filter.

    Returns:
        List of Tile_id strings in arbitrary but deterministic order (sorted).
    """
    if split_name not in splits:
        raise KeyError(f"Split '{split_name}' not in splits.yaml; available: {list(splits)}")
    regions = set(splits[split_name])
    mask = metadata["RegionName"].isin(regions)
    if class_filter is not None:
        if class_filter not in ("Positive", "Negative"):
            raise ValueError(f"class_filter must be Positive/Negative, got {class_filter!r}")
        mask &= metadata["TrainClass"] == class_filter
    return sorted(metadata.loc[mask, "Tile_id"].tolist())


def split_summary(
    metadata: pd.DataFrame, splits: dict[str, list[str]]
) -> dict[str, dict[str, int]]:
    """Per-split tile counts broken down by class. Useful for sanity checks and logs."""
    out: dict[str, dict[str, int]] = {}
    for split_name in splits:
        tids = get_tile_ids(split_name, metadata, splits)
        sub = metadata[metadata["Tile_id"].isin(tids)]
        out[split_name] = {
            "total": len(sub),
            "positive": int((sub["TrainClass"] == "Positive").sum()),
            "negative": int((sub["TrainClass"] == "Negative").sum()),
            "n_regions": len(splits[split_name]),
        }
    return out
