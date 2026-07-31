"""Split resolution: metadata.csv + splits.yaml → per-split tile lists.

See data/data.md §6 for the spatial-blocking split spec.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

REQUIRED_METADATA_COLUMNS = [
    "Tile_ID",
    "centroid_lat",
    "centroid_lon",
    "TrainClass",
    "RegionName",
    "UIDs",
]

VALID_SPLITS = ["train", "val_balanced", "val_realistic", "test_realistic"]


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Load metadata.csv; validate required columns and TrainClass values."""
    df = pd.read_csv(path, dtype={"Tile_ID": str, "UIDs": str})
    missing = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")
    bad_class = set(df["TrainClass"].unique()) - {"positive", "negative"}
    if bad_class:
        raise ValueError(f"TrainClass must be Positive or Negative; saw {bad_class}")
    df["UIDs"] = df["UIDs"].fillna("")
    return df


def _open_text(path: str | Path):
    """Open a file for reading, supporting both local paths and gs:// URIs."""
    p = str(path)
    if p.startswith("gs://"):
        import gcsfs
        return gcsfs.GCSFileSystem(token="google_default").open(p[5:], "r")
    return Path(p).open("r")


def load_splits_yaml(path: str | Path) -> dict[str, list[str]]:
    """Load splits.yaml. Returns {split_name: [region_name, ...]}."""
    with _open_text(path) as f:
        splits = yaml.safe_load(f)
    if not isinstance(splits, dict):
        raise ValueError(f"splits.yaml root must be a mapping: {path}")
    for split_name, regions in splits.items():
        if split_name not in VALID_SPLITS:
            raise ValueError(f"Unknown split '{split_name}'; expected one of {VALID_SPLITS}")
        if not isinstance(regions, list) or not all(isinstance(r, str) for r in regions):
            raise ValueError(f"splits.yaml[{split_name}] must be a list of region names")
    return splits


def load_tile_allowlist(path: str | Path) -> set[str]:
    """Load a tile-id allowlist: one Tile_ID per line, blanks ignored.

    Used by `splits.tile_allowlist` to restrict every split to a common subset of
    tiles. Unlike `splits.train_positive_subset_pct` (train positives only) this
    applies to all splits and both classes, because it exists to remove a
    per-tile *availability* difference rather than to scale the data: an EXTRA
    channel that covers only part of the domain otherwise makes its own presence
    a label cue (docs/arcticdem_diagnostic.md §4).
    """
    with _open_text(path) as f:
        ids = {line.strip() for line in f if line.strip()}
    if not ids:
        raise ValueError(f"tile allowlist is empty: {path}")
    return ids


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
        class_filter: optional "positive" or "negative" filter.

    Returns:
        List of Tile_ID strings in arbitrary but deterministic order (sorted).
    """
    if split_name not in splits:
        raise KeyError(f"Split '{split_name}' not in splits.yaml; available: {list(splits)}")
    regions = set(splits[split_name])
    mask = metadata["RegionName"].isin(regions)
    if class_filter is not None:
        if class_filter not in ("positive", "negative"):
            raise ValueError(f"class_filter must be positive/negative, got {class_filter!r}")
        mask &= metadata["TrainClass"] == class_filter
    return sorted(metadata.loc[mask, "Tile_ID"].tolist())


def split_summary(
    metadata: pd.DataFrame, splits: dict[str, list[str]]
) -> dict[str, dict[str, int]]:
    """Per-split tile counts broken down by class. Useful for sanity checks and logs."""
    out: dict[str, dict[str, int]] = {}
    for split_name in splits:
        tids = get_tile_ids(split_name, metadata, splits)
        sub = metadata[metadata["Tile_ID"].isin(tids)]
        out[split_name] = {
            "total": len(sub),
            "positive": int((sub["TrainClass"] == "positive").sum()),
            "negative": int((sub["TrainClass"] == "negative").sum()),
            "n_regions": len(splits[split_name]),
        }
    return out


def load_metadata_multiroot(
    primary_root: str,
    metadata_csv: str,
    additional_roots: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load metadata across the primary + additional dataset roots.

    Multiscale POC (data.md §3.5): additional roots (e.g. the 0.5x re-stage)
    contribute extra tiles tagged with a per-tile `data_root` column that
    RTSDataset uses to resolve file paths. Shared by train.py and check_data.py.

    Returns (combined_metadata, primary_metadata). Without additional roots the
    two are the same frame and no `data_root` column is added.
    """
    def _md_path(root: str) -> str:
        return f"{root.rstrip('/')}/{metadata_csv}"

    primary = load_metadata(_md_path(primary_root))
    if not additional_roots:
        return primary, primary
    frames = [primary.assign(data_root=primary_root)]
    for root in additional_roots:
        frames.append(load_metadata(_md_path(root)).assign(data_root=root))
    combined = pd.concat(frames, ignore_index=True)
    dup = combined["Tile_ID"].duplicated()
    if dup.any():
        raise ValueError(
            f"Tile_ID collision across data roots: {combined.loc[dup, 'Tile_ID'].tolist()[:5]}"
        )
    return combined, primary
