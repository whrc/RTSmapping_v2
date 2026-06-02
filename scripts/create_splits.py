"""Generate splits.yaml + splits_summary.json from metadata.csv + region geometry.

Implements the spatial-block split logic from data/data.md §6 with four
priority-ordered constraints:

  1. Test has ≥ min_test_positives positive tiles                        (hard)
  2. Val has ≥ min_val_ecoregions distinct ecoregions                    (hard)
  3. Train has ≥ min_train_positive_fraction of total positives          (warn)
  4. Tile counts within ±max_drift of (train/val/test)_fraction targets  (hard)

Strategy: randomized greedy with retries. We treat region assignment as the
atomic unit (never split a region across splits). If no feasible assignment
is found after N attempts, exit with error — that means the constraint set
is too tight for the region pool and the user needs to loosen thresholds or
add regions.

Usage:
  python scripts/create_splits.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

# Make repo root importable when invoked as `python scripts/create_splits.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.splits import load_metadata  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


def load_region_ecoregions(geojson_path: str | Path) -> dict[str, str]:
    """Map RegionName → Ecoregion from the domain GeoJSON.

    If the GeoJSON lacks an 'Ecoregion' property, fall back to RegionName itself
    (each region is its own ecoregion — the ≥2 val ecoregions constraint then
    just means ≥2 val regions).
    """
    try:
        import geopandas as gpd
    except ImportError:
        logger.warning("geopandas not available; treating each region as its own ecoregion")
        return {}

    gdf = gpd.read_file(geojson_path)
    name_col = _first_present(gdf, ["RegionName", "region_name", "name", "ECO_NAME"])
    eco_col = _first_present(gdf, ["Ecoregion", "ecoregion", "ECO_NAME"])
    if name_col is None:
        raise ValueError(f"{geojson_path}: no RegionName-like column found")
    if eco_col is None:
        logger.info("No Ecoregion column in %s; using RegionName as ecoregion", geojson_path)
        return {str(r): str(r) for r in gdf[name_col]}
    return {str(r): str(e) for r, e in zip(gdf[name_col], gdf[eco_col])}


def _first_present(df: "pd.DataFrame", candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def region_stats(metadata: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metadata.csv → per-region (positives, negatives, total)."""
    g = (
        metadata.assign(
            is_pos=(metadata["TrainClass"] == "positive").astype(int),
            is_neg=(metadata["TrainClass"] == "negative").astype(int),
        )
        .groupby("RegionName", as_index=False)
        .agg(positive=("is_pos", "sum"), negative=("is_neg", "sum"))
    )
    g["total"] = g["positive"] + g["negative"]
    return g


def try_assignment(
    regions: pd.DataFrame,
    ecoregions: dict[str, str],
    cfg: dict,
    rng: random.Random,
) -> dict[str, list[str]] | None:
    """Produce a random region-to-split assignment; return None if constraints fail."""
    fracs = {
        "train": cfg["splits"]["train_fraction"],
        "val_realistic": cfg["splits"]["val_fraction"],
        "test_realistic": cfg["splits"]["test_fraction"],
    }
    max_drift = cfg["splits"]["max_drift"]
    min_test_pos = cfg["splits"]["min_test_positives"]
    min_val_eco = cfg["splits"]["min_val_ecoregions"]

    total_tiles = int(regions["total"].sum())
    total_pos = int(regions["positive"].sum())
    targets = {k: v * total_tiles for k, v in fracs.items()}

    shuffled = regions.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)

    assignment: dict[str, list[str]] = {"train": [], "val_realistic": [], "test_realistic": []}
    running: dict[str, int] = {k: 0 for k in assignment}

    # Greedy: put each region in the split furthest behind its target (by tile count).
    for _, row in shuffled.iterrows():
        gap = {k: targets[k] - running[k] for k in assignment}
        pick = max(gap, key=gap.get)
        assignment[pick].append(row["RegionName"])
        running[pick] += int(row["total"])

    # Constraint 1: test positives
    test_pos = _count_positives(assignment["test_realistic"], regions)
    if test_pos < min_test_pos:
        return None

    # Constraint 2: val ecoregions
    val_ecos = {ecoregions.get(r, r) for r in assignment["val_realistic"]}
    if len(val_ecos) < min_val_eco:
        return None

    # Constraint 4: ±drift on tile counts
    for split, target in targets.items():
        actual = running[split]
        if total_tiles > 0 and abs(actual - target) / total_tiles > max_drift:
            return None

    # Constraint 3: train ≥70% of positives — warning only, don't fail
    train_pos = _count_positives(assignment["train"], regions)
    if total_pos > 0 and train_pos / total_pos < cfg["splits"]["min_train_positive_fraction"]:
        logger.warning(
            "Train positive fraction %.1f%% below target %.0f%% — accepting anyway",
            100 * train_pos / total_pos,
            100 * cfg["splits"]["min_train_positive_fraction"],
        )

    # val_balanced = val_realistic regions (same geography, different eval-time ratio).
    assignment["val_balanced"] = list(assignment["val_realistic"])
    return assignment


def _count_positives(region_names: list[str], regions: pd.DataFrame) -> int:
    return int(regions.loc[regions["RegionName"].isin(region_names), "positive"].sum())


def generate_splits(
    metadata: pd.DataFrame,
    ecoregions: dict[str, str],
    cfg: dict,
    max_attempts: int = 500,
) -> tuple[dict[str, list[str]], dict]:
    """Run randomized retries until constraints are satisfied."""
    rng = random.Random(cfg.get("seed", 42))
    regions = region_stats(metadata)
    logger.info("Region pool: %d regions, %d tiles (%d pos)",
                len(regions), int(regions["total"].sum()), int(regions["positive"].sum()))

    for attempt in range(1, max_attempts + 1):
        assignment = try_assignment(regions, ecoregions, cfg, rng)
        if assignment is not None:
            logger.info("Found feasible split on attempt %d", attempt)
            summary = _build_summary(assignment, regions, ecoregions, attempts=attempt)
            return assignment, summary

    raise RuntimeError(
        f"No feasible split found in {max_attempts} attempts — "
        "loosen constraints in configs/baseline.yaml §splits or add regions."
    )


def _build_summary(
    assignment: dict[str, list[str]],
    regions: pd.DataFrame,
    ecoregions: dict[str, str],
    attempts: int,
) -> dict:
    summary: dict = {"attempts": attempts, "splits": {}}
    for split_name, region_names in assignment.items():
        sub = regions[regions["RegionName"].isin(region_names)]
        ecos = sorted({ecoregions.get(r, r) for r in region_names})
        summary["splits"][split_name] = {
            "n_regions": len(region_names),
            "n_ecoregions": len(ecos),
            "tiles": int(sub["total"].sum()),
            "positive": int(sub["positive"].sum()),
            "negative": int(sub["negative"].sum()),
            "ecoregions": ecos,
            "regions": sorted(region_names),
        }
    return summary

# mian function now handles path as string to allow // path intro for gcs format
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where to write splits.yaml and splits_summary.json "
                             "(default: data_root from config)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42), deterministic=cfg.get("deterministic", True))

    data_root = cfg["data"]["data_root"]
    metadata_path = f"{data_root.rstrip('/')}/{cfg['data']['metadata_csv']}"
    regions_geojson = cfg["splits"]["regions_geojson"]

    logger.info("Loading metadata: %s", metadata_path)
    metadata = load_metadata(metadata_path)
    logger.info("Loading region geometry: %s", regions_geojson)
    ecoregions = load_region_ecoregions(regions_geojson)

    splits, summary = generate_splits(metadata, ecoregions, cfg)

    out_root = args.out_dir or data_root

    if str(out_root).startswith("gs://"):
        base = str(out_root).rstrip("/")
        splits_local = Path("splits.yaml")
        summary_local = Path("splits_summary.json")

        with splits_local.open("w") as f:
            yaml.safe_dump(splits, f, sort_keys=False)
        with summary_local.open("w") as f:
            json.dump(summary, f, indent=2)

        import subprocess
        subprocess.run(
            ["gsutil", "cp", str(splits_local), str(summary_local), f"{base}/"],
            check=True,
        )
        splits_out = f"{base}/splits.yaml"
        summary_out = f"{base}/splits_summary.json"
    else:
        out_path = Path(out_root)
        splits_out = out_path / "splits.yaml"
        summary_out = out_path / "splits_summary.json"

        out_path.mkdir(parents=True, exist_ok=True)
        with splits_out.open("w") as f:
            yaml.safe_dump(splits, f, sort_keys=False)
        with summary_out.open("w") as f:
            json.dump(summary, f, indent=2)

    logger.info("Wrote %s and %s", splits_out, summary_out)
    for split_name, info in summary["splits"].items():
        logger.info("  %-15s  regions=%d  ecos=%d  tiles=%d  pos=%d  neg=%d",
                    split_name, info["n_regions"], info["n_ecoregions"],
                    info["tiles"], info["positive"], info["negative"])
    return 0

if __name__ == "__main__":
    sys.exit(main())
