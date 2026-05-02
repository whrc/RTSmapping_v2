"""Pre-deployment normalization-drift check (inference.md §5.4).

Computes per-channel mean/std on a sample of 2025 tiles and compares against
the deployment package's normalization_stats.json. Flags concerning drift:

    |delta_mean| > 0.5 * sigma_training  OR
    |sigma_sample / sigma_training - 1| > 0.25

Run:
    python scripts/check_inference_normalization.py \\
        --deployment-package gs://.../rts-v2-seed42 \\
        --tile-list sample_tiles.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Thresholds for "concerning drift" (inference.md §5.4).
MEAN_DRIFT_K = 0.5     # delta_mean / sigma_training
STD_DRIFT_FRAC = 0.25  # |sigma_sample / sigma_training - 1|


def _iter_tile_arrays(data_root: Path | str, rgb_dir: str, tile_ids: list[str]):
    """Yield (3, H, W) uint8 arrays for each RGB tile."""
    for tid in tile_ids:
        path = f"{str(data_root).rstrip('/')}/{rgb_dir}/{tid}.tif"
        try:
            with rasterio.open(path) as src:
                yield src.read(out_dtype="uint8")
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", path, e)


def compute_sample_stats(
    tile_iter,
    n_channels: int,
    clip_percentiles: tuple[float, float] | None = None,
) -> dict:
    """Single-pass Welford mean/std over a sample of tiles.

    Returns the same schema as the RGB block of `normalization_stats.json`:
        {"mean": [m_R, m_G, m_B], "std": [s_R, s_G, s_B], "n_pixels": int}
    """
    count = 0
    mean = np.zeros(n_channels, dtype=np.float64)
    M2 = np.zeros(n_channels, dtype=np.float64)

    for arr in tile_iter:
        flat = arr.reshape(n_channels, -1).astype(np.float64)
        if clip_percentiles is not None:
            lo_p, hi_p = clip_percentiles
            lo = np.percentile(flat, lo_p, axis=1, keepdims=True)
            hi = np.percentile(flat, hi_p, axis=1, keepdims=True)
            flat = np.clip(flat, lo, hi)
        n_new = flat.shape[1]
        count_new = count + n_new
        delta = flat - mean[:, None]
        mean = mean + delta.sum(axis=1) / count_new
        delta2 = flat - mean[:, None]
        M2 += (delta * delta2).sum(axis=1)
        count = count_new

    if count == 0:
        raise RuntimeError("No tiles contributed to stats")

    var = M2 / count
    std = np.sqrt(var)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_pixels": int(count),
    }


def compute_drift(
    sample_stats: dict,
    training_stats_block: dict,
    channel_names: list[str],
) -> pd.DataFrame:
    """Compare sample vs training stats per channel.

    `training_stats_block` is one of `stats["rgb"]` or `stats["extra"]` from
    `data/normalization.py:build_stats_dict` — i.e. it carries `channel_names`,
    `mean`, `std` as parallel arrays. `sample_stats` is the dict returned by
    `compute_sample_stats`.
    """
    train_means = list(training_stats_block["mean"])
    train_stds = list(training_stats_block["std"])
    if not (len(channel_names) == len(train_means) == len(train_stds)):
        raise ValueError(
            f"Channel-array length mismatch: names={len(channel_names)}, "
            f"means={len(train_means)}, stds={len(train_stds)}"
        )
    sample_means = sample_stats["mean"]
    sample_stds = sample_stats["std"]

    rows = []
    for i, name in enumerate(channel_names):
        t_mean, t_std = float(train_means[i]), float(train_stds[i])
        s_mean, s_std = float(sample_means[i]), float(sample_stds[i])
        d_mean = abs(s_mean - t_mean)
        d_std_frac = abs(s_std / t_std - 1.0) if t_std > 0 else float("inf")
        concerning = (d_mean > MEAN_DRIFT_K * t_std) or (d_std_frac > STD_DRIFT_FRAC)
        rows.append({
            "channel": name,
            "train_mean": t_mean,
            "train_std": t_std,
            "sample_mean": s_mean,
            "sample_std": s_std,
            "abs_delta_mean": d_mean,
            "abs_delta_mean_over_sigma_train": d_mean / t_std if t_std > 0 else float("inf"),
            "std_ratio_minus_one": s_std / t_std - 1.0 if t_std > 0 else float("inf"),
            "concerning": concerning,
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--deployment-package", type=Path, required=True,
                   help="Path to the deployment-package directory (with "
                        "normalization_stats.json + model_config.yaml)")
    p.add_argument("--tile-list", type=Path, required=True,
                   help="CSV of sample tiles to evaluate; columns: Tile_id, "
                        "optionally data_root override")
    p.add_argument("--data-root", default=None,
                   help="Override data root; otherwise taken from model_config.yaml")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write drift_report.csv (default: alongside tile-list)")
    args = p.parse_args()

    setup_logging(level="INFO")
    pkg = args.deployment_package

    # Load training stats + model config from the package.
    training_stats = json.loads((pkg / "normalization_stats.json").read_text())
    model_cfg = yaml.safe_load((pkg / "model_config.yaml").read_text())

    # Schema (per data/normalization.py:build_stats_dict): {"rgb": {"channel_names": [...], "mean": [...], "std": [...]}, "extra": {...}}
    if "rgb" not in training_stats:
        raise ValueError(
            "normalization_stats.json has no 'rgb' block — package appears "
            "to be pre-schema-update (training.md §4.5). Re-package from a newer run."
        )
    rgb_block = training_stats["rgb"]
    rgb_names = list(rgb_block["channel_names"])
    if rgb_names != ["R", "G", "B"]:
        raise ValueError(
            f"Expected RGB channel order ['R', 'G', 'B'] in training stats; got {rgb_names}"
        )

    # When EXTRA stats are present, log their channel order so any drift-script
    # consumer is aware. (Strict mismatch with a config consumer's expected order
    # is enforced inside RTSDataset.__init__; this script just reports.)
    if "extra" in training_stats:
        extra_names = list(training_stats["extra"].get("channel_names", []))
        logger.info("Training stats include EXTRA channels: %s", extra_names)

    data_root = args.data_root or model_cfg.get("data", {}).get("data_root", ".")
    df = pd.read_csv(args.tile_list, dtype={"Tile_id": str})
    tile_ids = df["Tile_id"].tolist()
    logger.info("Computing sample stats across %d tiles", len(tile_ids))

    # RGB-only drift check. EXTRA channels live in separate GeoTIFFs and are
    # checked separately (TODO: extend when Phase 4 EXTRA stats land).
    sample_stats = compute_sample_stats(
        _iter_tile_arrays(data_root, "PLANET-RGB", tile_ids),
        n_channels=3,
        clip_percentiles=None,
    )

    drift = compute_drift(sample_stats, rgb_block, rgb_names)
    out_path = args.output or (args.tile_list.parent / "drift_report.csv")
    drift.to_csv(out_path, index=False)

    n_bad = int(drift["concerning"].sum())
    logger.info("Drift report: %d/%d channels concerning", n_bad, len(drift))
    logger.info("\n%s", drift.to_string(index=False))
    if n_bad > 0:
        logger.warning(
            "Concerning drift detected — consider histogram matching or "
            "retraining with 2025 data before deployment (inference.md §14)."
        )
        return 2  # distinct exit code for automation
    return 0


if __name__ == "__main__":
    sys.exit(main())
