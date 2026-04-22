"""Compute per-dataset normalization stats over the training split.

Single-pass Welford over all train-split tiles. Writes normalization_stats.json.
See data/data.md §5 and data/normalization.py.

Usage:
  python scripts/compute_normalization_stats.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.normalization import WelfordStats, build_stats_dict, save_stats  # noqa: E402
from data.splits import get_tile_ids, load_metadata, load_splits_yaml  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


def _tile_path(data_root: str, subdir: str, tile_id: str) -> str:
    return f"{data_root.rstrip('/')}/{subdir}/{tile_id}.tif"


def _read_rgb(path: str) -> np.ndarray:
    """Read RGB tile → (3, H, W) float32 in [0, 255] (raw uint8 promoted)."""
    with rasterio.open(path) as src:
        arr = src.read(out_dtype="uint8")  # (3, H, W)
    return arr.astype(np.float32)


def _read_extra(path: str, bands: list[int]) -> np.ndarray:
    """Read selected EXTRA bands → (len(bands), H, W) float32."""
    with rasterio.open(path) as src:
        # rasterio bands are 1-indexed; config bands are 0-indexed.
        arr = src.read([b + 1 for b in bands], out_dtype="float32")
    return arr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path (default: data.normalization_stats_path from config)")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Cap on number of train tiles to use (for smoke testing)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42), deterministic=False)

    data_root = cfg["data"]["data_root"]
    metadata_path = f"{data_root.rstrip('/')}/{cfg['data']['metadata_csv']}"
    splits_path = f"{data_root.rstrip('/')}/{cfg['data']['splits_yaml']}"

    logger.info("Loading metadata: %s", metadata_path)
    metadata = load_metadata(metadata_path)
    logger.info("Loading splits: %s", splits_path)
    splits = load_splits_yaml(splits_path)

    train_ids = get_tile_ids("train", metadata, splits)
    if args.max_tiles is not None:
        train_ids = train_ids[: args.max_tiles]
    logger.info("Computing stats over %d train tiles", len(train_ids))

    rgb_stats = WelfordStats(channel_names=["R", "G", "B"])

    extra_cfg = cfg["channels"]["extra"] or []
    extra_names = [c["name"] for c in extra_cfg]
    extra_bands = [int(c["band"]) for c in extra_cfg]
    extra_stats = WelfordStats(channel_names=extra_names) if extra_names else None

    rgb_dir = cfg["data"]["rgb_dir"]
    extra_dir = cfg["data"]["extra_dir"]

    for tid in tqdm(train_ids, desc="Welford pass"):
        rgb_arr = _read_rgb(_tile_path(data_root, rgb_dir, tid))
        rgb_stats.update(rgb_arr)
        if extra_stats is not None:
            extra_arr = _read_extra(_tile_path(data_root, extra_dir, tid), extra_bands)
            extra_stats.update(extra_arr)

    stats = build_stats_dict(
        rgb=rgb_stats,
        extra=extra_stats,
        dataset_version=cfg["data"]["version"],
        n_tiles_used=len(train_ids),
    )
    out_path = args.out or Path(cfg["data"]["normalization_stats_path"])

    if str(out_path).startswith("gs://"):
        logger.warning("Writing to local 'normalization_stats.json'; upload to %s manually.", out_path)
        out_path = Path("normalization_stats.json")

    save_stats(stats, out_path)
    logger.info("Saved %s", out_path)
    logger.info("RGB mean=%s", stats["rgb"]["mean"])
    logger.info("RGB std =%s", stats["rgb"]["std"])
    if extra_stats is not None:
        logger.info("EXTRA mean=%s", stats["extra"]["mean"])
        logger.info("EXTRA std =%s", stats["extra"]["std"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
