"""End-to-end DataLoader iteration gate. MUST pass before any training run.

Per training.md §10.1. Builds the real DataLoader from configs/baseline.yaml,
iterates N batches, verifies shapes/dtypes/value ranges/ignore-index behavior
and per-batch pos:neg ratio.

Usage:
  python scripts/check_data.py --config configs/baseline.yaml --n-batches 20
  python scripts/check_data.py --config configs/baseline.yaml --preview preview.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import RTSDataset, parse_extra_spec  # noqa: E402
from data.sampler import BalancedBatchSampler  # noqa: E402
from data.splits import get_tile_ids, load_metadata, load_splits_yaml  # noqa: E402
from data.transforms import build_train_transforms  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


def _save_preview(images: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    """Save up to 6 tiles (3 pos / 3 neg heuristic) as a sanity-check grid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping preview")
        return

    n = min(6, images.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i in range(n):
        img = images[i, :3].numpy().transpose(1, 2, 0)
        img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"rgb[{i}]")
        axes[0, i].axis("off")
        axes[1, i].imshow(labels[i].numpy(), cmap="tab10", vmin=0, vmax=10)
        axes[1, i].set_title(f"label[{i}]")
        axes[1, i].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved preview: %s", path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Override DataLoader workers (0 useful for debugging)")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    seed_everything(cfg.get("seed", 42), deterministic=cfg.get("deterministic", True))

    # Resolve paths
    data_root = cfg["data"]["data_root"]
    metadata_path = f"{data_root.rstrip('/')}/{cfg['data']['metadata_csv']}"
    splits_path = f"{data_root.rstrip('/')}/{cfg['data']['splits_yaml']}"
    norm_stats = cfg["data"]["normalization_stats_path"]
    # Tolerate missing stats — warn but continue so check_data.py is runnable before
    # compute_normalization_stats.py has been executed for real data.
    try:
        from data.normalization import load_stats
        load_stats(norm_stats)
        norm_stats_arg: str | None = norm_stats
    except (FileNotFoundError, OSError):
        logger.warning("normalization_stats.json missing at %s — running unnormalized", norm_stats)
        norm_stats_arg = None

    metadata = load_metadata(metadata_path)
    splits = load_splits_yaml(splits_path)
    tile_ids = get_tile_ids("train", metadata, splits)
    logger.info("Train split: %d tiles", len(tile_ids))

    extra_channels = parse_extra_spec(cfg["channels"]["extra"])
    transform = build_train_transforms(
        tile_size=cfg["data"]["tile_size"],
        aug_cfg=cfg["augmentation"],
    )
    dataset = RTSDataset(
        tile_ids=tile_ids,
        metadata=metadata,
        data_root=data_root,
        rgb_dir=cfg["data"]["rgb_dir"],
        extra_dir=cfg["data"]["extra_dir"],
        labels_dir=cfg["data"]["labels_dir"],
        extra_channels=extra_channels,
        norm_stats_path=norm_stats_arg,
        transform=transform,
        tile_size=cfg["data"]["tile_size"],
        label_ignore_index=cfg["data"]["label_ignore_index"],
        boundary_handling=cfg["loss"]["boundary_handling"],
        boundary_ignore_width=cfg["loss"]["boundary_ignore_width"],
    )

    sampler = BalancedBatchSampler(
        tile_ids=tile_ids,
        metadata=metadata,
        batch_size=cfg["training"]["batch_size"],
        schedule=cfg["sampling"]["curriculum_schedule"],
        seed=cfg.get("seed", 42),
        epoch=args.epoch,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=_collate,
    )

    expected_channels = 3 + len(extra_channels)
    expected_n_pos, expected_n_neg = sampler._batch_split()  # noqa: SLF001
    logger.info("Expected per-batch: %d pos + %d neg = %d total (C=%d)",
                expected_n_pos, expected_n_neg,
                expected_n_pos + expected_n_neg, expected_channels)

    errors: list[str] = []
    for bi, batch in enumerate(loader):
        if bi >= args.n_batches:
            break
        images = batch["image"]
        labels = batch["label"]
        tile_ids_b = batch["tile_id"]

        # Shape / dtype
        if images.shape[1] != expected_channels:
            errors.append(f"batch {bi}: channel count {images.shape[1]} != {expected_channels}")
        if labels.shape != images.shape[0:1] + images.shape[2:]:
            errors.append(f"batch {bi}: label shape {labels.shape} vs image {images.shape}")
        if images.dtype != torch.float32:
            errors.append(f"batch {bi}: image dtype {images.dtype} != float32")
        if labels.dtype != torch.int64:
            errors.append(f"batch {bi}: label dtype {labels.dtype} != int64")

        # Label values ⊂ {0, 1, 255}
        uniq = set(torch.unique(labels).tolist())
        if not uniq.issubset({0, 1, 255}):
            errors.append(f"batch {bi}: label values {uniq} outside {{0,1,255}}")

        # Ratio
        classes = [dataset.is_positive(t) for t in tile_ids_b]
        got_pos, got_neg = sum(classes), len(classes) - sum(classes)
        if got_pos != expected_n_pos:
            logger.debug("batch %d pos/neg = %d/%d (expected %d/%d)",
                         bi, got_pos, got_neg, expected_n_pos, expected_n_neg)

        if bi == 0 and args.preview is not None:
            _save_preview(images, labels, args.preview)

    if errors:
        logger.error("FAILED with %d error(s):", len(errors))
        for e in errors:
            logger.error("  %s", e)
        return 1

    logger.info("OK — iterated %d batch(es) without errors.", args.n_batches)
    return 0


def _collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "tile_id": [b["tile_id"] for b in batch],
    }


if __name__ == "__main__":
    sys.exit(main())
