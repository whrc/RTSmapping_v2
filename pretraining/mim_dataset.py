"""Dataset + masking for DINOv3-L MAE pretraining (spec: pretraining/pretraining.md §3).

Reads the materialized .npz corpus, normalizes with the shared ``apply_norm`` (CLAUDE
Rule 3, same code path as training/inference), and produces a random patch mask per
tile. Patches are the ViT patch size (16 for DINOv3-L/16 → 32×32=1024 tokens at 512px);
the mask marks which tokens are hidden. Kept deliberately thin: masking is a pure
function so it is unit-testable on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.normalization import apply_norm, build_norm_arrays, load_stats


def random_patch_mask(
    grid: int, mask_ratio: float, generator: np.random.Generator,
) -> np.ndarray:
    """Boolean (grid, grid) mask, True where the patch is masked (hidden).

    ``grid`` = tile_px // patch_px. Exactly ``round(mask_ratio * grid**2)`` patches
    are masked, chosen uniformly without replacement — the standard MAE/FCMAE scheme.
    """
    n = grid * grid
    k = int(round(mask_ratio * n))
    flat = np.zeros(n, dtype=bool)
    flat[generator.choice(n, size=k, replace=False)] = True
    return flat.reshape(grid, grid)


def expand_mask(patch_mask: np.ndarray, patch_px: int) -> np.ndarray:
    """Upsample a (grid, grid) patch mask to a (H, W) pixel mask."""
    return np.repeat(np.repeat(patch_mask, patch_px, axis=0), patch_px, axis=1)


class MIMCorpusDataset(Dataset):
    """4-ch normalized tiles + per-tile patch masks for masked pretraining."""

    def __init__(
        self,
        corpus_dir: str | Path,
        patch_px: int = 16,
        mask_ratio: float = 0.75,
        tile_px: int = 512,
        manifest: str | Path | None = None,
        stats: str | Path | None = None,
        seed: int = 42,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.tiles_dir = self.corpus_dir / "tiles"
        manifest = manifest or self.corpus_dir / "manifest.csv"
        stats = stats or self.corpus_dir / "normalization_stats.json"
        self.ids = pd.read_csv(manifest)["tile_id"].astype(str).tolist()
        self.norm_params = build_norm_arrays(load_stats(stats), with_extra=True)
        self.patch_px = patch_px
        self.mask_ratio = mask_ratio
        self.grid = tile_px // patch_px
        self.tile_px = tile_px
        self._seed = seed

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        tile_id = self.ids[i]
        with np.load(self.tiles_dir / f"{tile_id}.npz") as z:
            rgb = z["rgb"].astype(np.float32)          # (3, H, W), raw [0,255]
            ndvi = z["ndvi"].astype(np.float32)        # (H, W), NaN where no S2
        stack = np.concatenate([rgb, ndvi[None]], axis=0)   # (4, H, W)
        image = apply_norm(stack, self.norm_params)         # NaN NDVI → 0 (channel mean)
        # Per-item RNG derived from the epoch-agnostic seed + index so masks vary
        # across items but the dataset stays reproducible for a fixed seed.
        gen = np.random.default_rng(self._seed + i)
        patch_mask = random_patch_mask(self.grid, self.mask_ratio, gen)
        return {
            "image": torch.from_numpy(image),
            "patch_mask": torch.from_numpy(patch_mask),
            "tile_id": tile_id,
        }
