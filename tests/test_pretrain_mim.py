"""Unit tests for pretraining/mim_dataset.py masking + item loading (CPU-only)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data.normalization import save_stats
from pretraining.mim_dataset import MIMCorpusDataset, expand_mask, random_patch_mask


def test_random_patch_mask_ratio_and_shape():
    gen = np.random.default_rng(0)
    m = random_patch_mask(grid=16, mask_ratio=0.6, generator=gen)
    assert m.shape == (16, 16)
    assert m.sum() == round(0.6 * 256)
    assert m.dtype == bool


def test_random_patch_mask_varies_with_generator():
    a = random_patch_mask(16, 0.6, np.random.default_rng(1))
    b = random_patch_mask(16, 0.6, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_expand_mask_upsamples_to_pixels():
    patch = np.array([[True, False], [False, True]])
    px = expand_mask(patch, patch_px=4)
    assert px.shape == (8, 8)
    assert px[:4, :4].all() and not px[:4, 4:].any()
    assert px[4:, 4:].all() and not px[4:, :4].any()


def _write_corpus(tmp_path, n=3, tile_px=32):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    ids = []
    for i in range(n):
        tid = f"t{i}"
        ids.append(tid)
        rgb = np.full((3, tile_px, tile_px), 50, np.uint8)
        ndvi = np.full((tile_px, tile_px), 0.3, np.float16)
        ndvi[0, 0] = np.nan                     # a NoData NDVI pixel
        np.savez_compressed(tiles_dir / f"{tid}.npz", rgb=rgb, ndvi=ndvi)
    pd.DataFrame({"tile_id": ids}).to_csv(tmp_path / "manifest.csv", index=False)
    stats = {
        "dataset_version": "test", "computed_date": "now", "n_tiles_used": n,
        "rgb": {"channel_names": ["R", "G", "B"], "mean": [50.0, 50.0, 50.0],
                "std": [10.0, 10.0, 10.0]},
        "extra": {"channel_names": ["ndvi"], "mean": [0.3], "std": [0.1]},
    }
    save_stats(stats, tmp_path / "normalization_stats.json")
    return ids


def test_mae_patchify_shape_and_layout():
    # Construction works with any timm ViT (no forward → no size assert); patchify is
    # a pure tensor op independent of the backbone. Full-model forward/backward is
    # covered by the GPU pretrain smoke (pretraining.md §3), not the fast CPU suite.
    import torch

    from pretraining.mim_model import MaskedAutoencoderViT
    m = MaskedAutoencoderViT(backbone="vit_tiny_patch16_384", pretrained=False,
                             in_channels=4, patch_px=16)
    img = torch.zeros(1, 4, 64, 64)          # 4×4 = 16 patches
    img[:, :, :16, :16] = 1.0                # only the top-left patch is nonzero
    patches = m.patchify(img)                # (1, 16, 4*16*16)
    assert patches.shape == (1, 16, 4 * 16 * 16)
    assert patches[0, 0].eq(1.0).all()       # patch (0,0) row all ones
    assert patches[0, 1:].eq(0.0).all()      # every other patch row all zeros


def test_dataset_item_shapes_and_nan_neutralization(tmp_path):
    _write_corpus(tmp_path, n=3, tile_px=32)
    ds = MIMCorpusDataset(tmp_path, patch_px=8, mask_ratio=0.5, tile_px=32)
    assert len(ds) == 3
    item = ds[0]
    assert item["image"].shape == (4, 32, 32)
    assert item["patch_mask"].shape == (4, 4)       # 32/8
    assert item["patch_mask"].sum().item() == 8      # 0.5 * 16
    # NaN NDVI pixel neutralized to 0 (channel mean) by apply_norm.
    assert np.isfinite(item["image"].numpy()).all()
    assert item["image"][3, 0, 0].item() == 0.0
