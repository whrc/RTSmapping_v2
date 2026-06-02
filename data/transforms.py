"""Albumentations augmentation pipelines.

Per training.md §9.2. Covers:
  - Geometric transforms (applied to RGB, EXTRA, mask together)
  - Color transforms (RGB only — EXTRA and mask pass through)
  - Multi-scale (RandomScale + PadIfNeeded + CenterCrop)

Normalization and tensor conversion happen in data/dataset.py AFTER this pipeline,
so mean/std arrays (which depend on the channel stack) stay out of here.

Usage:
    train = build_train_transforms(tile_size=512, aug_cfg=cfg["augmentation"])
    eval_  = build_eval_transforms()
    out = train(image=rgb_hwc, extra=extra_hwc, mask=label)
"""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np
from scipy import ndimage


class TrainTransform:
    """Two-stage augmentation: color-only on RGB, then geometric on RGB+EXTRA+mask.

    Designed to honor training.md §9.2: color/radiometric ops apply only to
    RGB; EXTRA and mask must not see brightness/contrast/saturation/noise/CLAHE.
    Geometric ops apply to all three.
    """

    def __init__(self, color_stage: A.Compose, geometric_stage: A.Compose):
        self._color = color_stage
        self._geo = geometric_stage

    def __call__(self, *, image, extra=None, mask):
        # Stage 1: color ops on RGB only. mask flows through (Compose passes
        # the mask key through unchanged when no transform touches it).
        color_out = self._color(image=image, mask=mask)
        rgb = color_out["image"]
        mask = color_out["mask"]
        # Stage 2: geometric ops on RGB+EXTRA+mask together.
        if extra is not None:
            geo_out = self._geo(image=rgb, extra=extra, mask=mask)
            return {"image": geo_out["image"], "extra": geo_out["extra"], "mask": geo_out["mask"]}
        geo_out = self._geo(image=rgb, mask=mask)
        return {"image": geo_out["image"], "mask": geo_out["mask"]}


def build_train_transforms(tile_size: int, aug_cfg: dict[str, Any]) -> TrainTransform:
    """Training-time augmentation. Returns a TrainTransform callable.

    Color stage runs on RGB only (training.md §9.2). Geometric + multi-scale
    stage runs on RGB+EXTRA+mask together via additional_targets.
    """
    geo = aug_cfg["geometric"]
    col = aug_cfg["color"]
    ms = aug_cfg["multi_scale"]

    color_stage = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=col["brightness"],
            contrast_limit=col["contrast"],
            p=col["brightness_contrast_p"],
        ),
        A.HueSaturationValue(
            sat_shift_limit=int(col["saturation"] * 100),
            p=col["brightness_contrast_p"],
        ),
        A.GaussNoise(
            var_limit=tuple(col["gaussian_noise"]["var_limit"]),
            p=col["gaussian_noise"]["p"],
        ),
        A.CLAHE(
            clip_limit=col["clahe"]["clip_limit"],
            tile_grid_size=tuple(col["clahe"]["tile_grid"]),
            p=col["clahe"]["p"],
        ),
    ])

    geometric_stage = A.Compose(
        [
            A.RandomRotate90(p=geo["rot90_p"]),
            A.HorizontalFlip(p=geo["hflip_p"]),
            A.VerticalFlip(p=geo["vflip_p"]),
            A.ShiftScaleRotate(
                shift_limit=geo["shift_scale_rotate"]["shift"],
                scale_limit=geo["shift_scale_rotate"]["scale"],
                rotate_limit=geo["shift_scale_rotate"]["rotate"],
                p=geo["shift_scale_rotate"]["p"],
                border_mode=0,
            ),
            A.ElasticTransform(
                alpha=geo["elastic"]["alpha"],
                sigma=geo["elastic"]["sigma"],
                p=geo["elastic"]["p"],
            ),
            A.Affine(shear=(-geo["shear"]["shear_degrees"], geo["shear"]["shear_degrees"]),
                     p=geo["shear"]["p"]),
            A.RandomScale(
                scale_limit=(ms["scale_range"][0] - 1.0, ms["scale_range"][1] - 1.0),
                p=ms["p"],
            ),
            A.PadIfNeeded(min_height=tile_size, min_width=tile_size, border_mode=0),
            A.CenterCrop(height=tile_size, width=tile_size),
        ],
        additional_targets={"extra": "image"},
    )

    return TrainTransform(color_stage, geometric_stage)


def build_eval_transforms() -> A.Compose:
    """No-op pipeline. Normalization done in the Dataset."""
    return A.Compose([], additional_targets={"extra": "image"})


def dilate_label_boundary(label: np.ndarray, width: int, ignore_index: int = 255) -> np.ndarray:
    """Set pixels within `width` of a class boundary to ignore_index.

    Used when loss.boundary_handling == 'ignore' (training.md §5.5, approach 1).
    Operates in-place on a copy.
    """
    if width <= 0:
        return label
    out = label.copy()
    # A boundary pixel is any positive pixel whose 4-neighbour is not positive (or vice-versa).
    pos = label == 1
    struct = ndimage.generate_binary_structure(2, 1)
    boundary = ndimage.binary_dilation(pos, structure=struct, iterations=width) ^ \
               ndimage.binary_erosion(pos, structure=struct, iterations=width, border_value=0)
    out[boundary] = ignore_index
    return out
