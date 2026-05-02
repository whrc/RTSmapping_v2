"""Regression tests for data/transforms.py.

Focus: EXTRA channels must NOT receive color/radiometric augmentation
(training.md §9.2). Geometric augmentation (flips, rotations, scale,
elastic) DOES apply to EXTRA.
"""

from __future__ import annotations

import numpy as np

from data.transforms import build_train_transforms


def _aug_cfg(*, color_p: float, geo_p: float) -> dict:
    """Build an augmentation config with all color ops at color_p and all geometric
    ops at geo_p. Multi-scale stays off (would change tile shape)."""
    return {
        "geometric": {
            "rot90_p": geo_p, "hflip_p": geo_p, "vflip_p": geo_p,
            "shift_scale_rotate": {"shift": 0.1, "scale": 0.1, "rotate": 30, "p": geo_p},
            "elastic": {"alpha": 120, "sigma": 6, "p": 0.0},   # leave elastic off; deterministic shapes only
            "shear": {"shear_degrees": 10, "p": 0.0},
        },
        "color": {
            "brightness": 0.5, "contrast": 0.5, "saturation": 0.5,
            "brightness_contrast_p": color_p,
            "gaussian_noise": {"var_limit": [10, 50], "p": color_p},
            "clahe": {"clip_limit": 4.0, "tile_grid": [8, 8], "p": color_p},
        },
        "multi_scale": {"scale_range": [1.0, 1.0], "p": 0.0},
    }


def _make_inputs(seed: int):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    extra = rng.standard_normal((64, 64, 2)).astype(np.float32)
    mask = rng.integers(0, 2, size=(64, 64), dtype=np.uint8)
    return rgb, extra, mask


def test_color_aug_does_not_touch_extra():
    """With color_p=1.0 and geo_p=0.0, EXTRA must come back bit-identical."""
    rgb, extra, mask = _make_inputs(seed=42)

    transform = build_train_transforms(tile_size=64, aug_cfg=_aug_cfg(color_p=1.0, geo_p=0.0))
    out = transform(image=rgb, extra=extra, mask=mask)

    assert out["extra"].shape == extra.shape
    assert out["extra"].dtype == extra.dtype
    np.testing.assert_array_equal(
        out["extra"], extra,
        err_msg="EXTRA pixels were modified by color-only augmentation",
    )
    # Mask is also untouched by color ops.
    np.testing.assert_array_equal(out["mask"], mask)
    # RGB SHOULD have changed (color_p=1.0 means at least one op fires).
    assert not np.array_equal(out["image"], rgb), \
        "RGB unchanged despite color_p=1.0 — color stage may not be running"


def test_geometric_aug_applies_to_extra_and_mask():
    """With hflip_p=1.0 (only geometric op firing), EXTRA and mask must transform together."""
    # Disable color so we can isolate the geometric effect.
    rgb, extra, mask = _make_inputs(seed=7)
    cfg = _aug_cfg(color_p=0.0, geo_p=0.0)
    cfg["geometric"]["hflip_p"] = 1.0   # only hflip fires; deterministic
    cfg["geometric"]["rot90_p"] = 0.0
    cfg["geometric"]["vflip_p"] = 0.0
    cfg["geometric"]["shift_scale_rotate"]["p"] = 0.0

    transform = build_train_transforms(tile_size=64, aug_cfg=cfg)
    out = transform(image=rgb, extra=extra, mask=mask)

    np.testing.assert_array_equal(out["image"], np.ascontiguousarray(rgb[:, ::-1, :]))
    np.testing.assert_array_equal(out["extra"], np.ascontiguousarray(extra[:, ::-1, :]))
    np.testing.assert_array_equal(out["mask"], np.ascontiguousarray(mask[:, ::-1]))


def test_extra_none_path_still_works():
    """Existing RGB-only call path (no extra kwarg) must still work."""
    rgb, _, mask = _make_inputs(seed=11)
    transform = build_train_transforms(tile_size=64, aug_cfg=_aug_cfg(color_p=0.0, geo_p=0.0))
    out = transform(image=rgb, mask=mask)
    np.testing.assert_array_equal(out["image"], rgb)
    np.testing.assert_array_equal(out["mask"], mask)
    assert "extra" not in out
