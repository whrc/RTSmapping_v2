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


def _scale_down_cfg() -> dict:
    """All ops off except a deterministic 0.5x RandomScale (forces PadIfNeeded to pad)."""
    cfg = _aug_cfg(color_p=0.0, geo_p=0.0)
    cfg["multi_scale"] = {"scale_range": [0.5, 0.5], "p": 1.0}
    return cfg


def test_pad_mask_ignore_default_is_background():
    """Default (flag absent): the RandomScale pad border in the mask is background (0).

    This documents the baked-in baseline behaviour the A/B compares against.
    """
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.uint8)  # all-RTS so any 0 in output is from padding
    transform = build_train_transforms(tile_size=64, aug_cfg=_scale_down_cfg())
    out = transform(image=rgb, mask=mask)
    assert (out["mask"] == 0).any(), "expected a background-labeled pad border by default"
    assert (out["mask"] == 255).sum() == 0, "no ignore pixels should appear by default"


def test_pad_mask_ignore_true_labels_border_ignore():
    """With multi_scale.pad_mask_ignore=true, the pad border becomes ignore (255)."""
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.uint8)
    cfg = _scale_down_cfg()
    cfg["multi_scale"]["pad_mask_ignore"] = True
    transform = build_train_transforms(tile_size=64, aug_cfg=cfg, ignore_index=255)
    out = transform(image=rgb, mask=mask)
    assert (out["mask"] == 255).any(), "expected an ignore-labeled pad border with the flag on"
    assert (out["mask"] == 0).sum() == 0, "pad border should be ignore(255), not background(0)"


def test_extra_none_path_still_works():
    """Existing RGB-only call path (no extra kwarg) must still work."""
    rgb, _, mask = _make_inputs(seed=11)
    transform = build_train_transforms(tile_size=64, aug_cfg=_aug_cfg(color_p=0.0, geo_p=0.0))
    out = transform(image=rgb, mask=mask)
    np.testing.assert_array_equal(out["image"], rgb)
    np.testing.assert_array_equal(out["mask"], mask)
    assert "extra" not in out


# --- Auto-augment policies (RandAugment / TrivialAugment), family F ---

def _auto_cfg(mode: str, **kw) -> dict:
    """Base aug cfg (geometry+scale off) with an auto_policy block."""
    cfg = _aug_cfg(color_p=0.0, geo_p=0.0)  # geometry/scale off → mask & shape stay fixed
    cfg["auto_policy"] = {"mode": mode, **kw}
    return cfg


def test_auto_policy_default_none_is_handtuned():
    """No auto_policy key → hand-tuned color stage (locked baseline), runs unchanged."""
    rgb, extra, mask = _make_inputs(seed=1)
    cfg = _aug_cfg(color_p=1.0, geo_p=0.0)
    assert "auto_policy" not in cfg
    out = build_train_transforms(tile_size=64, aug_cfg=cfg)(image=rgb, extra=extra, mask=mask)
    assert out["image"].shape == rgb.shape
    np.testing.assert_array_equal(out["extra"], extra)  # color stage never touches EXTRA


def test_trivialaugment_runs_preserves_shape_and_mask():
    rgb, extra, mask = _make_inputs(seed=2)
    out = build_train_transforms(tile_size=64, aug_cfg=_auto_cfg("trivialaugment", magnitude=1.0))(
        image=rgb, extra=extra, mask=mask)
    assert out["image"].shape == rgb.shape and out["image"].dtype == rgb.dtype
    np.testing.assert_array_equal(out["mask"], mask)      # photometric ops never touch the mask
    np.testing.assert_array_equal(out["extra"], extra)    # nor EXTRA


def test_randaugment_runs_with_num_ops():
    rgb, extra, mask = _make_inputs(seed=3)
    out = build_train_transforms(tile_size=64, aug_cfg=_auto_cfg("randaugment", num_ops=2, magnitude=0.5))(
        image=rgb, extra=extra, mask=mask)
    assert out["image"].shape == rgb.shape
    np.testing.assert_array_equal(out["mask"], mask)


def test_auto_policy_pool_excludes_shadow_scramblers():
    """The op pool must omit shadow-cue scramblers (solarize/invert/posterize/equalize/
    channelshuffle/grayscale) — RTS keys on headwall shadows + tonal contrast."""
    from data.transforms import _AUTOAUG_EXCLUDED, _auto_policy_pool
    names = {type(op).__name__.lower() for op in _auto_policy_pool(0.5)}
    for banned in _AUTOAUG_EXCLUDED:
        assert not any(banned in n for n in names), f"shadow scrambler {banned!r} leaked into pool"


def test_auto_policy_invalid_mode_raises():
    import pytest
    rgb, _, mask = _make_inputs(seed=4)
    with pytest.raises(ValueError):
        build_train_transforms(tile_size=64, aug_cfg=_auto_cfg("autoaugment"))(image=rgb, mask=mask)
