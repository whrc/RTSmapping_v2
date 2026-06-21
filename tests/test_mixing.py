"""Unit tests for data/mixing.py (copy-paste / mosaic / cutmix / mixup).

All ops are pure array transforms; tests use synthetic tiles + a fake sample_fn (no I/O).
Invariants checked: shape/dtype preserved, labels stay in {0,1,255}, off-by-default
identity, and copy-paste/mosaic raise positive-pixel density (the rare-object lever).
"""

from __future__ import annotations

import numpy as np

from data.mixing import MixingAugmenter, copy_paste, cutmix, mixup, mosaic

S = 64
IGN = 255


def _tile(seed, *, n_extra=2, pos_block=None):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (S, S, 3), dtype=np.uint8)
    extra = rng.standard_normal((S, S, n_extra)).astype(np.float32) if n_extra else None
    label = np.zeros((S, S), dtype=np.uint8)
    if pos_block is not None:
        y, x, h, w = pos_block
        label[y:y + h, x:x + w] = 1
    return rgb, extra, label


def _valid_labels(lab):
    return set(np.unique(lab)).issubset({0, 1, IGN})


def test_copy_paste_adds_positives_and_preserves_dtype():
    rgb, extra, label = _tile(1)                                   # all-negative target
    s_rgb, s_extra, s_lab = _tile(2, pos_block=(10, 10, 16, 16))   # source with one RTS instance
    rng = np.random.default_rng(0)
    out_rgb, out_extra, out_lab = copy_paste(rgb, extra, label, s_rgb, s_extra, s_lab, rng)
    assert out_rgb.shape == rgb.shape and out_rgb.dtype == np.uint8
    assert out_extra.shape == extra.shape and out_extra.dtype == np.float32
    assert _valid_labels(out_lab)
    assert (out_lab == 1).sum() > 0          # an instance was pasted
    assert not np.array_equal(out_rgb, rgb)  # pixels changed


def test_copy_paste_no_source_instances_is_identity():
    rgb, extra, label = _tile(1, pos_block=(0, 0, 8, 8))
    s_rgb, s_extra, s_lab = _tile(2)         # all-negative source → nothing to paste
    out = copy_paste(rgb, extra, label, s_rgb, s_extra, s_lab, np.random.default_rng(0))
    np.testing.assert_array_equal(out[0], rgb)
    np.testing.assert_array_equal(out[2], label)


def test_copy_paste_preserves_ignore_pixels():
    rgb, extra, label = _tile(1)
    label[:] = IGN                            # whole target is ignore
    s_rgb, s_extra, s_lab = _tile(2, pos_block=(20, 20, 16, 16))
    _, _, out_lab = copy_paste(rgb, extra, label, s_rgb, s_extra, s_lab, np.random.default_rng(0))
    assert (out_lab == 1).sum() == 0          # ignore not overwritten with positive
    assert (out_lab == IGN).all()


def test_copy_paste_rgb_only_path():
    rgb, _, label = _tile(1, n_extra=0)
    s_rgb, _, s_lab = _tile(2, n_extra=0, pos_block=(10, 10, 12, 12))
    out_rgb, out_extra, out_lab = copy_paste(rgb, None, label, s_rgb, None, s_lab, np.random.default_rng(0))
    assert out_extra is None and out_rgb.shape == (S, S, 3) and _valid_labels(out_lab)


def test_mosaic_output_shape_and_density():
    tiles = [_tile(i, pos_block=(8, 8, 24, 24)) for i in range(4)]  # all positive
    out_rgb, out_extra, out_lab = mosaic(tiles, np.random.default_rng(0), tile_size=S)
    assert out_rgb.shape == (S, S, 3) and out_lab.shape == (S, S)
    assert out_extra.shape == (S, S, 2)
    assert _valid_labels(out_lab) and (out_lab == 1).sum() > 0


def test_cutmix_swaps_patch():
    rgb, extra, label = _tile(1)
    s_rgb, s_extra, s_lab = _tile(2, pos_block=(0, 0, S, S))   # all-positive source
    out_rgb, _, out_lab = cutmix(rgb, extra, label, s_rgb, s_extra, s_lab, np.random.default_rng(3))
    assert out_rgb.shape == rgb.shape and _valid_labels(out_lab)
    assert (out_lab == 1).sum() > 0          # source positives entered the target


def test_mixup_blends_and_unions_labels():
    rgb, extra, label = _tile(1, pos_block=(0, 0, 16, 16))
    s_rgb, s_extra, s_lab = _tile(2, pos_block=(40, 40, 16, 16))
    out_rgb, out_extra, out_lab = mixup(rgb, extra, label, s_rgb, s_extra, s_lab,
                                        np.random.default_rng(0), alpha=0.5)
    assert out_rgb.shape == rgb.shape and out_rgb.dtype == np.uint8
    assert _valid_labels(out_lab)
    # union of the two positive blocks present
    assert out_lab[0:16, 0:16].max() == 1 and out_lab[40:56, 40:56].max() == 1


def _sample_fn_factory():
    def fn(positive_only):
        return _tile(99, pos_block=(10, 10, 16, 16) if positive_only else None)
    return fn


def test_augmenter_off_by_default_is_identity():
    aug = MixingAugmenter({"mixing": {}}, _sample_fn_factory(), tile_size=S)
    assert aug.enabled is False
    rgb, extra, label = _tile(1, pos_block=(5, 5, 8, 8))
    out_rgb, out_extra, out_lab = aug(rgb, extra, label, np.random.default_rng(0))
    np.testing.assert_array_equal(out_rgb, rgb)
    np.testing.assert_array_equal(out_lab, label)
    np.testing.assert_array_equal(out_extra, extra)


def test_augmenter_copy_paste_p1_fires():
    cfg = {"mixing": {"copy_paste": {"p": 1.0, "max_instances": 2}}}
    aug = MixingAugmenter(cfg, _sample_fn_factory(), tile_size=S)
    assert aug.enabled is True
    rgb, extra, label = _tile(1)             # negative target
    _, _, out_lab = aug(rgb, extra, label, np.random.default_rng(0))
    assert (out_lab == 1).sum() > 0          # copy-paste fired
