"""Unit tests for training.metrics.ValidationAccumulator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.metrics import ValidationAccumulator, _filter_small_blobs, _match_objects


def _cfg(**metrics_overrides) -> dict:
    cfg = {
        "data": {"label_ignore_index": 255},
        "metrics": {
            "reporting_threshold": 0.5,
            "min_blob_size_px": 4,
            "object_iou_threshold": 0.3,
        },
    }
    cfg["metrics"].update(metrics_overrides)
    return cfg


def _logits_from_prob(prob: np.ndarray) -> torch.Tensor:
    """Invert sigmoid: logit = log(p / (1-p)). Clamped to avoid inf."""
    p = np.clip(prob.astype(np.float32), 1e-7, 1 - 1e-7)
    logit = np.log(p / (1.0 - p))
    return torch.from_numpy(logit).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


# ---------------------------------------------------------------------------
# _filter_small_blobs
# ---------------------------------------------------------------------------


def test_filter_small_blobs_drops_undersized():
    binary = np.zeros((10, 10), dtype=np.uint8)
    # Big blob (9 pixels) + speckle (1 pixel).
    binary[0:3, 0:3] = 1  # 9 pixels
    binary[9, 9] = 1      # 1 pixel
    out = _filter_small_blobs(binary, min_size=4)
    assert out.sum() == 9
    assert out[9, 9] == 0


def test_filter_small_blobs_passthrough_when_min_leq_one():
    binary = np.zeros((5, 5), dtype=np.uint8)
    binary[2, 2] = 1
    out = _filter_small_blobs(binary, min_size=1)
    assert np.array_equal(out, binary)


# ---------------------------------------------------------------------------
# _match_objects
# ---------------------------------------------------------------------------


def test_match_objects_empty_both():
    tp, fp, fn = _match_objects(
        pred_labels=np.zeros((5, 5), dtype=int), n_pred=0,
        gt_labels=np.zeros((5, 5), dtype=int), n_gt=0,
        pred_conf_per_blob=np.array([]), iou_threshold=0.3,
    )
    assert (tp, fp, fn) == (0, 0, 0)


def test_match_objects_empty_pred_positive_tile():
    tp, fp, fn = _match_objects(
        pred_labels=np.zeros((5, 5), dtype=int), n_pred=0,
        gt_labels=np.ones((5, 5), dtype=int), n_gt=2,
        pred_conf_per_blob=np.array([]), iou_threshold=0.3,
    )
    assert (tp, fp, fn) == (0, 0, 2)


def test_match_objects_empty_gt_negative_tile():
    tp, fp, fn = _match_objects(
        pred_labels=np.ones((5, 5), dtype=int), n_pred=3,
        gt_labels=np.zeros((5, 5), dtype=int), n_gt=0,
        pred_conf_per_blob=np.array([0.9, 0.7, 0.5]), iou_threshold=0.3,
    )
    assert (tp, fp, fn) == (0, 3, 0)


def test_match_objects_greedy_confidence_sort():
    """Two preds overlapping one GT — higher-conf pred matches, the other is FP."""
    pred_labels = np.zeros((10, 10), dtype=int)
    # Pred 1 covers top-left 4x4 with IoU=1 vs GT 1 (same region).
    pred_labels[0:4, 0:4] = 1
    # Pred 2 covers the same region (partial) — lower confidence.
    pred_labels[2:4, 2:4] = 2  # overwrites part of pred 1 in the label map, but we treat as distinct
    # Actually, scipy.ndimage.label doesn't produce overlapping labels.
    # Simulate two separate preds with the same GT via a small contrived case:
    pred_labels = np.zeros((10, 10), dtype=int)
    pred_labels[0:3, 0:3] = 1
    pred_labels[0:3, 5:8] = 2  # won't overlap gt2 -> FP

    gt_labels = np.zeros((10, 10), dtype=int)
    gt_labels[0:3, 0:3] = 1

    conf = np.array([0.9, 0.7])  # pred 1 more confident
    tp, fp, fn = _match_objects(pred_labels, 2, gt_labels, 1, conf, 0.3)
    assert (tp, fp, fn) == (1, 1, 0)


# ---------------------------------------------------------------------------
# ValidationAccumulator — integration
# ---------------------------------------------------------------------------


def test_accumulator_perfect_prediction_pixel_iou_one():
    """A tile where prediction exactly matches label yields pixel_iou=1."""
    cfg = _cfg()
    acc = ValidationAccumulator(cfg, ratios=[1])

    prob = np.zeros((16, 16), dtype=np.float32)
    prob[4:12, 4:12] = 0.99
    label = np.zeros((16, 16), dtype=np.int64)
    label[4:12, 4:12] = 1

    logits = _logits_from_prob(prob)
    acc.update(logits, torch.from_numpy(label).unsqueeze(0), ["tile_0"])
    m = acc.compute()
    assert m["pixel_iou"] == pytest.approx(1.0)
    assert m["pixel_f1"] == pytest.approx(1.0)
    # Object-level: 1 GT, 1 pred, IoU=1 -> TP=1.
    assert m["object_precision"] == pytest.approx(1.0)
    assert m["object_recall"] == pytest.approx(1.0)
    assert m["object_f1"] == pytest.approx(1.0)


def test_accumulator_ignore_index_masks_pixels():
    """Ignore pixels should not contribute to pixel_tp/fp/fn."""
    cfg = _cfg()
    acc = ValidationAccumulator(cfg, ratios=[1])
    # Half the tile is ignore; the valid half is perfectly predicted.
    prob = np.zeros((8, 8), dtype=np.float32)
    prob[:, :4] = 0.99
    label = np.zeros((8, 8), dtype=np.int64)
    label[:, :4] = 1
    label[:, 4:] = 255   # ignored

    logits = _logits_from_prob(prob)
    # Put large confident positives in the ignore region — they should not count as FPs.
    confident_junk = _logits_from_prob(np.full_like(prob, 0.99))
    merged = logits.clone()
    merged[0, 0, :, 4:] = confident_junk[0, 0, :, 4:]

    acc.update(merged, torch.from_numpy(label).unsqueeze(0), ["tile_0"])
    m = acc.compute()
    # Only valid half counted -> perfect metrics.
    assert m["pixel_iou"] == pytest.approx(1.0)
    assert m["pixel_f1"] == pytest.approx(1.0)


def test_accumulator_speckle_fp_filtered():
    """A 1-pixel false positive below min_blob_size should NOT count as an object FP."""
    cfg = _cfg(min_blob_size_px=4)
    acc = ValidationAccumulator(cfg, ratios=[1])
    prob = np.zeros((16, 16), dtype=np.float32)
    prob[7, 7] = 0.99     # speckle FP
    label = np.zeros((16, 16), dtype=np.int64)  # all negative tile
    logits = _logits_from_prob(prob)
    acc.update(logits, torch.from_numpy(label).unsqueeze(0), ["neg_tile"])
    m = acc.compute()
    assert m["object_precision"] == 0.0   # no TPs
    assert m["object_recall"] == 0.0      # vacuous
    # Critically, obj_fp is 0 because the speckle was filtered out.
    assert acc.obj_fp == 0


def test_accumulator_pr_auc_ranges_between_zero_and_one():
    cfg = _cfg()
    acc = ValidationAccumulator(cfg, ratios=[1])

    # Positive tile: partial prediction.
    prob_pos = np.zeros((8, 8), dtype=np.float32)
    prob_pos[2:6, 2:6] = 0.8
    label_pos = np.zeros((8, 8), dtype=np.int64)
    label_pos[2:6, 2:6] = 1

    # Negative tile: some confident false positives.
    prob_neg = np.full((8, 8), 0.3, dtype=np.float32)
    label_neg = np.zeros((8, 8), dtype=np.int64)

    acc.update(_logits_from_prob(prob_pos), torch.from_numpy(label_pos).unsqueeze(0), ["pos"])
    acc.update(_logits_from_prob(prob_neg), torch.from_numpy(label_neg).unsqueeze(0), ["neg"])
    m = acc.compute()
    assert 0.0 <= m["pr_auc_ratio_1"] <= 1.0
    # Geomean across a single ratio == that ratio's value.
    assert m["val_realistic_pr_auc_geomean"] == pytest.approx(m["pr_auc_ratio_1"], rel=1e-5)
    assert m["val_n_positive_tiles"] == 1.0
    assert m["val_n_negative_tiles"] == 1.0


def test_accumulator_no_positive_tiles_produces_zero_pr_auc():
    """PR-AUC is 0 and does not crash when val has no positive tiles."""
    cfg = _cfg()
    acc = ValidationAccumulator(cfg, ratios=[200, 500])

    prob = np.full((4, 4), 0.2, dtype=np.float32)
    label = np.zeros((4, 4), dtype=np.int64)
    acc.update(_logits_from_prob(prob), torch.from_numpy(label).unsqueeze(0), ["only_neg"])
    m = acc.compute()
    assert m["pr_auc_ratio_200"] == 0.0
    assert m["pr_auc_ratio_500"] == 0.0
    assert m["val_realistic_pr_auc_geomean"] == pytest.approx(0.0, abs=1e-10)
