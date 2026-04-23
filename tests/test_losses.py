"""Unit tests for losses.segmentation_losses.

Covers: hand-computed reference values, Tversky(a=b=0.5) == Dice sanity,
ignore-index masking, finite gradients at extreme logits, build_loss
dispatcher.
"""

from __future__ import annotations

import math

import pytest
import torch

from losses import (
    CompoundLoss,
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    build_loss,
)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------


def test_focal_loss_matches_hand_computed():
    """At logit=0, p=0.5, gamma=2, alpha=0.25, y=1:
    FL = -0.25 * (1 - 0.5)^2 * log(0.5) = 0.25 * 0.25 * 0.6931 ≈ 0.04332.
    """
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25, ignore_index=255)
    logits = torch.zeros(1, 1, 1, 1)
    label = torch.ones(1, 1, 1, dtype=torch.long)
    loss = loss_fn(logits, label)
    expected = 0.25 * 0.25 * math.log(2.0)
    assert math.isclose(loss.item(), expected, rel_tol=1e-5)


def test_focal_loss_zero_at_perfect_prediction():
    """Very high-confidence correct prediction -> near-zero loss."""
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    logits = torch.full((1, 1, 4, 4), 30.0)
    label = torch.ones(1, 4, 4, dtype=torch.long)
    loss = loss_fn(logits, label)
    assert loss.item() < 1e-10


def test_focal_loss_ignore_mask_respected():
    """Pixels with label=255 contribute nothing; adding them should not change loss."""
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25, ignore_index=255)
    logits = torch.zeros(1, 1, 2, 2)
    # All valid: two positives, two negatives.
    label_full = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.long)
    loss_full = loss_fn(logits, label_full)

    # Same two valid pixels; add a large junk logit at ignore positions.
    logits_with_junk = torch.tensor([[[[0.0, 0.0], [10.0, -10.0]]]])  # last two values shouldn't matter
    label_masked = torch.tensor([[[1, 0], [255, 255]]], dtype=torch.long)
    loss_masked = loss_fn(logits_with_junk, label_masked)

    # Both losses are the mean over valid pixels — 2 pixels either way.
    # Mean of (positive-at-0 + negative-at-0) must match.
    assert math.isclose(loss_full.item(), loss_masked.item(), rel_tol=1e-5)


@pytest.mark.parametrize("logit_value", [-30.0, -10.0, 0.0, 10.0, 30.0])
@pytest.mark.parametrize("y_value", [0, 1])
def test_focal_loss_finite_gradient_at_extreme_logits(logit_value, y_value):
    """Gradient must be finite across the full float range — no NaN/Inf."""
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
    logits = torch.full((1, 1, 1, 1), logit_value, requires_grad=True)
    label = torch.full((1, 1, 1), y_value, dtype=torch.long)
    loss = loss_fn(logits, label)
    loss.backward()
    grad = logits.grad
    assert torch.isfinite(grad).all(), (
        f"Non-finite gradient at logit={logit_value}, y={y_value}: {grad}"
    )


# ---------------------------------------------------------------------------
# Dice loss
# ---------------------------------------------------------------------------


def test_dice_loss_perfect_prediction_near_zero():
    """Confident correct prediction -> dice ~ 1 -> loss ~ 0."""
    loss_fn = DiceLoss(eps=1e-6)
    logits = torch.full((1, 1, 4, 4), 20.0)
    label = torch.ones(1, 4, 4, dtype=torch.long)
    loss = loss_fn(logits, label)
    assert loss.item() < 1e-3


def test_dice_loss_empty_mask_stable():
    """All-negative tile with eps > 0 should produce a finite loss (not NaN)."""
    loss_fn = DiceLoss(eps=1.0)
    logits = torch.full((1, 1, 4, 4), -20.0)
    label = torch.zeros(1, 4, 4, dtype=torch.long)
    loss = loss_fn(logits, label)
    # p ~ 0, y ~ 0 -> (2*0+1)/(0+0+1) = 1 -> loss ~ 0, but finite regardless.
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Tversky loss
# ---------------------------------------------------------------------------


def test_tversky_reduces_to_dice_at_half_half():
    """Tversky(alpha=beta=0.5, eps=e) is algebraically equal to Dice(eps=2e).

    Dice      = (2 TP + e_D) / (2 TP + FN + FP + e_D)
    Tversky   = (TP + e_T)   / (TP + 0.5 (FN + FP) + e_T)
              = (2 TP + 2 e_T) / (2 TP + FN + FP + 2 e_T)    [*2/2]
    so e_D = 2 e_T.
    """
    dice = DiceLoss(eps=2.0)
    tversky = TverskyLoss(alpha=0.5, beta=0.5, eps=1.0)
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    label = torch.randint(0, 2, (2, 8, 8), dtype=torch.long)
    assert math.isclose(dice(logits, label).item(), tversky(logits, label).item(), rel_tol=1e-5)


def test_tversky_beta_greater_alpha_penalizes_fps_more():
    """With FPs dominating, beta>alpha should produce higher loss than alpha>beta."""
    # Logits heavily predict positive everywhere, label is all negative -> pure FPs.
    logits = torch.full((1, 1, 8, 8), 5.0)
    label = torch.zeros(1, 8, 8, dtype=torch.long)
    fp_heavy = TverskyLoss(alpha=0.3, beta=0.7)
    fn_heavy = TverskyLoss(alpha=0.7, beta=0.3)
    assert fp_heavy(logits, label) > fn_heavy(logits, label)


# ---------------------------------------------------------------------------
# Compound loss
# ---------------------------------------------------------------------------


def test_compound_loss_weighted_sum():
    """CompoundLoss returns lambda_focal * focal + lambda_dice * dice."""
    focal = FocalLoss(gamma=2.0, alpha=0.25)
    dice = DiceLoss(eps=1.0)
    compound = CompoundLoss(focal, dice, lambda_focal=0.5, lambda_dice=2.0)

    torch.manual_seed(0)
    logits = torch.randn(1, 1, 8, 8)
    label = torch.randint(0, 2, (1, 8, 8), dtype=torch.long)

    fl = focal(logits, label)
    dl = dice(logits, label)
    cl = compound(logits, label)
    assert math.isclose(cl.item(), 0.5 * fl.item() + 2.0 * dl.item(), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# build_loss dispatcher
# ---------------------------------------------------------------------------


def _cfg(function: str, **loss_overrides) -> dict:
    return {
        "data": {"label_ignore_index": 255},
        "loss": {"function": function, **loss_overrides},
    }


@pytest.mark.parametrize("name,expected_cls", [
    ("focal", FocalLoss),
    ("dice", DiceLoss),
    ("tversky", TverskyLoss),
    ("compound", CompoundLoss),
])
def test_build_loss_dispatch(name, expected_cls):
    loss = build_loss(_cfg(name))
    assert isinstance(loss, expected_cls)


def test_build_loss_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss.function"):
        build_loss(_cfg("hinge"))
