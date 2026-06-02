"""Binary segmentation losses: Focal, Dice, Tversky, Compound.

Design notes:
  - All losses take `logits` of shape (B, 1, H, W) and integer `label` of
    shape (B, H, W) with values {0, 1, ignore_index}. No sigmoid is applied
    upstream (training.md §4.2).
  - Numerical stability: BCE is computed via binary_cross_entropy_with_logits;
    sigmoid(logits) is numerically stable in PyTorch and used as the forward
    probability value. Never compute log(sigmoid(x)) naively.
  - Ignore handling: pixels where label == ignore_index contribute zero loss
    and zero gradient. This reuses the existing boundary-dilation machinery
    (training.md §5.5) and NoData masking (training.md §4.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare(
    logits: torch.Tensor,
    label: torch.Tensor,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten to (B, H*W) for logits + binary target + valid-mask.

    `label` is int64 with values {0, 1, ignore_index}. The returned target is
    float in {0., 1.}, and the mask is float in {0., 1.} where 0 means "ignore".
    """
    if logits.ndim == 4:
        if logits.shape[1] != 1:
            raise ValueError(
                f"Expected single-channel logits, got shape {tuple(logits.shape)}"
            )
        logits = logits.squeeze(1)  # (B, H, W)
    if logits.shape != label.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != label shape {tuple(label.shape)}"
        )
    mask = (label != ignore_index).to(logits.dtype)
    # Replace ignore values with 0 in the target so multiplication is safe.
    target = torch.where(label == 1, 1, 0).to(logits.dtype)
    return logits, target, mask


class FocalLoss(nn.Module):
    """Binary focal loss on logits.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), where
    p_t = p if y=1 else (1-p), alpha_t = alpha if y=1 else (1-alpha).

    Args:
        gamma: Focusing parameter >= 0. Higher emphasises hard examples.
        alpha: Positive-class weight in [0, 1].
        ignore_index: Label value excluded from the loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, ignore_index: int = 255):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        logits, target, mask = _prepare(logits, label, self.ignore_index)
        # bce = -log(p_t) stably (no log(sigmoid(x))).
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = target * p + (1.0 - target) * (1.0 - p)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = alpha_t * focal_weight * bce
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom


class DiceLoss(nn.Module):
    """Soft Dice loss, averaged over the batch.

    dice = (2 * sum(p * y) + eps) / (sum(p) + sum(y) + eps), L = 1 - dice.

    Args:
        eps: Smoothing constant, prevents division by zero on empty masks.
        ignore_index: Label value excluded from the loss.
    """

    def __init__(self, eps: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.eps = float(eps)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        logits, target, mask = _prepare(logits, label, self.ignore_index)
        p = torch.sigmoid(logits) * mask
        y = target * mask
        # Per-sample reduction over spatial dims, then mean over batch.
        inter = (p * y).flatten(1).sum(dim=1)
        denom = p.flatten(1).sum(dim=1) + y.flatten(1).sum(dim=1)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return (1.0 - dice).mean()


class TverskyLoss(nn.Module):
    """Generalised Dice with asymmetric FN/FP weighting.

    TL = 1 - (TP + eps) / (TP + alpha * FN + beta * FP + eps).
    alpha = beta = 0.5 reduces to Dice. For precision-focused runs set
    beta > alpha (training.md §5.2).

    Args:
        alpha: Weight on false negatives.
        beta: Weight on false positives.
        eps: Smoothing constant.
        ignore_index: Label value excluded from the loss.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        eps: float = 1.0,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        logits, target, mask = _prepare(logits, label, self.ignore_index)
        p = torch.sigmoid(logits) * mask
        y = target * mask
        tp = (p * y).flatten(1).sum(dim=1)
        fn = ((1.0 - p) * y).flatten(1).sum(dim=1)
        fp = (p * (1.0 - y)).flatten(1).sum(dim=1)
        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        return (1.0 - tversky).mean()


class CompoundLoss(nn.Module):
    """Weighted sum of Focal + Dice (training.md §5.3).

    L = lambda_focal * Focal + lambda_dice * Dice.

    Do NOT stack Tversky on top of Dice — they're the same family.
    """

    def __init__(
        self,
        focal: FocalLoss,
        dice: DiceLoss,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
    ):
        super().__init__()
        self.focal = focal
        self.dice = dice
        self.lambda_focal = float(lambda_focal)
        self.lambda_dice = float(lambda_dice)

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.lambda_focal * self.focal(logits, label) + self.lambda_dice * self.dice(logits, label)


def build_loss(cfg: dict) -> nn.Module:
    """Dispatch on `cfg['loss']['function']`.

    Supported: focal | dice | tversky | compound. See training.md §5.

    Soft-label handling is deferred to v2.1 — `data/dataset.py` raises if
    `boundary_handling: soft_labels` is requested (training.md §5.5).
    """
    loss_cfg = cfg["loss"]
    ignore_index = int(cfg["data"]["label_ignore_index"])
    name = loss_cfg["function"].lower()

    if name == "focal":
        return FocalLoss(
            gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=loss_cfg.get("focal_alpha", 0.25),
            ignore_index=ignore_index,
        )
    if name == "dice":
        return DiceLoss(
            eps=loss_cfg.get("dice_eps", 1.0),
            ignore_index=ignore_index,
        )
    if name == "tversky":
        return TverskyLoss(
            alpha=loss_cfg.get("tversky_alpha", 0.3),
            beta=loss_cfg.get("tversky_beta", 0.7),
            eps=loss_cfg.get("tversky_eps", 1.0),
            ignore_index=ignore_index,
        )
    if name == "compound":
        focal = FocalLoss(
            gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=loss_cfg.get("focal_alpha", 0.25),
            ignore_index=ignore_index,
        )
        dice = DiceLoss(
            eps=loss_cfg.get("dice_eps", 1.0),
            ignore_index=ignore_index,
        )
        return CompoundLoss(
            focal=focal,
            dice=dice,
            lambda_focal=loss_cfg.get("lambda_focal", 1.0),
            lambda_dice=loss_cfg.get("lambda_dice", 1.0),
        )
    raise ValueError(
        f"Unknown loss.function: {name!r}. "
        f"Expected one of: focal, dice, tversky, compound."
    )
