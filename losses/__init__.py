"""Segmentation losses for the RTS pipeline.

All losses operate on **logits** (training.md §4.2). All honour
`ignore_index=255` produced by boundary-dilation (training.md §5.5) and
NoData masking (training.md §4.4).

Public API:
    FocalLoss, DiceLoss, TverskyLoss, CompoundLoss
    build_loss(cfg) -> nn.Module
"""

from __future__ import annotations

from losses.segmentation_losses import (
    CompoundLoss,
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    build_loss,
)

__all__ = [
    "FocalLoss",
    "DiceLoss",
    "TverskyLoss",
    "CompoundLoss",
    "build_loss",
]
