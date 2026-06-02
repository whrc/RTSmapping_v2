"""Backbone freeze / unfreeze + optimizer param-group builder.

smp models expose the encoder as `model.encoder`. Phase 1 (frozen) trains only
the decoder + segmentation head; Phase 2 (full) trains the whole model with a
reduced LR on the backbone (training.md §9.1).

Param groups are **named** so the scheduler can target them by name. The
optimizer is built once with two groups (decoder + backbone); freezing is done
by setting `requires_grad=False` on backbone params, which stops gradient
computation regardless of LR.
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def freeze_backbone(model: nn.Module) -> None:
    """Disable gradient for all encoder parameters."""
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    logger.info(
        "Backbone frozen (%d params)",
        sum(p.numel() for p in model.encoder.parameters()),
    )


def unfreeze_backbone(model: nn.Module) -> None:
    """Re-enable gradient for all encoder parameters."""
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    logger.info("Backbone unfrozen")


def build_param_groups(
    model: nn.Module,
    decoder_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Return two named param groups for the AdamW optimizer.

    The "name" key is inspected by the scheduler (training.scheduler) to set
    each group's LR per epoch. PyTorch ignores unknown keys in param_group
    dicts.

    Args:
        model: The segmentation model (must expose `.encoder`).
        decoder_lr: Initial LR for non-encoder params.
        backbone_lr: Initial LR for encoder params.
        weight_decay: Applied identically to both groups.
    """
    backbone_params = list(model.encoder.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    decoder_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    return [
        {"name": "decoder", "params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay},
        {"name": "backbone", "params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
    ]
