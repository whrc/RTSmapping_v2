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


def _vit_layer_index(param_name: str, n_blocks: int) -> int:
    """Depth index of an encoder param for LLRD: stem=0, blocks.i=i+1, top (final
    norm / other)=n_blocks+1. Higher index = closer to the head = higher LR."""
    if ".blocks." in param_name:
        return int(param_name.split(".blocks.")[1].split(".")[0]) + 1
    tail = param_name.split("encoder.", 1)[-1]
    if tail.startswith(("patch_embed", "cls_token", "pos_embed", "reg_token", "mask_token")):
        return 0
    return n_blocks + 1


def build_llrd_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    llrd_decay: float,
) -> list[dict]:
    """Layer-wise LR-decay param groups for a ViT-style encoder (§8.2a).

    Each encoder layer (stem → blocks → final-norm) becomes its own named-"backbone"
    group carrying `lr_scale = llrd_decay ** (top - layer_index)`, so the top encoder
    layer keeps the full backbone LR and earlier layers decay toward the stem
    (protecting general low-level pretrained features). Non-encoder params (decoder,
    pyramid, head) form one "decoder" group with `lr_scale = 1.0`.

    The scheduler multiplies each group's per-epoch LR by `lr_scale` (so the encoder
    vs decoder ratio is still set by `backbone_lr_multiplier`; LLRD adds the taper).
    All groups start at `lr`; the scheduler overrides per epoch.

    Args:
        model: must expose `.encoder` with a `.blocks` ModuleList, either directly (ViT/Eva)
            or one level down under timm's `features_only` wrapper (SAM2/Hiera).
        lr: initial LR for every group (overwritten by the scheduler).
        weight_decay: applied to all groups.
        llrd_decay: per-layer decay factor in (0, 1] (e.g. 0.7).
    """
    if not (0.0 < llrd_decay <= 1.0):
        raise ValueError(f"llrd_decay must be in (0, 1], got {llrd_decay}")
    enc = model.encoder
    from models.foundation import inner_encoder  # local: keeps timm off freeze.py's import path

    # Hierarchical encoders (SAM2/Hiera) sit under timm's features_only wrapper, which holds
    # the real module at `.model` — so `.blocks` is one level down. fm_sam2_rgb disabled LLRD
    # believing Hiera had no `.blocks` at all; it has 16. Param NAMES still contain ".blocks."
    # either way, so _vit_layer_index is unchanged.
    blocks = getattr(enc, "blocks", None)
    if blocks is None:
        blocks = getattr(inner_encoder(enc), "blocks", None)
    if blocks is None:
        raise ValueError(
            f"encoder {type(enc).__name__} exposes no `.blocks` ModuleList (directly or via "
            "the features_only wrapper) — LLRD cannot map layers; set llrd_decay: null"
        )
    n_blocks = len(blocks)
    top = n_blocks + 1
    enc_ids = {id(p) for p in enc.parameters()}

    by_layer: dict[int, list] = {}
    decoder_params: list = []
    for name, p in model.named_parameters():
        if id(p) in enc_ids:
            by_layer.setdefault(_vit_layer_index(name, n_blocks), []).append(p)
        else:
            decoder_params.append(p)

    groups: list[dict] = []
    for layer in sorted(by_layer):
        scale = llrd_decay ** (top - layer)
        groups.append({"name": "backbone", "params": by_layer[layer], "lr": lr,
                       "weight_decay": weight_decay, "lr_scale": scale})
    groups.append({"name": "decoder", "params": decoder_params, "lr": lr,
                   "weight_decay": weight_decay, "lr_scale": 1.0})
    logger.info("LLRD param groups: %d encoder layers (decay=%g) + 1 decoder group",
                len(by_layer), llrd_decay)
    return groups
