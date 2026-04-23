"""Two-phase learning-rate schedule.

Phase 1 (epochs 1..freeze_backbone_epochs): decoder-only at `frozen_lr`;
    backbone params are frozen (requires_grad=False).
Phase 2 (epochs freeze_backbone_epochs+1..max_epochs):
    - Decoder: linear warmup `warmup_start_lr -> base_lr` over `warmup_epochs`,
      then cosine anneal `base_lr -> min_lr` over the remainder.
    - Backbone: separate linear warmup `0 -> base_lr * backbone_lr_multiplier`
      over `backbone_warmup_epochs` (training.md §9.1, risk #17 in plan), then
      cosine matching the decoder's schedule scaled by the multiplier.

Backbone warmup exists because the backbone's LR jumps from 0 (Phase 1) to
its Phase-2 value in a single step; a brief linear ramp prevents catastrophic
forgetting of ImageNet features.

Implementation note: we return a `set_lrs(optimizer, epoch)` callable rather
than a torch.optim.lr_scheduler subclass. Simpler to reason about across the
freeze / unfreeze transition where param groups may be rebuilt.
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import torch.optim as optim

logger = logging.getLogger(__name__)


def _cosine(t: float, t_max: float, lr_hi: float, lr_lo: float) -> float:
    """Cosine decay from lr_hi at t=0 to lr_lo at t=t_max."""
    if t_max <= 0:
        return lr_lo
    t = max(0.0, min(float(t), float(t_max)))
    return lr_lo + (lr_hi - lr_lo) * 0.5 * (1.0 + math.cos(math.pi * t / t_max))


def _linear(t: float, t_max: float, lr_lo: float, lr_hi: float) -> float:
    """Linear ramp from lr_lo at t=0 to lr_hi at t=t_max."""
    if t_max <= 0:
        return lr_hi
    t = max(0.0, min(float(t), float(t_max)))
    return lr_lo + (lr_hi - lr_lo) * (t / t_max)


def make_lr_setter(cfg: dict) -> Callable[[optim.Optimizer, int], None]:
    """Return a callable `set_lrs(optimizer, epoch)` that updates group LRs.

    Epochs are 1-indexed. The function reads both optimizer.param_groups and
    `group['name']` to set per-group LRs. Groups without a `name` key are
    treated as decoder.
    """
    sched = cfg["lr_schedule"]
    max_epochs = int(cfg["training"]["max_epochs"])

    freeze_epochs = int(sched["freeze_backbone_epochs"])
    frozen_lr = float(sched["frozen_lr"])
    base_lr = float(sched["base_lr"])
    backbone_mult = float(sched["backbone_lr_multiplier"])
    warmup_epochs = int(sched["warmup_epochs"])
    warmup_start_lr = float(sched["warmup_start_lr"])
    min_lr = float(sched["min_lr"])
    backbone_warmup = int(sched.get("backbone_warmup_epochs", 0))

    backbone_peak = base_lr * backbone_mult
    backbone_min = min_lr * backbone_mult
    phase2_total = max(1, max_epochs - freeze_epochs)
    cosine_tmax = max(1, phase2_total - warmup_epochs)

    def _decoder_lr(p2_epoch: int) -> float:
        if p2_epoch <= warmup_epochs:
            return _linear(p2_epoch, warmup_epochs, warmup_start_lr, base_lr)
        return _cosine(p2_epoch - warmup_epochs, cosine_tmax, base_lr, min_lr)

    def _backbone_lr(p2_epoch: int) -> float:
        if p2_epoch <= backbone_warmup:
            return _linear(p2_epoch, backbone_warmup, 0.0, backbone_peak)
        # After backbone warmup, follow decoder's cosine shape scaled by multiplier.
        if p2_epoch <= warmup_epochs:
            # Plateau at peak while decoder still warms up.
            return backbone_peak
        return _cosine(p2_epoch - warmup_epochs, cosine_tmax, backbone_peak, backbone_min)

    def set_lrs(optimizer: optim.Optimizer, epoch: int) -> None:
        if epoch <= freeze_epochs:
            for group in optimizer.param_groups:
                group["lr"] = frozen_lr
            return

        p2_epoch = epoch - freeze_epochs
        dec_lr = _decoder_lr(p2_epoch)
        bb_lr = _backbone_lr(p2_epoch)

        for group in optimizer.param_groups:
            if group.get("name") == "backbone":
                group["lr"] = bb_lr
            else:
                group["lr"] = dec_lr

    logger.info(
        "LR setter built: freeze_epochs=%d, base_lr=%g, backbone_mult=%g, "
        "warmup=%d, backbone_warmup=%d, min_lr=%g",
        freeze_epochs, base_lr, backbone_mult, warmup_epochs, backbone_warmup, min_lr,
    )
    return set_lrs
