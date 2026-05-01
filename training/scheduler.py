"""Learning-rate schedules.

Two modes, selected by `cfg["lr_schedule"]["scheduler"]`:

- `"warmup_cosine"` (default): two-phase per-epoch schedule.
  - Phase 1 (epochs 1..freeze_backbone_epochs): decoder-only at `frozen_lr`;
    backbone params are frozen (requires_grad=False).
  - Phase 2 (epochs freeze_backbone_epochs+1..max_epochs):
    - Decoder: linear warmup `warmup_start_lr -> base_lr` over `warmup_epochs`
      such that p2_epoch=1 lands at `warmup_start_lr` and p2_epoch=warmup_epochs
      lands at `base_lr`. Then cosine `base_lr -> min_lr` over the remainder.
    - Backbone: separate linear warmup `0 -> base_lr * backbone_lr_multiplier`
      over `backbone_warmup_epochs`, then cosine matching the decoder shape.

- `"lr_range_test"` (Phase 0 §3.2): logarithmic per-step ramp from
  `lr_range_min` to `lr_range_max` across the full run. All param groups receive
  the same LR. Caller drives this per-step rather than per-epoch (see train.py).

The returned callable is `set_lrs(optimizer, epoch, *, step=0, total_steps=1)`.
The unused arguments are ignored by whichever mode applies. Caller is responsible
for invoking with the appropriate cadence (per-epoch for warmup_cosine, per-step
for lr_range_test).

`scheduler_state` saved in resume checkpoints is informational under both modes
since both schedules are pure functions of (epoch, step) — no internal state to
restore.
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


def make_lr_setter(cfg: dict) -> Callable[..., None]:
    """Return a callable `set_lrs(optimizer, epoch, *, step=0, total_steps=1)`.

    Dispatches on `cfg["lr_schedule"]["scheduler"]` (default "warmup_cosine").
    """
    sched_type = cfg["lr_schedule"].get("scheduler", "warmup_cosine")
    if sched_type == "lr_range_test":
        return _make_lr_range_test_setter(cfg)
    if sched_type in (None, "warmup_cosine"):
        return _make_warmup_cosine_setter(cfg)
    raise ValueError(f"Unknown lr_schedule.scheduler: {sched_type!r}")


def _make_warmup_cosine_setter(cfg: dict) -> Callable[..., None]:
    """Two-phase warmup→cosine schedule. Per-epoch.

    Epochs are 1-indexed. Reads `optimizer.param_groups` and `group['name']`
    to set per-group LRs. Groups without a `name` key are treated as decoder.
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

    # Linear warmup spans p2_epoch ∈ [1..warmup_epochs] mapped to t ∈ [0..warmup_epochs-1]
    # so that p2_epoch=1 → warmup_start_lr and p2_epoch=warmup_epochs → base_lr.
    warmup_tmax = max(1, warmup_epochs - 1)
    bb_warmup_tmax = max(1, backbone_warmup - 1) if backbone_warmup > 0 else 0

    def _decoder_lr(p2_epoch: int) -> float:
        if p2_epoch <= warmup_epochs:
            return _linear(p2_epoch - 1, warmup_tmax, warmup_start_lr, base_lr)
        return _cosine(p2_epoch - warmup_epochs, cosine_tmax, base_lr, min_lr)

    def _backbone_lr(p2_epoch: int) -> float:
        if backbone_warmup > 0 and p2_epoch <= backbone_warmup:
            return _linear(p2_epoch - 1, bb_warmup_tmax, 0.0, backbone_peak)
        if p2_epoch <= warmup_epochs:
            return backbone_peak
        return _cosine(p2_epoch - warmup_epochs, cosine_tmax, backbone_peak, backbone_min)

    def set_lrs(optimizer: optim.Optimizer, epoch: int, *, step: int = 0, total_steps: int = 1) -> None:
        del step, total_steps  # unused in this mode
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
        "LR setter built (warmup_cosine): freeze_epochs=%d, base_lr=%g, backbone_mult=%g, "
        "warmup=%d, backbone_warmup=%d, min_lr=%g",
        freeze_epochs, base_lr, backbone_mult, warmup_epochs, backbone_warmup, min_lr,
    )
    return set_lrs


def _make_lr_range_test_setter(cfg: dict) -> Callable[..., None]:
    """Logarithmic per-step LR ramp from lr_range_min → lr_range_max.

    Per training/experiments.md §3.2: ramp LR across the run's training steps;
    pick the order of magnitude where the loss curve has the steepest stable
    descent before divergence. All param groups receive the same LR (the
    decoder/backbone split is irrelevant in single-epoch range test).
    """
    sched = cfg["lr_schedule"]
    lr_min = float(sched["lr_range_min"])
    lr_max = float(sched["lr_range_max"])
    if lr_min <= 0 or lr_max <= 0 or lr_min >= lr_max:
        raise ValueError(f"lr_range_test requires 0 < lr_range_min < lr_range_max; got {lr_min}, {lr_max}")
    log_min = math.log(lr_min)
    log_max = math.log(lr_max)

    def set_lrs(optimizer: optim.Optimizer, epoch: int = 1, *, step: int = 0, total_steps: int = 1) -> None:
        del epoch  # unused; range test is per-step
        denom = max(1, int(total_steps) - 1)
        frac = max(0.0, min(1.0, float(step) / denom))
        lr = math.exp(log_min + frac * (log_max - log_min))
        for group in optimizer.param_groups:
            group["lr"] = lr

    logger.info("LR setter built (lr_range_test): %g → %g logarithmic", lr_min, lr_max)
    return set_lrs


def is_lr_range_test(cfg: dict) -> bool:
    """True if cfg selects the lr_range_test scheduler (helper for callers)."""
    return cfg.get("lr_schedule", {}).get("scheduler") == "lr_range_test"
