"""Unit tests for training.scheduler.make_lr_setter."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from training.scheduler import make_lr_setter


def _base_cfg(**overrides) -> dict:
    cfg = {
        "training": {"max_epochs": 50},
        "lr_schedule": {
            "frozen_lr": 1.0e-3,
            "base_lr": 1.0e-4,
            "backbone_lr_multiplier": 0.1,
            "freeze_backbone_epochs": 10,
            "warmup_epochs": 5,
            "warmup_start_lr": 1.0e-6,
            "min_lr": 1.0e-6,
            "backbone_warmup_epochs": 3,
        },
    }
    for k, v in overrides.items():
        cfg["lr_schedule"][k] = v
    return cfg


def _make_optimizer() -> torch.optim.Optimizer:
    linear = nn.Linear(2, 2)
    return torch.optim.AdamW(
        [
            {"name": "decoder", "params": [linear.weight]},
            {"name": "backbone", "params": [linear.bias]},
        ],
        lr=0.0,
    )


def _lr(optim, name) -> float:
    return next(g["lr"] for g in optim.param_groups if g["name"] == name)


def test_phase1_holds_frozen_lr_on_both_groups():
    cfg = _base_cfg()
    set_lrs = make_lr_setter(cfg)
    optim = _make_optimizer()
    for epoch in [1, 5, 10]:
        set_lrs(optim, epoch)
        assert _lr(optim, "decoder") == pytest.approx(1e-3)
        assert _lr(optim, "backbone") == pytest.approx(1e-3)


def test_phase2_decoder_linear_warmup():
    cfg = _base_cfg()
    set_lrs = make_lr_setter(cfg)
    optim = _make_optimizer()
    # Phase 2 starts at epoch 11. Warmup over 5 epochs: 11..15.
    # Mapping: p2_epoch ∈ [1..warmup_epochs] → t ∈ [0..warmup_epochs-1] over t_max=warmup_epochs-1,
    # so warmup starts AT warmup_start_lr and ends AT base_lr.
    set_lrs(optim, 11)
    lr_1 = _lr(optim, "decoder")
    set_lrs(optim, 15)
    lr_5 = _lr(optim, "decoder")
    # At p2_epoch=1 (epoch 11): t=0 → warmup_start_lr exactly.
    assert lr_1 == pytest.approx(1e-6, rel=1e-6)
    # At p2_epoch=5 (epoch 15): t=4=t_max → base_lr exactly.
    assert lr_5 == pytest.approx(1e-4, rel=1e-6)


def test_phase2_backbone_linear_warmup_shorter():
    cfg = _base_cfg()  # backbone_warmup_epochs=3, peaks at 1e-5
    set_lrs = make_lr_setter(cfg)
    optim = _make_optimizer()
    # Backbone peak = base_lr * multiplier = 1e-4 * 0.1 = 1e-5.
    # Warmup mapping: p2_epoch ∈ [1..3] → t ∈ [0..2] over t_max=2.
    set_lrs(optim, 11)  # p2_epoch=1, t=0 → 0.0
    assert _lr(optim, "backbone") == pytest.approx(0.0, abs=1e-12)
    set_lrs(optim, 13)  # p2_epoch=3, t=2=t_max → backbone_peak
    assert _lr(optim, "backbone") == pytest.approx(1e-5, rel=1e-6)
    set_lrs(optim, 14)  # p2_epoch=4 — past backbone warmup, still in decoder warmup → plateau at peak
    assert _lr(optim, "backbone") == pytest.approx(1e-5, rel=1e-6)


def test_cosine_anneal_reaches_min_lr_at_max_epoch():
    cfg = _base_cfg()
    set_lrs = make_lr_setter(cfg)
    optim = _make_optimizer()
    # At the final epoch, cosine should be at min_lr.
    set_lrs(optim, cfg["training"]["max_epochs"])
    assert _lr(optim, "decoder") == pytest.approx(1e-6, abs=1e-9)


def test_cosine_lr_between_peak_and_min_during_decay():
    """Sampled after warmup, cosine LR must strictly be in (min_lr, base_lr)."""
    cfg = _base_cfg()
    set_lrs = make_lr_setter(cfg)
    optim = _make_optimizer()
    # Phase 2 cosine spans epochs 16..50. Pick an interior point.
    set_lrs(optim, 33)
    lr = _lr(optim, "decoder")
    assert 1e-6 < lr < 1e-4
    # Sanity: an earlier cosine epoch should have higher LR than a later one.
    set_lrs(optim, 40)
    lr_later = _lr(optim, "decoder")
    assert lr > lr_later


def test_cosine_exact_halfway_at_t_over_tmax_0p5():
    """At p2_epoch = warmup + cosine_tmax/2, LR = (base_lr + min_lr) / 2."""
    cfg = _base_cfg()
    set_lrs = make_lr_setter(cfg)
    optim = _make_optimizer()
    # cosine_tmax = phase2_total - warmup = 40 - 5 = 35. Halfway cosine_t = 17.5.
    # Our schedule is evaluated at integer epochs, so cosine_t = 17 and 18 bracket the halfway.
    # Pick p2_epoch = 5 + 17.5 -> epoch = 10 + 22.5, not an integer. Verify with
    # the two surrounding integers.
    set_lrs(optim, 32)  # p2_epoch=22, cosine_t=17
    lr_17 = _lr(optim, "decoder")
    set_lrs(optim, 33)  # p2_epoch=23, cosine_t=18
    lr_18 = _lr(optim, "decoder")
    midpoint = 0.5 * (1e-4 + 1e-6)
    # Midpoint lies between the two sampled values.
    assert min(lr_17, lr_18) <= midpoint <= max(lr_17, lr_18)


def test_phase1_epoch_zero_handled_safely():
    """Even though epochs are 1-indexed, epoch 0 (or negative) shouldn't crash."""
    cfg = _base_cfg()
    set_lrs = make_lr_setter(cfg)
    optim = _make_optimizer()
    set_lrs(optim, 0)  # pre-training reset — treat as phase 1.
    assert _lr(optim, "decoder") == pytest.approx(1e-3)


def _range_test_cfg() -> dict:
    return {
        "training": {"max_epochs": 1},
        "lr_schedule": {
            "scheduler": "lr_range_test",
            "lr_range_min": 1.0e-7,
            "lr_range_max": 1.0e-1,
            # The remaining keys are required by the warmup_cosine path but
            # ignored under lr_range_test; setting them to harmless values so
            # the cfg is well-formed.
            "frozen_lr": 1e-3, "base_lr": 1e-4, "backbone_lr_multiplier": 0.1,
            "freeze_backbone_epochs": 9999, "warmup_epochs": 0,
            "warmup_start_lr": 1e-6, "min_lr": 1e-6, "backbone_warmup_epochs": 0,
        },
    }


def test_lr_range_test_endpoints_and_log_midpoint():
    cfg = _range_test_cfg()
    set_lrs = make_lr_setter(cfg)
    optim = _make_optimizer()
    total = 100
    # Step 0 → lr_min.
    set_lrs(optim, step=0, total_steps=total)
    assert _lr(optim, "decoder") == pytest.approx(1e-7, rel=1e-6)
    # Step total-1 → lr_max.
    set_lrs(optim, step=total - 1, total_steps=total)
    assert _lr(optim, "decoder") == pytest.approx(1e-1, rel=1e-6)
    # Exact midpoint (step / (total-1) == 0.5) gives sqrt(lr_min * lr_max).
    set_lrs(optim, step=50, total_steps=101)
    assert _lr(optim, "decoder") == pytest.approx(math.sqrt(1e-7 * 1e-1), rel=1e-6)


def test_lr_range_test_applies_same_lr_to_all_groups():
    cfg = _range_test_cfg()
    set_lrs = make_lr_setter(cfg)
    optim = _make_optimizer()
    set_lrs(optim, step=42, total_steps=100)
    decoder_lr = _lr(optim, "decoder")
    backbone_lr = _lr(optim, "backbone")
    assert decoder_lr == pytest.approx(backbone_lr)


def test_lr_range_test_rejects_invalid_bounds():
    cfg = _range_test_cfg()
    cfg["lr_schedule"]["lr_range_min"] = 1e-1
    cfg["lr_schedule"]["lr_range_max"] = 1e-3  # max < min
    with pytest.raises(ValueError):
        make_lr_setter(cfg)


def test_unknown_scheduler_raises():
    cfg = _base_cfg()
    cfg["lr_schedule"]["scheduler"] = "wat"
    with pytest.raises(ValueError):
        make_lr_setter(cfg)
