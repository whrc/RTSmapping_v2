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
    set_lrs(optim, 11)
    lr_1 = _lr(optim, "decoder")
    set_lrs(optim, 15)
    lr_5 = _lr(optim, "decoder")
    # At p2_epoch=1 (epoch 11): linear from 1e-6 toward 1e-4, so LR ≈ 1e-6 + (1e-4 - 1e-6) * 1/5.
    expected_1 = 1e-6 + (1e-4 - 1e-6) * 1.0 / 5.0
    assert lr_1 == pytest.approx(expected_1, rel=1e-6)
    # At p2_epoch=5 (epoch 15): linear reaches base_lr = 1e-4.
    assert lr_5 == pytest.approx(1e-4, rel=1e-6)


def test_phase2_backbone_linear_warmup_shorter():
    cfg = _base_cfg()  # backbone_warmup_epochs=3, peaks at 1e-5
    set_lrs = make_lr_setter(cfg)
    optim = _make_optimizer()
    # Backbone peak = base_lr * multiplier = 1e-4 * 0.1 = 1e-5.
    set_lrs(optim, 11)  # p2_epoch=1
    assert _lr(optim, "backbone") == pytest.approx(1e-5 * 1.0 / 3.0, rel=1e-6)
    set_lrs(optim, 14)  # p2_epoch=4 — past backbone warmup, still in decoder warmup -> plateau at peak
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
