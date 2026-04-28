"""Unit tests for models.segmentation.build_model.

Scope: verify build_model produces a working nn.Module with the contracted
output shape, logits output (no sigmoid applied), and correctly initialised
final-conv bias. Uses untrained weights (pretrained=False) for speed.
"""

from __future__ import annotations

import math

import pytest
import torch

from models import build_model


def _base_cfg(**model_overrides) -> dict:
    """Minimal config matching configs/baseline.yaml schema for model.build_model."""
    return {
        "channels": {"rgb": True, "extra": []},
        "model": {
            "architecture": "unetplusplus",
            "backbone": "efficientnet-b5",
            "pretrained": False,
            "output_bias_prior": 0.5,
            **model_overrides,
        },
    }


def test_build_model_rgb_only_output_shape():
    """Baseline RGB-only config produces (B, 1, 512, 512) logits."""
    cfg = _base_cfg()
    model = build_model(cfg).eval()
    x = torch.zeros(2, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 512, 512)


def test_build_model_with_extra_channels():
    """Adding 4 EXTRA channels makes in_channels=7 without crashing."""
    cfg = _base_cfg()
    cfg["channels"]["extra"] = [
        {"name": "ndvi", "band": 0},
        {"name": "nir", "band": 1},
        {"name": "re", "band": 2},
        {"name": "sr", "band": 3},
    ]
    model = build_model(cfg).eval()
    x = torch.zeros(1, 7, 256, 256)  # smaller to keep test fast
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 256)


def test_output_bias_initialized_to_class_prior():
    """Final-conv bias is set to -log((1-pi)/pi); at pi=0.5 that's 0.0."""
    cfg = _base_cfg(output_bias_prior=0.5)
    model = build_model(cfg)
    bias = model.segmentation_head[0].bias.detach().item()
    assert math.isclose(bias, 0.0, abs_tol=1e-6)


def test_output_bias_for_imbalanced_prior():
    """At pi=0.01, bias = -log(99) ~= -4.595; verifies the prior flows through."""
    cfg = _base_cfg(output_bias_prior=0.01)
    model = build_model(cfg)
    bias = model.segmentation_head[0].bias.detach().item()
    expected = -math.log((1.0 - 0.01) / 0.01)
    assert math.isclose(bias, expected, abs_tol=1e-5)


def test_output_is_logits_not_probabilities():
    """Raw outputs should span a range typical of logits, not [0, 1]."""
    cfg = _base_cfg()
    model = build_model(cfg).eval()
    # Random input forces varied features; still not trained, but logits should
    # have values outside [0, 1] somewhere after a random-conv pass.
    torch.manual_seed(0)
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        y = model(x)
    assert y.min().item() < 0.0 or y.max().item() > 1.0, (
        "Output appears bounded to [0, 1] — activation may not be None "
        "(expected logits per training.md §4.2)"
    )


def test_invalid_bias_prior_rejected():
    """Prior outside (0, 1) raises rather than producing nan/inf bias."""
    cfg = _base_cfg(output_bias_prior=0.0)
    with pytest.raises(ValueError, match="output_bias_prior"):
        build_model(cfg)


def test_unknown_architecture_rejected():
    """Unsupported architecture is a clear error, not a silent smp traceback."""
    cfg = _base_cfg(architecture="segformer")
    with pytest.raises(ValueError, match="Unsupported model.architecture"):
        build_model(cfg)
