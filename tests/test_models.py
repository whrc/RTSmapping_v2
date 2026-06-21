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


def test_build_model_segformer_output_shape_and_bias():
    """SegFormer (mit_b5) builds, returns (B, 1, H, W) logits at input
    resolution, and shares the .segmentation_head[0] bias path so the rare-class
    prior is applied identically to UnetPlusPlus (fair architecture comparison)."""
    cfg = _base_cfg(architecture="segformer", backbone="mit_b5",
                    output_bias_prior=0.01)
    model = build_model(cfg).eval()
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 256)
    bias = model.segmentation_head[0].bias.detach().item()
    assert math.isclose(bias, -math.log((1.0 - 0.01) / 0.01), abs_tol=1e-5)


@pytest.mark.parametrize("arch", ["deeplabv3plus", "fpn", "pspnet", "manet"])
def test_build_model_smp_decoder_sweep(arch):
    """§8.2 architecture sweep: each smp decoder drop-in builds on EffB5, returns
    (B, 1, H, W) logits at input resolution, and shares the .segmentation_head[0]
    bias path so the rare-class prior is applied identically (fair comparison)."""
    cfg = _base_cfg(architecture=arch, backbone="efficientnet-b5",
                    output_bias_prior=0.01)
    model = build_model(cfg).eval()
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 256)
    bias = model.segmentation_head[0].bias.detach().item()
    assert math.isclose(bias, -math.log((1.0 - 0.01) / 0.01), abs_tol=1e-5)


def test_unknown_architecture_rejected():
    """Unsupported architecture is a clear error, not a silent smp traceback.

    (Uses 'bogusnet' — 'segformer' is now a supported architecture.)"""
    cfg = _base_cfg(architecture="bogusnet")
    with pytest.raises(ValueError, match="Unsupported model.architecture"):
        build_model(cfg)


# --- Fusion methods (second-wave Step 1): F1 stem_init, F2 chan_attn -----------
# F4 (ensemble) is eval-side (average two trained models' probabilities) — no
# build_model branch beyond accepting the value, so it is covered by
# test_fusion_default_and_ensemble_are_plain_models below.


def _extra_fusion_cfg(fusion: str, n_extra: int = 2, **kw) -> dict:
    """Small (efficientnet-b0) config with EXTRA channels + a fusion method."""
    cfg = _base_cfg(backbone="efficientnet-b0", fusion=fusion, **kw)
    cfg["channels"]["extra"] = [{"name": f"e{i}", "band": i} for i in range(n_extra)]
    return cfg


def test_fusion_default_and_ensemble_are_plain_models():
    """No fusion key (= early) and fusion='ensemble' both build a normal model
    (ensemble averaging is an eval-side step, not a model change)."""
    for cfg in (_extra_fusion_cfg("early"), _extra_fusion_cfg("ensemble")):
        model = build_model(cfg).eval()
        x = torch.zeros(2, 5, 64, 64)  # 3 RGB + 2 EXTRA
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 1, 64, 64)


def test_fusion_stem_init_zeroes_extra_input_channels():
    """F1: encoder stem conv has zero weights on EXTRA channels (>=3), nonzero RGB."""
    cfg = _extra_fusion_cfg("stem_init", n_extra=2)
    model = build_model(cfg)
    stem = next(m for m in model.encoder.modules()
                if isinstance(m, torch.nn.Conv2d) and m.in_channels == 5)
    w = stem.weight.detach()
    assert torch.count_nonzero(w[:, 3:]) == 0   # EXTRA channels zeroed
    assert torch.count_nonzero(w[:, :3]) > 0    # RGB channels intact


def test_fusion_stem_init_invariant_to_extra_at_init():
    """F1: at init, changing only the EXTRA channels does not change the output —
    the model starts identical to RGB-only."""
    cfg = _extra_fusion_cfg("stem_init", n_extra=2)
    model = build_model(cfg).eval()
    torch.manual_seed(0)
    rgb = torch.randn(2, 3, 64, 64)
    x1 = torch.cat([rgb, torch.zeros(2, 2, 64, 64)], dim=1)
    x2 = torch.cat([rgb, torch.randn(2, 2, 64, 64)], dim=1)
    with torch.no_grad():
        y1, y2 = model(x1), model(x2)
    assert torch.allclose(y1, y2, atol=1e-5)


def test_fusion_chan_attn_shape_gate_and_delegation():
    """F2: wrapper returns (B,1,H,W) logits, exposes .encoder + .segmentation_head
    (freeze.py + bias-init compatibility), and the gate is per-channel in (0,1)."""
    cfg = _extra_fusion_cfg("chan_attn", n_extra=2, output_bias_prior=0.01)
    model = build_model(cfg).eval()
    assert hasattr(model, "encoder") and hasattr(model, "segmentation_head")
    # bias-init flowed through to the delegated head
    assert math.isclose(model.segmentation_head[0].bias.detach().item(),
                        -math.log((1.0 - 0.01) / 0.01), abs_tol=1e-5)
    x = torch.zeros(2, 5, 64, 64)
    with torch.no_grad():
        y = model(x)
        g = model.gate(x)
    assert y.shape == (2, 1, 64, 64)
    assert g.shape == (2, 5, 1, 1)
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0


@pytest.mark.parametrize("fusion", ["dual_encoder", "cross_modal"])
def test_heavy_fusion_shape_delegation_and_uses_extra(fusion):
    """F3/F5: (B,1,H,W) logits, exposes .encoder + .segmentation_head (freeze + bias-init),
    has a second EXTRA encoder, and the output actually depends on the EXTRA channels."""
    cfg = _extra_fusion_cfg(fusion, n_extra=2, output_bias_prior=0.01)
    model = build_model(cfg).eval()
    assert hasattr(model, "encoder") and hasattr(model, "segmentation_head")
    assert hasattr(model, "extra_encoder")  # dual-encoder scaffold
    assert math.isclose(model.segmentation_head[0].bias.detach().item(),
                        -math.log((1.0 - 0.01) / 0.01), abs_tol=1e-5)
    torch.manual_seed(0)
    rgb = torch.randn(2, 3, 64, 64)
    x1 = torch.cat([rgb, torch.zeros(2, 2, 64, 64)], dim=1)
    x2 = torch.cat([rgb, torch.randn(2, 2, 64, 64)], dim=1)
    with torch.no_grad():
        y1, y2 = model(x1), model(x2)
    assert y1.shape == (2, 1, 64, 64)
    assert not torch.allclose(y1, y2, atol=1e-4)  # EXTRA stream influences the output


def test_heavy_fusion_rejects_rgb_only():
    """F3/F5 need EXTRA channels — building with in_channels=3 must raise."""
    with pytest.raises(ValueError, match="EXTRA"):
        build_model(_base_cfg(backbone="efficientnet-b0", fusion="dual_encoder"))


def test_fusion_chan_attn_param_groups_split():
    """F2 gate params land in the non-encoder (decoder) group so the freeze
    schedule + backbone-LR multiplier still target the right params."""
    from training.freeze import build_param_groups
    cfg = _extra_fusion_cfg("chan_attn", n_extra=2)
    model = build_model(cfg)
    groups = {g["name"]: g["params"] for g in build_param_groups(model, 1e-3, 1e-4, 1e-2)}
    gate_ids = {id(p) for p in model.gate.parameters()}
    decoder_ids = {id(p) for p in groups["decoder"]}
    assert gate_ids <= decoder_ids  # every gate param is in the decoder group


def test_fusion_unknown_rejected():
    cfg = _extra_fusion_cfg("bogus_fusion")
    with pytest.raises(ValueError, match="model.fusion"):
        build_model(cfg)


# --- Foundation encoder (second-wave Step 4) ----------------------------------


def test_build_model_foundation_rgb(monkeypatch):
    """arch='foundation' builds a ViT segmenter, returns (B,1,H,W) logits, and the
    class-prior bias flows through its .segmentation_head[0]."""
    cfg = _base_cfg(architecture="foundation", backbone="vit_base_patch16_dinov3",
                    output_bias_prior=0.01)
    model = build_model(cfg).eval()
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 64, 64)
    assert math.isclose(model.segmentation_head[0].bias.detach().item(),
                        -math.log((1.0 - 0.01) / 0.01), abs_tol=1e-5)


def test_build_model_foundation_rejects_extra_channels():
    """Foundation is RGB-only for now; declaring EXTRA is a clear error (Step 4b)."""
    cfg = _base_cfg(architecture="foundation", backbone="vit_base_patch16_dinov3")
    cfg["channels"]["extra"] = [{"name": "ndvi", "band": 0}]
    with pytest.raises(ValueError, match="RGB-only"):
        build_model(cfg)
