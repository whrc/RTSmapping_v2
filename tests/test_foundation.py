"""Forward-path tests for the foundation-encoder segmenter (models/foundation.py).

CPU-only, pretrained=False (no weight download). Validates the ViT
get_intermediate_layers → simple feature pyramid → FPN decoder → logits chain and the
compatibility hooks (.encoder for freeze/LLRD, .segmentation_head[0] for bias-init).
"""

from __future__ import annotations

import math

import torch

from models.foundation import FoundationSegmenter
from models.segmentation import _init_output_bias

BACKBONE = "vit_base_patch16_dinov3"


def test_foundation_forward_shape():
    """ViT encoder → (B, 1, H, W) logits at input resolution."""
    model = FoundationSegmenter(BACKBONE, pretrained=False).eval()
    x = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_foundation_taps_four_blocks_incl_deepest():
    model = FoundationSegmenter(BACKBONE, pretrained=False)
    depth = len(model.encoder.blocks)
    assert len(model.tap_indices) == 4
    assert model.tap_indices[-1] == depth - 1
    assert model.tap_indices == sorted(model.tap_indices)


def test_foundation_exposes_encoder_and_head():
    """freeze.py needs .encoder; _init_output_bias needs .segmentation_head[0].bias."""
    model = FoundationSegmenter(BACKBONE, pretrained=False)
    assert hasattr(model, "encoder")
    assert isinstance(model.segmentation_head[0], torch.nn.Conv2d)
    _init_output_bias(model, 0.01)
    assert math.isclose(model.segmentation_head[0].bias.detach().item(),
                        -math.log((1.0 - 0.01) / 0.01), abs_tol=1e-5)


def test_foundation_output_is_logits():
    model = FoundationSegmenter(BACKBONE, pretrained=False).eval()
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.min().item() < 0.0 or y.max().item() > 1.0


def test_foundation_extra_channels_forward_shape():
    """RGB+EXTRA (in_channels=4) → (B,1,H,W); patch-embed widened to 4 input channels."""
    model = FoundationSegmenter(BACKBONE, pretrained=False, in_channels=4).eval()
    conv = model.encoder.patch_embed.proj
    assert conv.in_channels == 4
    x = torch.zeros(2, 4, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_foundation_extra_channels_zero_init_is_rgb_only_at_init():
    """At init the EXTRA channels are zero-init → changing only EXTRA leaves output unchanged
    (epoch-0 == RGB-only; the fair, F1-style smart-stem start)."""
    model = FoundationSegmenter(BACKBONE, pretrained=False, in_channels=4).eval()
    assert torch.count_nonzero(model.encoder.patch_embed.proj.weight[:, 3:]) == 0
    torch.manual_seed(0)
    rgb = torch.randn(1, 3, 64, 64)
    x1 = torch.cat([rgb, torch.zeros(1, 1, 64, 64)], dim=1)
    x2 = torch.cat([rgb, torch.randn(1, 1, 64, 64)], dim=1)
    with torch.no_grad():
        y1, y2 = model(x1), model(x2)
    assert torch.allclose(y1, y2, atol=1e-5)


# --- SAM2 / Hiera hierarchical backbone (family E) ---

SAM2_BACKBONE = "sam2_hiera_tiny"


def test_sam2_hierarchical_forward_shape():
    """Hiera native {/4,/8,/16,/32} pyramid → 1×1 proj → FPN → (B,1,H,W) at input res."""
    model = FoundationSegmenter(SAM2_BACKBONE, pretrained=False).eval()
    assert model._hierarchical and len(model.proj) == 4
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 256)


def test_sam2_exposes_encoder_and_head():
    """freeze/LP-FT needs .encoder.parameters(); _init_output_bias needs .segmentation_head[0]."""
    model = FoundationSegmenter(SAM2_BACKBONE, pretrained=False)
    assert hasattr(model, "encoder") and any(True for _ in model.encoder.parameters())
    assert isinstance(model.segmentation_head[0], torch.nn.Conv2d)
    _init_output_bias(model, 0.01)
    assert math.isclose(model.segmentation_head[0].bias.detach().item(),
                        -math.log((1.0 - 0.01) / 0.01), abs_tol=1e-5)


def test_sam2_rejects_extra_channels():
    """SAM2/Hiera path is RGB-only for now (features_only hides the stem) → clear error."""
    import pytest
    with pytest.raises(NotImplementedError):
        FoundationSegmenter(SAM2_BACKBONE, pretrained=False, in_channels=4)
