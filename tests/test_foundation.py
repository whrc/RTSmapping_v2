"""Forward-path tests for the foundation-encoder segmenter (models/foundation.py).

CPU-only, pretrained=False (no weight download). Validates the ViT
get_intermediate_layers → simple feature pyramid → FPN decoder → logits chain and the
compatibility hooks (.encoder for freeze/LLRD, .segmentation_head[0] for bias-init).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
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


def test_sam2_extra_channels_forward_shape():
    """SAM2/Hiera RGB+EXTRA: stem widened to 4 channels via the wrapper reach-through.

    Replaces test_sam2_rejects_extra_channels (2026-07-28). That test pinned a limitation
    that was never real — the stem is reachable at encoder.model.patch_embed — and the
    limitation cost fm_sam2_rgb the NDVI channel.
    """
    model = FoundationSegmenter(SAM2_BACKBONE, pretrained=False, in_channels=4).eval()
    x = torch.zeros(1, 4, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 256, 256)


def test_sam2_extra_channels_zero_init_is_rgb_only_at_init():
    """EXTRA stem weights zero-init ⇒ epoch-0 output ignores EXTRA (same fairness guarantee
    the ViT path gets), so a SAM2+NDVI arm starts exactly where SAM2+RGB does."""
    torch.manual_seed(0)
    model = FoundationSegmenter(SAM2_BACKBONE, pretrained=False, in_channels=4).eval()
    rgb = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y_zero = model(torch.cat([rgb, torch.zeros(1, 1, 256, 256)], dim=1))
        y_rand = model(torch.cat([rgb, torch.randn(1, 1, 256, 256)], dim=1))
    assert torch.allclose(y_zero, y_rand, atol=1e-5)


def test_sam2_features_only_wrapper_exposes_inner_model():
    """timm's features_only wrapper DOES expose `.model` → HieraDet with .blocks/.patch_embed.

    `fm_sam2_rgb` disabled LLRD and barred EXTRA channels on the stated premise that the
    wrapper hides both (see configs/fm_sam2_rgb.yaml and models/foundation.py:95-96). That
    premise is false, and it cost that run the project's single biggest lever (NDVI, +0.07
    over RGB) plus the layer-wise LR taper every successful DINOv3 arm used. Pin the reach-
    through so the two limitations stay a deliberate choice rather than a stale assumption.
    """
    model = FoundationSegmenter(SAM2_BACKBONE, pretrained=False)
    inner = getattr(model.encoder, "model", None)
    assert inner is not None, "features_only wrapper no longer exposes .model"
    assert hasattr(inner, "blocks") and len(inner.blocks) > 0, "no .blocks → LLRD infeasible"
    assert hasattr(inner, "patch_embed"), "no .patch_embed → EXTRA fusion infeasible"


# The backbone fm_sam2_rgb actually trained. Only exercised when its weights are already
# in the local HF cache, so the module's "no weight download" contract still holds.
SAM2_TRAINED_BACKBONE = "sam2_hiera_small"
_HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
_SAM2_CACHED = (_HF_HOME / "hub" / "models--timm--sam2_hiera_small.fb_r896").exists()


@pytest.mark.skipif(not _SAM2_CACHED, reason="sam2_hiera_small weights not in local HF cache")
def test_sam2_pretrained_weights_are_actually_loaded():
    """`pretrained=True` must change every encoder tensor vs a random init.

    Nothing in the repo verified this: fm_sam2_rgb.yaml sets no `encoder_init`, so the
    tensor-count check in models/segmentation.py never ran, and every other SAM2 test uses
    pretrained=False. A silent zero-tensor load would have left the suite green and looked
    exactly like the 0.5558 result we are investigating. (Audited 2026-07-28: 202/202
    tensors, 33,947,328 params — matching that run's "Backbone frozen (33947328 params)".)
    """
    pt = FoundationSegmenter(SAM2_TRAINED_BACKBONE, pretrained=True).encoder.state_dict()
    rand = FoundationSegmenter(SAM2_TRAINED_BACKBONE, pretrained=False).encoder.state_dict()
    assert pt.keys() == rand.keys()
    n_diff = sum(1 for k in pt if not torch.equal(pt[k], rand[k]))
    assert n_diff == len(pt), f"only {n_diff}/{len(pt)} encoder tensors carry pretrained values"


# --- DINOv3 satellite (SAT-493M) ViT-L (family E) ---

DINOV3_SAT_L = "vit_large_patch16_dinov3.sat493m"


def test_dinov3_sat_vitl_forward_shape():
    """Satellite DINOv3 ViT-L builds via the isotropic-ViT path (forward_intermediates) →
    (B,1,H,W). It's a plain ViT (NOT hierarchical) and exposes .blocks so LP-FT + LLRD apply
    (unlike the SAM2/Hiera path). Same code path as the web ViT-B; only the weights differ."""
    model = FoundationSegmenter(DINOV3_SAT_L, pretrained=False).eval()
    assert not model._hierarchical and hasattr(model.encoder, "blocks")
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 64, 64)
