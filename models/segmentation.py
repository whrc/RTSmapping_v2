"""UNet++/EfficientNet-B5 segmentation model builder.

The model outputs **logits** (no sigmoid). Losses in losses/ operate on logits
via F.logsigmoid for stability; sigmoid applies only at metric time and at
inference. See training.md §4.2.

The final-conv bias is initialised to -log((1-pi)/pi) so that the initial
sigmoid output matches the class prior pi. Under extreme imbalance this
prevents focal loss from being dominated by negative pixels in the first few
hundred steps (Lin et al. 2017, "Focal Loss for Dense Object Detection").

Extending to new architectures: add an `elif` branch in `build_model` for
SegFormer / DINOv3 (training.md §3.2 priority list). No factory classes.
"""

from __future__ import annotations

import logging
import math

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# smp decoder drop-ins for the Phase-5 architecture sweep (experiments.md §8.2).
# All share UnetPlusPlus's constructor signature and expose `.segmentation_head[0]`
# (a Conv2d with bias), so _init_output_bias works unchanged. DeepLabV3+ leads
# (ASPP multi-scale context). Decoder-only swaps reuse the EffB5 encoder + frozen HPs.
_SMP_DECODERS = {
    "deeplabv3plus": smp.DeepLabV3Plus,
    "fpn": smp.FPN,
    "pspnet": smp.PSPNet,
    "manet": smp.MAnet,
}


def _derive_in_channels(cfg: dict) -> int:
    """Return 3 (RGB) + number of EXTRA channels declared in config."""
    extra = cfg.get("channels", {}).get("extra", []) or []
    return 3 + len(extra)


def _init_output_bias(model: nn.Module, prior: float) -> None:
    """Set the final-conv bias to -log((1 - prior) / prior).

    The segmentation head in smp.UnetPlusPlus is a small Sequential whose
    first layer is the final Conv2d with `classes` output channels.
    """
    if not (0.0 < prior < 1.0):
        raise ValueError(f"output_bias_prior must be in (0, 1), got {prior}")
    bias_init = -math.log((1.0 - prior) / prior)
    final_conv = model.segmentation_head[0]
    if not hasattr(final_conv, "bias") or final_conv.bias is None:
        raise RuntimeError("segmentation head has no bias to initialise")
    with torch.no_grad():
        final_conv.bias.fill_(bias_init)
    logger.info("Initialised output bias to %.4f (prior=%.4f)", bias_init, prior)


def _zero_extra_stem_channels(model: nn.Module, in_channels: int, n_rgb: int = 3) -> None:
    """F1 (smart stem init): zero the encoder stem-conv weights on the EXTRA input
    channels so the model starts identical to RGB-only and *learns* to add EXTRA.

    smp's imagenet init broadcasts the averaged RGB filter onto the new channels
    (an arbitrary init); zeroing them makes epoch-0 == the RGB baseline.
    """
    stem = next(
        (m for m in model.encoder.modules()
         if isinstance(m, nn.Conv2d) and m.in_channels == in_channels),
        None,
    )
    if stem is None:
        raise RuntimeError(f"F1 stem_init: no encoder stem conv with in_channels={in_channels}")
    with torch.no_grad():
        stem.weight[:, n_rgb:].zero_()
    logger.info("F1 stem_init: zeroed %d EXTRA stem-conv input channels", in_channels - n_rgb)


class ChannelAttentionFusion(nn.Module):
    """F2 (input channel-attention): a learned per-channel gate (squeeze-excite over
    input channels) applied before the encoder, so the network can down-weight
    noisy/redundant EXTRA bands. Initialised near-identity (gate ≈ 1) so the model
    starts ≈ early fusion and learns to suppress. Delegates `.encoder` /
    `.segmentation_head` so freeze.py param-groups + output-bias init are unchanged.
    """

    def __init__(self, base: nn.Module, in_channels: int, reduction: int = 2):
        super().__init__()
        self.base = base
        hidden = max(in_channels // reduction, 2)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, 1),
            nn.Sigmoid(),
        )
        # near-identity init: constant high pre-sigmoid bias → gate ≈ 0.98 for all
        # channels regardless of input, so epoch-0 ≈ early fusion.
        nn.init.zeros_(self.gate[3].weight)
        nn.init.constant_(self.gate[3].bias, 4.0)

    @property
    def encoder(self) -> nn.Module:
        return self.base.encoder

    @property
    def segmentation_head(self) -> nn.Module:
        return self.base.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x * self.gate(x))


class _DualEncoder(nn.Module):
    """Shared scaffold for F3/F5 heavy fusion: a separate EXTRA encoder alongside the
    base (RGB) UNet++, fused per encoder stage, then the base decoder + head.

    The base is built with ``in_channels=3`` (RGB); the EXTRA stream is a second
    encoder of the same backbone with ``in_channels=n_extra``. Encoder feature stages
    1..N share channel dims across the two streams (the EfficientNet stem only differs
    at the stage-0 input passthrough), so stages 1..N fuse elementwise and stage-0 keeps
    the RGB passthrough the decoder was built for. Delegates ``.encoder`` /
    ``.segmentation_head`` so freeze.py param-groups + output-bias init are unchanged.
    Input x is the usual RGB+EXTRA channel stack (split at n_rgb).
    """

    def __init__(self, base: nn.Module, backbone: str, n_extra: int, pretrained: bool, n_rgb: int = 3):
        super().__init__()
        from segmentation_models_pytorch.encoders import get_encoder

        self.base = base
        self.n_rgb = n_rgb
        self.extra_encoder = get_encoder(
            backbone, in_channels=n_extra, depth=5,
            weights="imagenet" if pretrained else None,
        )

    @property
    def encoder(self) -> nn.Module:
        return self.base.encoder

    @property
    def segmentation_head(self) -> nn.Module:
        return self.base.segmentation_head

    def _fuse(self, fr: list, fe: list) -> list:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, extra = x[:, : self.n_rgb], x[:, self.n_rgb :]
        fr = self.base.encoder(rgb)
        fe = self.extra_encoder(extra)
        fused = self._fuse(fr, fe)
        return self.base.segmentation_head(self.base.decoder(fused))


class DualEncoderLateFusion(_DualEncoder):
    """F3: late fusion by elementwise sum of the two encoders' per-stage features."""

    def _fuse(self, fr: list, fe: list) -> list:
        return [fr[0]] + [a + b for a, b in zip(fr[1:], fe[1:])]


class _CrossModalResidual(nn.Module):
    """Per-stage residual cross-modal attention: fused = rgb + sigmoid(W·[rgb,extra]) ⊙ extra.

    A learned gate (from BOTH modalities) reweights the EXTRA features added back onto the
    RGB stream — residual so the RGB path is preserved. Near-identity init (gate bias 0 →
    ~0.5) gives EXTRA a fair but bounded initial contribution. Approximates the residual
    cross-modal attention of the JSTARS fusion design (exact §IV.C formulas not reproduced).
    """

    def __init__(self, c: int):
        super().__init__()
        hidden = max(c // 4, 4)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * c, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1), nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)  # sigmoid(0)=0.5 at init

    def forward(self, r: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return r + self.gate(torch.cat([r, e], dim=1)) * e


class CrossModalAttentionFusion(_DualEncoder):
    """F5: residual cross-modal attention fusion at encoder stages 1..N (JSTARS-style)."""

    def __init__(self, base, backbone, n_extra, pretrained, n_rgb: int = 3):
        super().__init__(base, backbone, n_extra, pretrained, n_rgb)
        # stage channel dims 1..N (skip stage-0 input passthrough)
        dims = list(self.base.encoder.out_channels)[1:]
        self.blocks = nn.ModuleList([_CrossModalResidual(c) for c in dims])

    def _fuse(self, fr: list, fe: list) -> list:
        return [fr[0]] + [blk(a, b) for blk, a, b in zip(self.blocks, fr[1:], fe[1:])]


def build_model(cfg: dict) -> nn.Module:
    """Construct the segmentation model from a config dict.

    Reads `model.architecture`, `model.backbone`, `model.pretrained`,
    `model.output_bias_prior`, and `model.fusion` (default `early`). Derives
    `in_channels` from `channels.extra`.

    `model.fusion` (the second-wave fusion axis):
      - `early` (default): RGB+EXTRA concatenated into one encoder (current behaviour).
      - `stem_init`: early fusion, but EXTRA stem-conv channels zero-init (F1).
      - `chan_attn`: a learned per-channel input gate before the encoder (F2).
      - `ensemble`: builds a normal (early) model; F4 averaging is an eval-side step.
      Dual-encoder / cross-modal fusion (F3/F5) are separate model classes, not here.

    Args:
        cfg: Parsed YAML config (see configs/baseline.yaml).

    Returns:
        nn.Module outputting logits of shape (B, 1, H, W).

    Raises:
        KeyError: Required config keys are missing.
        ValueError: Unsupported architecture or invalid bias prior.

    Notes:
        EXTRA-channel pretrained-weight behaviour: smp >= 0.3 with
        `in_channels > 3` and `encoder_weights="imagenet"` averages the RGB
        channel dim and broadcasts across new channels. This is an arbitrary
        initialisation — semantic verification (does the model use EXTRA
        signal?) happens at ablation time, not here.
    """
    arch = cfg["model"]["architecture"]
    backbone = cfg["model"]["backbone"]
    pretrained = cfg["model"].get("pretrained", True)
    prior = cfg["model"].get("output_bias_prior", 0.5)
    in_channels = _derive_in_channels(cfg)
    fusion = cfg["model"].get("fusion", "early")

    # F3/F5 heavy fusion use a dual-encoder: the base (RGB) encoder is built with 3
    # channels, the EXTRA stream is a separate encoder. All other fusions feed the full
    # RGB+EXTRA stack into one encoder.
    _dual = fusion in ("dual_encoder", "cross_modal")
    if _dual and arch != "unetplusplus":
        raise ValueError(f"fusion={fusion!r} (F3/F5) supports only arch='unetplusplus', got {arch!r}")
    if _dual and in_channels <= 3:
        raise ValueError(f"fusion={fusion!r} (F3/F5) needs EXTRA channels; in_channels={in_channels}")
    enc_in = 3 if _dual else in_channels

    encoder_weights = "imagenet" if pretrained else None

    if arch == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=enc_in,
            classes=1,
            activation=None,  # logits (training.md §4.2)
        )
    elif arch == "segformer":
        # Transformer architecture (training.md §3.2). Uses MixVisionTransformer
        # encoders (mit_b0..mit_b5), NOT the EfficientNet backbones. smp's
        # Segformer exposes the same .segmentation_head[0] Conv2d (with bias) as
        # UnetPlusPlus, so _init_output_bias below works unchanged, and the head
        # already returns (B, 1, H, W) logits at input resolution.
        model = smp.Segformer(
            encoder_name=backbone,        # expect mit_b0..mit_b5
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,  # logits (training.md §4.2)
        )
    elif arch in _SMP_DECODERS:
        # smp decoder swap on the same encoder backbone (experiments.md §8.2).
        model = _SMP_DECODERS[arch](
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,  # logits (training.md §4.2)
        )
    elif arch == "foundation":
        # Foundation ViT encoder (DINOv3/DINOv2/SAM) → simple feature pyramid → FPN decoder
        # (experiments.md §8.2 / second-wave Step 4). RGB-only for now; +EXTRA (a patch-embed
        # adapter / second stream) is Step 4b. Lazy import keeps timm-ViT cost off other paths.
        from models.foundation import FoundationSegmenter

        # RGB or RGB+EXTRA (early fusion via a widened, EXTRA-zero-init patch-embed conv).
        model = FoundationSegmenter(backbone=backbone, pretrained=pretrained, in_channels=in_channels)
    else:
        raise ValueError(
            f"Unsupported model.architecture: {arch!r}. "
            f"Supported: 'unetplusplus', 'segformer', 'foundation', {sorted(_SMP_DECODERS)}. "
            f"Add an elif branch in build_model for new architectures "
            f"(e.g. DINOv3/SAM3 foundation encoders; see training.md §3.2 / experiments.md §8.2a)."
        )

    _init_output_bias(model, prior)

    if fusion in ("early", "ensemble"):
        pass  # ensemble: train a normal model; F4 averaging happens at eval time
    elif fusion == "stem_init":
        _zero_extra_stem_channels(model, in_channels)
    elif fusion == "chan_attn":
        model = ChannelAttentionFusion(model, in_channels)
    elif fusion == "dual_encoder":   # F3
        model = DualEncoderLateFusion(model, backbone, in_channels - 3, pretrained)
    elif fusion == "cross_modal":    # F5
        model = CrossModalAttentionFusion(model, backbone, in_channels - 3, pretrained)
    else:
        raise ValueError(
            f"Unsupported model.fusion: {fusion!r}. Supported: 'early', 'stem_init', "
            f"'chan_attn', 'ensemble' (eval-side), 'dual_encoder' (F3), 'cross_modal' (F5)."
        )

    logger.info(
        "Built %s(%s) with in_channels=%d, pretrained=%s, fusion=%s",
        arch, backbone, in_channels, pretrained, fusion,
    )
    return model
