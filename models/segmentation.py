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


def build_model(cfg: dict) -> nn.Module:
    """Construct the segmentation model from a config dict.

    Reads `model.architecture`, `model.backbone`, `model.pretrained`, and
    `model.output_bias_prior`. Derives `in_channels` from `channels.extra`.

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

    encoder_weights = "imagenet" if pretrained else None

    if arch == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,  # logits (training.md §4.2)
        )
    else:
        raise ValueError(
            f"Unsupported model.architecture: {arch!r}. "
            f"Add an elif branch in build_model for SegFormer / DINOv3 "
            f"(see training.md §3.2)."
        )

    _init_output_bias(model, prior)
    logger.info(
        "Built %s(%s) with in_channels=%d, pretrained=%s",
        arch, backbone, in_channels, pretrained,
    )
    return model
