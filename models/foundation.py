"""Foundation-encoder segmentation model (second-wave Step 4 / experiments.md §8.2).

A plain ViT (DINOv3 / DINOv2 / SAM-ViT via timm) emits single-scale tokens at a constant
stride, but a segmentation decoder needs a multi-scale pyramid. We bridge with the two
field-standard pieces:

  1. DINOv2/v3 dense recipe — `forward_intermediates(indices=4 blocks, output_fmt="NCHW")`
     taps 4 evenly-spaced transformer blocks (shallow=texture … deep=semantics) as 2D maps.
     (timm's unified API; works across the VisionTransformer *and* Eva families — DINOv3 is
     an Eva model and lacks the older `get_intermediate_layers`.)
  2. Simple Feature Pyramid (ViTDet, Li et al. 2022) — resample those same-resolution maps
     to strides {4, 8, 16, 32} with transposed/strided convs.

**Hierarchical backbones** (SAM2/Hiera — `sam2_*`/`hiera_*`) skip the ViT bridge entirely:
they already emit a native {/4,/8,/16,/32} pyramid, so we build them via timm `features_only`
and 1×1-project each stage straight into the same FPN decoder (RGB-only — the features_only
wrapper hides the patch-embed stem; no LLRD since there's no `.blocks`). See `__init__`.

Then a light FPN decoder + a Conv2d seg head. The head is `self.segmentation_head` whose
[0] is the final Conv2d (bias), so models.segmentation._init_output_bias works unchanged;
`self.encoder` is the ViT/Hiera so training/freeze.py + the LLRD/linear-probe schedule plug in.

Outputs logits (B, 1, H, W). Decoder differs from UNet++ (a ViT can't cleanly reuse smp's
CNN-encoder decoder interface) — standard practice for ViT segmentation; the comparison to
EffB5 stays fair on head/bias/loss/eval.
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# strides of the synthesized pyramid, shallow→deep block taps mapped onto these.
_PYRAMID_STRIDES = (4, 8, 16, 32)


def inner_encoder(encoder: nn.Module) -> nn.Module:
    """The real backbone module, unwrapping timm's `features_only` container.

    `timm.create_model(..., features_only=True)` returns a FeatureListNet/FeatureGetterNet
    that holds the backbone at `.model`, so `.blocks` / `.patch_embed` are one level down.
    Everything that needs to reach the encoder's internals (LLRD grouping in
    training/freeze.py, stem widening below) goes through here rather than assuming a flat
    encoder. Non-wrapped encoders (plain ViT) are returned unchanged.
    """
    return getattr(encoder, "model", encoder)


class _SimpleFeaturePyramid(nn.Module):
    """4 same-resolution ViT maps (at /patch) → {/4,/8,/16,/32}, each projected to dim."""

    def __init__(self, embed_dim: int, dim: int, patch: int):
        super().__init__()
        self.patch = patch
        self.blocks = nn.ModuleList()
        for stride in _PYRAMID_STRIDES:
            # ratio of target stride to the ViT's native stride (patch). <1 → upsample.
            ratio = stride / patch
            layers: list[nn.Module] = [nn.Conv2d(embed_dim, dim, 1)]
            if ratio < 1:  # upsample by 1/ratio
                up = int(round(1 / ratio))
                layers.append(nn.Upsample(scale_factor=up, mode="bilinear", align_corners=False))
            elif ratio > 1:  # downsample by ratio
                down = int(round(ratio))
                layers.append(nn.MaxPool2d(kernel_size=down, stride=down))
            layers.append(nn.Conv2d(dim, dim, 3, padding=1))
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, maps: list[torch.Tensor]) -> list[torch.Tensor]:
        return [blk(m) for blk, m in zip(self.blocks, maps)]


class _FPNDecoder(nn.Module):
    """Top-down FPN fuse → finest (/4) feature map."""

    def __init__(self, dim: int):
        super().__init__()
        self.smooth = nn.ModuleList([nn.Conv2d(dim, dim, 3, padding=1) for _ in _PYRAMID_STRIDES])

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for i in range(len(feats) - 2, -1, -1):
            x = feats[i] + F.interpolate(x, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.smooth[i](x)
        return x  # at the finest stride (/4)


class FoundationSegmenter(nn.Module):
    """DINOv3/DINOv2/SAM-ViT encoder → simple feature pyramid → FPN decoder → logits."""

    def __init__(self, backbone: str, pretrained: bool, dim: int = 256, n_taps: int = 4,
                 in_channels: int = 3):
        super().__init__()
        # SAM2/Hiera are *hierarchical* (Hiera backbone) — they emit a native {/4,/8,/16,/32}
        # pyramid, so we skip the SimpleFeaturePyramid and just 1×1-project each native stage.
        # An isotropic ViT (DINOv3/v2) needs the tap+resample bridge below.
        self._hierarchical = backbone.startswith(("sam2", "hiera"))
        if self._hierarchical:
            self.encoder = timm.create_model(
                backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
            )
            # EXTRA early fusion works here too: the features_only wrapper keeps the real
            # module at `.model`, so the Hiera stem is reachable (see inner_encoder). The
            # original SAM2 run assumed otherwise and was locked to RGB, which cost it NDVI
            # — the project's largest single channel effect (+0.07 over RGB).
            if in_channels != 3:
                self._expand_patch_embed(in_channels)
            chs = self.encoder.feature_info.channels()  # e.g. [96, 192, 384, 768]
            self.proj = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in chs])
            self.decoder = _FPNDecoder(dim)
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(dim, 1, kernel_size=1),  # [0] = final conv (bias) for _init_output_bias
                nn.Upsample(scale_factor=_PYRAMID_STRIDES[0], mode="bilinear", align_corners=False),
            )
            return

        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        if in_channels != 3:
            self._expand_patch_embed(in_channels)
        self.patch = self._infer_patch(backbone)
        depth = len(self.encoder.blocks)
        # 4 evenly-spaced block indices (deepest last).
        self.tap_indices = [max(0, round((j + 1) * depth / n_taps) - 1) for j in range(n_taps)]
        embed_dim = self.encoder.embed_dim
        self.pyramid = _SimpleFeaturePyramid(embed_dim, dim, self.patch)
        self.decoder = _FPNDecoder(dim)
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),  # [0] = final conv (bias) for _init_output_bias
            nn.Upsample(scale_factor=_PYRAMID_STRIDES[0], mode="bilinear", align_corners=False),
        )

    def _expand_patch_embed(self, in_channels: int, n_rgb: int = 3) -> None:
        """Widen the ViT patch-embed conv to `in_channels` for RGB+EXTRA early fusion.

        Copies the pretrained RGB weights and **zero-inits the EXTRA channels** (smart
        stem init, cf. models.segmentation F1): epoch-0 output == RGB-only, then the model
        learns to use EXTRA. Fair vs EffB5+EXTRA, which also early-fuses at the stem.
        """
        pe = inner_encoder(self.encoder).patch_embed
        conv = pe.proj if isinstance(getattr(pe, "proj", None), nn.Conv2d) else \
            next(m for m in pe.modules() if isinstance(m, nn.Conv2d))
        new = nn.Conv2d(in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                        stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :n_rgb] = conv.weight
            if conv.bias is not None:
                new.bias.copy_(conv.bias)
        pe.proj = new
        logger.info("Foundation patch-embed expanded 3 → %d channels (EXTRA zero-init)", in_channels)

    @staticmethod
    def _infer_patch(backbone: str) -> int:
        for tok in backbone.split("_"):
            if tok.startswith("patch"):
                return int(tok[len("patch"):])
        return 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if self._hierarchical:
            maps = self.encoder(x)  # native {/4,/8,/16,/32} pyramid
            feats = [p(m) for p, m in zip(self.proj, maps)]
        else:
            maps = self.encoder.forward_intermediates(
                x, indices=self.tap_indices, norm=True, output_fmt="NCHW",
                intermediates_only=True,
            )
            feats = self.pyramid(list(maps))
        fused = self.decoder(feats)
        logits = self.segmentation_head(fused)
        if logits.shape[-2:] != (H, W):  # patch-grid rounding → resize to input
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits
