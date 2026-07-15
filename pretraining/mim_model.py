"""SimMIM-style masked autoencoder over the DINOv3-L ViT (spec: pretraining/pretraining.md §3).

Masked patches are replaced by a learnable mask token at the patch-embed output via a forward
hook, so the encoder's *own* ``forward_features`` (rope + pos-embed + blocks + norm) runs
unchanged — no reimplementation of the Eva forward. A light linear decoder predicts per-patch
pixels; loss is normalized-pixel MSE on masked patches only (MAE objective). After pretraining
only ``self.encoder`` is saved — it drops into fine-tuning via ``model.encoder_init``.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


def _inflate_patch_embed(encoder: nn.Module, in_channels: int, n_rgb: int = 3) -> None:
    """Widen the ViT patch-embed conv to ``in_channels`` (RGB copied, EXTRA zero-init).

    Mirrors ``models.foundation.FoundationSegmenter._expand_patch_embed`` exactly (kept in
    sync so the pretrained state_dict matches the fine-tune encoder's 4-ch stem).
    """
    pe = encoder.patch_embed
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


class MaskedAutoencoderViT(nn.Module):
    """DINOv3-L encoder + mask-token injection + linear pixel decoder for MAE pretraining."""

    def __init__(
        self,
        backbone: str = "vit_large_patch16_dinov3.sat493m",
        pretrained: bool = True,
        in_channels: int = 4,
        patch_px: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        if in_channels != 3:
            _inflate_patch_embed(self.encoder, in_channels)
        self.in_channels = in_channels
        self.patch_px = patch_px
        self.num_prefix = int(getattr(self.encoder, "num_prefix_tokens", 0))
        embed_dim = self.encoder.embed_dim
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = nn.Linear(embed_dim, patch_px * patch_px * in_channels)
        self._cur_flat_mask: torch.Tensor | None = None
        self.encoder.patch_embed.register_forward_hook(self._inject_mask_token)

    def _inject_mask_token(self, module, inputs, output):
        """Replace masked patch embeddings with the mask token (SimMIM masking)."""
        if self._cur_flat_mask is None:
            return output
        # Eva patch_embed emits (B, gh, gw, C); flatten the grid to (B, N, C).
        b, gh, gw, c = output.shape
        flat = output.reshape(b, gh * gw, c)
        m = self._cur_flat_mask.unsqueeze(-1)                 # (B, N, 1)
        flat = torch.where(m, self.mask_token.to(flat.dtype), flat)
        return flat.reshape(b, gh, gw, c)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, C*p*p) patch pixels, N = (H/p)*(W/p)."""
        p, c = self.patch_px, self.in_channels
        b, _, h, w = imgs.shape
        gh, gw = h // p, w // p
        x = imgs.reshape(b, c, gh, p, gw, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * p * p)
        return x

    def forward(self, image: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """Return the scalar MAE loss (normalized-pixel MSE on masked patches).

        Args:
            image: (B, C, H, W) normalized tile.
            patch_mask: (B, gh, gw) bool, True where the patch is masked/hidden.
        """
        b = image.shape[0]
        flat_mask = patch_mask.reshape(b, -1)                 # (B, N)
        self._cur_flat_mask = flat_mask
        tokens = self.encoder.forward_features(image)         # (B, prefix+N, C)
        self._cur_flat_mask = None
        patch_tokens = tokens[:, self.num_prefix:, :]         # (B, N, C)
        pred = self.decoder(patch_tokens)                     # (B, N, C*p*p)

        target = self.patchify(image)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / torch.sqrt(var + 1e-6)     # per-patch norm (MAE)

        loss = ((pred - target) ** 2).mean(dim=-1)            # (B, N)
        m = flat_mask.to(loss.dtype)
        return (loss * m).sum() / m.sum().clamp_min(1.0)
