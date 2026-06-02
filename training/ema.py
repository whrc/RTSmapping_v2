"""Exponential moving average of model weights.

EMAModel keeps a shadow copy of the model's parameters and buffers that is
updated after each optimizer step. At validation time, the EMA copy is swapped
into the model via the `swap_in` context manager; after validation, the live
weights are restored. The final deployment checkpoint saves EMA weights only
(training.md §4.3).

EMA is constructed at unfreeze (not epoch 1) — averaging a frozen backbone is
wasted compute. The train loop controls when to build the EMAModel.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class EMAModel:
    """Track an EMA of the given model's state_dict.

    Shadow tensors live on the same device and dtype as the source params. Use
    `state_dict()` / `load_state_dict()` to persist across training resumption.

    Args:
        model: Source model. A snapshot is taken at construction.
        decay: EMA decay rate in (0, 1). Typical values 0.999-0.9999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend each floating-point tensor toward the live model weights.

        Non-float buffers (e.g. BatchNorm counts) are copied verbatim — averaging
        integer counts would be meaningless.
        """
        for k, v in model.state_dict().items():
            shadow = self.shadow[k]
            if v.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(v.detach())

    @contextmanager
    def swap_in(self, model: nn.Module) -> Iterator[None]:
        """Temporarily load EMA weights into `model` for validation.

        Live weights are restored on exit, even on exception.
        """
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = float(sd["decay"])
        self.shadow = sd["shadow"]
