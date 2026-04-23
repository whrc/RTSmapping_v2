"""Segmentation model factory for the RTS pipeline.

Public API:
    build_model(cfg) -> nn.Module

See training.md §3 for architecture choices and §4.2 for the logits-output contract.
"""

from __future__ import annotations

from models.segmentation import build_model

__all__ = ["build_model"]
