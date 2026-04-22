"""Reproducibility: seed all RNGs and set deterministic CUDNN.

Deterministic mode costs 10-20% throughput (training.md §2.3) but is non-negotiable
for the baseline per CLAUDE.md.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs; optionally force deterministic CUDNN.

    Args:
        seed: RNG seed. Default 42 per CLAUDE.md §Technical Constraints.
        deterministic: If True, set CUDNN deterministic and disable benchmark.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
