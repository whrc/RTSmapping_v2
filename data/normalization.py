"""Per-dataset normalization: Welford online mean/std + save/load normalization_stats.json.

Per data/data.md §5, stats are computed once over the train split, saved alongside
the model checkpoint, and applied identically at inference time.

The Welford algorithm lets us compute mean/std in a single streaming pass without
holding all tiles in memory — important when the dataset is terabytes of GeoTIFFs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass
class WelfordChannelStats:
    """Running mean/variance for a single channel via Welford's algorithm."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squared differences from running mean

    def update(self, values: np.ndarray) -> None:
        """Incorporate a flat batch of values (any shape; will be ravelled)."""
        flat = values.ravel().astype(np.float64, copy=False)
        n = flat.size
        if n == 0:
            return
        batch_mean = flat.mean()
        batch_m2 = ((flat - batch_mean) ** 2).sum()

        new_count = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / new_count
        self.m2 += batch_m2 + delta * delta * self.count * n / new_count
        self.count = new_count

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))


@dataclass
class WelfordStats:
    """Multi-channel running stats. Channel order fixed at construction time."""

    channel_names: list[str]
    per_channel: list[WelfordChannelStats] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.per_channel:
            self.per_channel = [WelfordChannelStats() for _ in self.channel_names]
        if len(self.per_channel) != len(self.channel_names):
            raise ValueError("channel_names and per_channel length mismatch")

    def update(self, array: np.ndarray) -> None:
        """Update from an array shaped (C, H, W) or (C, ...).

        Channel dimension must match len(channel_names).
        """
        if array.shape[0] != len(self.channel_names):
            raise ValueError(
                f"Expected {len(self.channel_names)} channels, got array shape {array.shape}"
            )
        for i in range(array.shape[0]):
            self.per_channel[i].update(array[i])

    def means(self) -> list[float]:
        return [c.mean for c in self.per_channel]

    def stds(self) -> list[float]:
        return [c.std for c in self.per_channel]


def build_stats_dict(
    rgb: WelfordStats,
    extra: WelfordStats | None,
    dataset_version: str,
    n_tiles_used: int,
) -> dict:
    """Assemble the normalization_stats.json schema.

    RGB block is always present; EXTRA block only if requested. EXTRA uses
    the channel names the user chose (e.g., ndvi, nir, re, sr, or anything else)
    — not a fixed registry.
    """
    out: dict = {
        "dataset_version": dataset_version,
        "computed_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_tiles_used": n_tiles_used,
        "rgb": {
            "channel_names": rgb.channel_names,
            "mean": rgb.means(),
            "std": rgb.stds(),
        },
    }
    if extra is not None and len(extra.channel_names) > 0:
        out["extra"] = {
            "channel_names": extra.channel_names,
            "mean": extra.means(),
            "std": extra.stds(),
        }
    return out


def save_stats(stats: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: str | Path) -> dict:
    with Path(path).open("r") as f:
        return json.load(f)


def stats_to_arrays(stats: dict, with_extra: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) vectors for the full channel stack (RGB + optional EXTRA).

    Order: RGB first (R, G, B), then EXTRA in the order recorded in the stats file.
    """
    mean = list(stats["rgb"]["mean"])
    std = list(stats["rgb"]["std"])
    if with_extra:
        if "extra" not in stats:
            raise KeyError("stats file has no 'extra' block but with_extra=True")
        mean.extend(stats["extra"]["mean"])
        std.extend(stats["extra"]["std"])
    return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)
