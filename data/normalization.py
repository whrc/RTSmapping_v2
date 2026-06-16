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
    extra_modes: list[str] | None = None,
    extra_clips: list[list[float] | None] | None = None,
    extra_scales: list[float | None] | None = None,
) -> dict:
    """Assemble the normalization_stats.json schema.

    RGB block is always present; EXTRA block only if requested. EXTRA uses
    the channel names the user chose (e.g., ndvi, nir, re, sr, or anything else)
    — not a fixed registry.

    Per-EXTRA-channel normalization treatment (data/data.md §9), parallel lists
    aligned with the EXTRA channel order; all optional (default = plain z-score):
      * ``extra_modes``  — "zscore" | "fixed_scale"
      * ``extra_clips``  — [lo, hi] raw-value clip bounds, or None (zscore channels)
      * ``extra_scales`` — divisor for "fixed_scale" channels (e.g. SE_PROTO → 0.5)
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
        block: dict = {
            "channel_names": extra.channel_names,
            "mean": extra.means(),
            "std": extra.stds(),
        }
        if extra_modes is not None:
            block["mode"] = extra_modes
        if extra_clips is not None:
            block["clip"] = extra_clips
        if extra_scales is not None:
            block["scale"] = extra_scales
        out["extra"] = block
    return out


def save_stats(stats: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(stats, f, indent=2)


def _open_text(path: str | Path):
    """Open a text file, supporting both local paths and gs:// URIs."""
    p = str(path)
    if p.startswith("gs://"):
        import gcsfs
        return gcsfs.GCSFileSystem(token="google_default").open(p[5:], "r")
    return Path(p).open("r")


def load_stats(path: str | Path) -> dict:
    with _open_text(path) as f:
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


def build_norm_arrays(stats: dict, with_extra: bool) -> dict[str, np.ndarray]:
    """Per-channel normalization parameters as aligned arrays (RGB then EXTRA).

    The single source for how each channel is normalized (data/data.md §9), used by
    both training (``data/dataset.py``) and inference (``inference/tiles.py``) via
    ``apply_norm`` so the two are identical (CLAUDE Rule 3). RGB is always plain
    z-score (no clip); EXTRA honours the per-channel ``mode``/``clip``/``scale``
    recorded in the stats file (absent ⇒ plain z-score, backward-compatible).
    """
    mean, std = stats_to_arrays(stats, with_extra)
    c = len(mean)
    clip_lo = np.full(c, np.nan, dtype=np.float32)
    clip_hi = np.full(c, np.nan, dtype=np.float32)
    is_fixed = np.zeros(c, dtype=bool)
    scale = np.ones(c, dtype=np.float32)
    if with_extra and "extra" in stats:
        e = stats["extra"]
        n_rgb = len(stats["rgb"]["mean"])
        n_e = len(e["mean"])
        modes = e.get("mode", ["zscore"] * n_e)
        clips = e.get("clip", [None] * n_e)
        scales = e.get("scale", [None] * n_e)
        for i in range(n_e):
            ch = n_rgb + i
            if modes[i] == "fixed_scale":
                is_fixed[ch] = True
                scale[ch] = float(scales[i]) if scales[i] else 1.0
            elif clips[i] is not None:
                clip_lo[ch], clip_hi[ch] = float(clips[i][0]), float(clips[i][1])
    return {"mean": mean, "std": std, "clip_lo": clip_lo, "clip_hi": clip_hi,
            "is_fixed": is_fixed, "scale": scale}


def apply_norm(img: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    """Normalize a (C, H, W) raw float image per ``build_norm_arrays`` params.

    zscore channels: optional clip to [lo, hi] then (x - μ) / σ.
    fixed_scale channels: x / scale (no z-score; preserves the meaningful zero).
    """
    out = img.astype(np.float32, copy=True)
    lo, hi = params["clip_lo"], params["clip_hi"]
    clipped = np.isfinite(lo)
    for ch in np.nonzero(clipped)[0]:
        out[ch] = np.clip(out[ch], lo[ch], hi[ch])
    zs = ~params["is_fixed"]
    mean, std = params["mean"], params["std"]
    out[zs] = (out[zs] - mean[zs, None, None]) / std[zs, None, None]
    fx = params["is_fixed"]
    if fx.any():
        out[fx] = img[fx] / params["scale"][fx, None, None]
    return out


def fill_nodata_with_mean(
    rgb: np.ndarray,
    mask: np.ndarray,
    means: np.ndarray,
    channel_axis: int,
) -> np.ndarray:
    """Substitute the per-channel mean at NoData pixels, in place (§4.4 / inference §5.3).

    The single shared NoData-fill used by both training (``data/dataset.py``, HWC uint8,
    per-band zero mask for band dropout) and inference (``inference/tiles.py``, CHW float32,
    per-pixel alpha mask broadcast across channels), so both neutralise NoData identically
    (CLAUDE Rule 3): a filled pixel sits at ~0 after z-scoring instead of injecting a hard
    zero edge. For integer rasters the mean is rounded to ``rgb``'s dtype so the on-disk
    raw-value contract holds (a ≤0.5/σ residual after z-score is inherent to uint8 storage).

    Args:
        rgb: image array; channel axis given by ``channel_axis``.
        mask: boolean NoData mask, same shape as ``rgb`` (per-channel). Callers with a
            per-pixel mask broadcast it across channels first.
        means: per-channel means in raw units; first ``n_channels`` are used.
        channel_axis: which axis of ``rgb`` is the channel axis (-1 for HWC, 0 for CHW).

    Returns:
        ``rgb`` (mutated in place).
    """
    n_ch = rgb.shape[channel_axis]
    fill = np.asarray(means)[:n_ch]
    if rgb.dtype.kind in "iu":
        fill = np.rint(fill).astype(rgb.dtype)
    band = np.moveaxis(rgb, channel_axis, 0)
    band_mask = np.moveaxis(mask, channel_axis, 0)
    for c in range(n_ch):
        band[c][band_mask[c]] = fill[c]
    return rgb
