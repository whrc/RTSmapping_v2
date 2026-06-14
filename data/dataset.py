"""RTSDataset — load RGB + EXTRA + label tiles, augment, normalize, return tensors.

Key decisions:
  - EXTRA band set is declared in config (data.md §9 treats NDVI/NIR/RE/SR as examples,
    not a fixed registry). `extra_spec` is a list of {name, band} dicts.
  - Normalization stats load from JSON (data/normalization.py). Mean/std vectors
    sized for RGB(+EXTRA) applied AFTER albumentations augmentation.
  - Boundary-ignore dilation (training.md §5.5 approach 1) applied to the label
    before augmentation, via data.transforms.dilate_label_boundary.
  - GCS access: rely on rasterio's native VSI support for gs:// URIs
    (GOOGLE_APPLICATION_CREDENTIALS env var must be set) OR a gcsfuse mount
    that lets us use plain filesystem paths. Caller controls via `data_root`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from data.normalization import fill_nodata_with_mean, load_stats, stats_to_arrays
from data.transforms import dilate_label_boundary

logger = logging.getLogger(__name__)

# GCS reads over rasterio's /vsigs/ layer occasionally fail transiently (a
# truncated range read surfaces as TIFFReadDirectory / TIFFReadEncodedStrip).
# A single such failure in a DataLoader worker otherwise crashes a multi-hour
# run, so tile reads retry with exponential backoff. Genuinely corrupt tiles
# still fail all attempts and surface loudly with the tile id.
_READ_ATTEMPTS = 4
_READ_BACKOFF_S = 0.5


def _read_with_retry(read_fn: Any, *, tile_id: str, what: str) -> np.ndarray:
    """Run a tile-read callable, retrying transient GCS/VSI read errors."""
    last_exc: Exception | None = None
    for attempt in range(_READ_ATTEMPTS):
        try:
            return read_fn()
        except Exception as exc:  # rasterio: RasterioIOError / CPLE_AppDefinedError
            last_exc = exc
            if attempt < _READ_ATTEMPTS - 1:
                logger.warning("Read %s failed for tile %s (attempt %d/%d): %s",
                               what, tile_id, attempt + 1, _READ_ATTEMPTS, exc)
                time.sleep(_READ_BACKOFF_S * (2 ** attempt))
    raise RuntimeError(
        f"Failed to read {what} for tile {tile_id} after {_READ_ATTEMPTS} attempts"
    ) from last_exc


@dataclass
class ExtraChannel:
    name: str
    band: int  # 0-indexed position in the EXTRA multi-band GeoTIFF


def parse_extra_spec(extra_cfg: list[dict[str, Any]] | None) -> list[ExtraChannel]:
    """Turn channels.extra YAML block into typed ExtraChannel list. [] when disabled."""
    if not extra_cfg:
        return []
    out = []
    for entry in extra_cfg:
        if "name" not in entry or "band" not in entry:
            raise ValueError(f"Each channels.extra entry needs 'name' and 'band': {entry}")
        out.append(ExtraChannel(name=str(entry["name"]), band=int(entry["band"])))
    return out


def substitute_nodata(
    rgb: np.ndarray,
    label: np.ndarray,
    rgb_mean: np.ndarray,
    ignore_index: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    """§4.4 NoData handling for RGB tiles (training side).

    Zero is the NoData sentinel in the PlanetScope RGB tiles. Two cases:
      * **all-band-zero pixel** (true NoData — no coverage): set its label to
        `ignore_index` so loss skips it (no signal to learn from), and
      * **any zero band** (incl. band dropout, e.g. a missing blue channel):
        substitute that band with its raw per-channel mean, so after z-score the
        pixel sits at ~0 (neutral) instead of injecting a hard zero edge or
        distorting the input — while the valid bands of a partially-degraded
        tile keep carrying their signal.

    Computed pre-augmentation so the ignore label rides the same geometric
    transform as the rest of the mask. Returns new (rgb, label) arrays.

    Args:
        rgb: (H, W, 3) uint8 raw RGB.
        label: (H, W) uint8 label.
        rgb_mean: (3,) raw per-channel means (z-score stats, raw units).
        ignore_index: label value for ignored pixels.
    """
    nodata = (rgb == 0).all(axis=-1)            # (H, W) — all bands zero
    label = label.copy()
    label[nodata] = ignore_index
    # Per-band fill (band dropout): substitute each zero band with its mean via the
    # shared helper so training/inference neutralise NoData identically (Rule 3).
    rgb = fill_nodata_with_mean(rgb.copy(), rgb == 0, rgb_mean, channel_axis=-1)
    return rgb, label


class RTSDataset(Dataset):
    """Return dict: {'image': (C, H, W) float32 tensor, 'label': (H, W) int64 tensor, 'tile_id': str}.

    C = 3 (RGB) + len(extra_channels).
    """

    def __init__(
        self,
        tile_ids: list[str],
        metadata: pd.DataFrame,
        data_root: str,
        rgb_dir: str,
        extra_dir: str,
        labels_dir: str,
        extra_channels: list[ExtraChannel],
        norm_stats_path: str | None,
        transform,  # albumentations Compose
        tile_size: int = 512,
        label_ignore_index: int = 255,
        boundary_handling: str = "none",   # none | ignore (soft_labels deferred to a later iteration)
        boundary_ignore_width: int = 3,
        nodata_handling: bool = False,      # §4.4: zero→mean input + 255 label for all-band-zero
    ):
        if boundary_handling == "soft_labels":
            raise NotImplementedError(
                "boundary_handling='soft_labels' is deferred to a later iteration "
                "(training.md §5.5). Use 'none' or 'ignore' for v1.0."
            )
        if boundary_handling not in ("none", "ignore"):
            raise ValueError(
                f"boundary_handling must be 'none' or 'ignore'; got {boundary_handling!r}"
            )
        self.tile_ids = tile_ids
        self.metadata = metadata.set_index("Tile_ID")
        self.data_root = data_root.rstrip("/")
        self.rgb_dir = rgb_dir
        self.extra_dir = extra_dir
        self.labels_dir = labels_dir
        self.extra_channels = extra_channels
        self.transform = transform
        self.tile_size = tile_size
        self.label_ignore_index = label_ignore_index
        self.boundary_handling = boundary_handling
        self.boundary_ignore_width = boundary_ignore_width
        self.nodata_handling = nodata_handling

        if norm_stats_path is not None:
            stats = load_stats(norm_stats_path)
            # training.md §4.5: assert channel-name agreement before any vector
            # arithmetic. Catches the "R-stats applied to G-channel" failure
            # mode where compute_normalization_stats was re-run after the
            # config's EXTRA order changed but the consumer expects the old order.
            expected_rgb = ["R", "G", "B"]
            actual_rgb = list(stats.get("rgb", {}).get("channel_names", []))
            if actual_rgb != expected_rgb:
                raise ValueError(
                    f"normalization stats RGB channel_names {actual_rgb!r} "
                    f"does not match expected {expected_rgb!r}"
                )
            if extra_channels:
                expected_extra = [c.name for c in extra_channels]
                actual_extra = list(stats.get("extra", {}).get("channel_names", []))
                if actual_extra != expected_extra:
                    raise ValueError(
                        f"normalization stats EXTRA channel_names {actual_extra!r} "
                        f"does not match config order {expected_extra!r}"
                    )
            self.mean, self.std = stats_to_arrays(stats, with_extra=bool(extra_channels))
        else:
            # Permitted for smoke tests; real runs must supply stats.
            logger.warning("RTSDataset created without normalization stats; output will be unnormalized")
            n_channels = 3 + len(extra_channels)
            self.mean = np.zeros(n_channels, dtype=np.float32)
            self.std = np.ones(n_channels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.tile_ids)

    def _path(self, subdir: str, tile_id: str) -> str:
        return f"{self.data_root}/{subdir}/{tile_id}.tif"

    def _read_rgb(self, tile_id: str) -> np.ndarray:
        """(H, W, 3) uint8."""
        def _do() -> np.ndarray:
            with rasterio.open(self._path(self.rgb_dir, tile_id)) as src:
                return src.read(out_dtype="uint8").transpose(1, 2, 0)  # (H, W, 3)
        return _read_with_retry(_do, tile_id=tile_id, what="RGB")

    def _read_extra(self, tile_id: str) -> np.ndarray:
        """(H, W, N) float32, where N = len(self.extra_channels)."""
        bands_1idx = [c.band + 1 for c in self.extra_channels]
        def _do() -> np.ndarray:
            with rasterio.open(self._path(self.extra_dir, tile_id)) as src:
                return src.read(bands_1idx, out_dtype="float32").transpose(1, 2, 0)  # (H, W, N)
        return _read_with_retry(_do, tile_id=tile_id, what="EXTRA")

    def _read_label(self, tile_id: str) -> np.ndarray:
        """(H, W) uint8. Negative tiles have no label file; return all-zeros."""
        if not self.is_positive(tile_id):
            return np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
        def _do() -> np.ndarray:
            with rasterio.open(self._path(self.labels_dir, tile_id)) as src:
                return src.read(1, out_dtype="uint8")
        return _read_with_retry(_do, tile_id=tile_id, what="label")

    def is_positive(self, tile_id: str) -> bool:
        return bool(self.metadata.loc[tile_id, "TrainClass"] == "positive")

    def __getitem__(self, idx: int) -> dict:
        tid = self.tile_ids[idx]
        rgb = self._read_rgb(tid)                             # (H, W, 3) uint8
        label = self._read_label(tid)                         # (H, W) uint8

        if self.nodata_handling:                              # §4.4 (before boundary/aug)
            rgb, label = substitute_nodata(rgb, label, self.mean, self.label_ignore_index)

        if self.boundary_handling == "ignore":
            label = dilate_label_boundary(label, self.boundary_ignore_width,
                                          self.label_ignore_index)

        extra = self._read_extra(tid) if self.extra_channels else None

        if extra is not None:
            aug = self.transform(image=rgb, extra=extra, mask=label)
            stacked = np.concatenate([aug["image"], aug["extra"]], axis=-1)   # (H, W, C)
        else:
            aug = self.transform(image=rgb, mask=label)
            stacked = aug["image"]                                            # (H, W, 3)
        label_out = aug["mask"]

        img = stacked.astype(np.float32).transpose(2, 0, 1)                   # (C, H, W)
        img = (img - self.mean[:, None, None]) / self.std[:, None, None]

        return {
            "image": torch.from_numpy(img),
            "label": torch.from_numpy(label_out.astype(np.int64)),
            "tile_id": tid,
        }
