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
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

from data.normalization import load_stats, stats_to_arrays
from data.transforms import dilate_label_boundary

logger = logging.getLogger(__name__)


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
        boundary_handling: str = "none",   # none | ignore
        boundary_ignore_width: int = 3,
    ):
        self.tile_ids = tile_ids
        self.metadata = metadata.set_index("Tile_id")
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

        if norm_stats_path is not None:
            stats = load_stats(norm_stats_path)
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
        with rasterio.open(self._path(self.rgb_dir, tile_id)) as src:
            arr = src.read(out_dtype="uint8")  # (3, H, W)
        return arr.transpose(1, 2, 0)

    def _read_extra(self, tile_id: str) -> np.ndarray:
        """(H, W, N) float32, where N = len(self.extra_channels)."""
        bands_1idx = [c.band + 1 for c in self.extra_channels]
        with rasterio.open(self._path(self.extra_dir, tile_id)) as src:
            arr = src.read(bands_1idx, out_dtype="float32")  # (N, H, W)
        return arr.transpose(1, 2, 0)

    def _read_label(self, tile_id: str) -> np.ndarray:
        """(H, W) uint8."""
        with rasterio.open(self._path(self.labels_dir, tile_id)) as src:
            return src.read(1, out_dtype="uint8")

    def is_positive(self, tile_id: str) -> bool:
        return bool(self.metadata.loc[tile_id, "TrainClass"] == "Positive")

    def __getitem__(self, idx: int) -> dict:
        tid = self.tile_ids[idx]
        rgb = self._read_rgb(tid)                             # (H, W, 3) uint8
        label = self._read_label(tid)                         # (H, W) uint8

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
