"""Windowed 512x512 tile reads from basemap quads + training normalization.

Tiles are projected-grid windows (inference.md §4.1) that may straddle quad
boundaries; read_tile mosaics every intersecting quad into one array. NoData
follows inference.md §5.3: the quad alpha band marks NoData (alpha == 0), as do
pixels not covered by any indexed quad.

Normalization reuses data/normalization.py stats (CLAUDE.md Rule 3): NoData
pixels get the per-channel training mean *before* z-scoring (matching training,
training.md §4.4), and the caller masks them out of the prediction afterwards.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

TILE_SIZE_PX = 512  # CLAUDE.md technical constraint; matches training tiles

# Transient-GCS-read retry (same rationale as data/dataset.py).
_READ_ATTEMPTS = 4
_RETRY_BASE_DELAY_S = 1.0


def _read_window_with_retry(
    path: str,
    bounds: tuple[float, float, float, float],
    out_size: int | None = None,
) -> np.ndarray:
    """Read all bands of `path` within `bounds`, boundless with 0-fill, retried.

    With `out_size`, the window is decimated/resampled to (out_size, out_size):
    RGB bands bilinear, alpha band (if present) nearest — so the NoData mask
    stays crisp instead of blending validity at coverage edges.
    """
    last_exc: Exception | None = None
    for attempt in range(_READ_ATTEMPTS):
        try:
            with rasterio.open(path) as src:
                window = from_bounds(*bounds, transform=src.transform)
                if out_size is None:
                    return src.read(window=window, boundless=True, fill_value=0)
                from rasterio.enums import Resampling
                rgb = src.read(indexes=list(range(1, min(src.count, 3) + 1)),
                               window=window, boundless=True, fill_value=0,
                               out_shape=(min(src.count, 3), out_size, out_size),
                               resampling=Resampling.bilinear)
                if src.count >= 4:
                    alpha = src.read(indexes=[4], window=window, boundless=True,
                                     fill_value=0, out_shape=(1, out_size, out_size),
                                     resampling=Resampling.nearest)
                    return np.concatenate([rgb, alpha], axis=0)
                return rgb
        except rasterio.errors.RasterioIOError as exc:
            last_exc = exc
            delay = _RETRY_BASE_DELAY_S * 2 ** attempt
            logger.warning("Read failed (%s) attempt %d/%d: %s; retrying in %.0fs",
                           path, attempt + 1, _READ_ATTEMPTS, exc, delay)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def read_tile(
    bbox: tuple[float, float, float, float],
    quad_index: pd.DataFrame,
    tile_size_px: int = TILE_SIZE_PX,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one inference tile by mosaicking the intersecting quads.

    Args:
        bbox: (minx, miny, maxx, maxy) in EPSG:3857.
        quad_index: DataFrame from inference.quad_index (bounds + gcs_path).
        tile_size_px: output tile edge in pixels.
        scale: inference scale per inference.md §6.2 — at scale s the bbox
            covers tile_size_px/s native pixels and the read is decimated to
            tile_size_px (e.g. s=0.5 → 2× GSD, 4× ground area, same 512²).

    Returns:
        (rgb, nodata_mask): rgb float32 (3, H, W) raw values [0, 255];
        nodata_mask bool (H, W), True where no valid imagery exists.
    """
    minx, miny, maxx, maxy = bbox
    hits = quad_index[(quad_index["minx"] < maxx) & (quad_index["maxx"] > minx)
                      & (quad_index["miny"] < maxy) & (quad_index["maxy"] > miny)]

    rgb = np.zeros((3, tile_size_px, tile_size_px), dtype=np.float32)
    valid = np.zeros((tile_size_px, tile_size_px), dtype=bool)
    res_x = (maxx - minx) / tile_size_px
    res_y = (maxy - miny) / tile_size_px

    for _, quad in hits.iterrows():
        data = _read_window_with_retry(quad["gcs_path"], bbox,
                                       out_size=None if scale == 1.0 else tile_size_px)
        if data.shape[1] != tile_size_px or data.shape[2] != tile_size_px:
            raise ValueError(
                f"Quad {quad['quad_id']} window is {data.shape[1:]} for a "
                f"{tile_size_px}px tile — resolution mismatch with the tile grid")
        alpha = data[3] > 0 if data.shape[0] >= 4 else np.ones(data.shape[1:], bool)
        # Restrict to the quad's own extent: the boundless read 0-fills outside,
        # which is indistinguishable from alpha=0 — both stay invalid.
        col0 = int(round((quad["minx"] - minx) / res_x))
        row0 = int(round((maxy - quad["maxy"]) / res_y))
        cover = np.zeros_like(alpha)
        r0, r1 = max(row0, 0), min(row0 + int(round((quad["maxy"] - quad["miny"]) / res_y)), tile_size_px)
        c0, c1 = max(col0, 0), min(col0 + int(round((quad["maxx"] - quad["minx"]) / res_x)), tile_size_px)
        cover[r0:r1, c0:c1] = True
        ok = alpha & cover & ~valid
        rgb[:, ok] = data[:3, ok].astype(np.float32)
        valid |= ok

    return rgb, ~valid


class InferenceTileDataset(Dataset):
    """Tile-list dataset for batched inference (inference.md §8.1).

    Yields dicts with normalized image tensors and the NoData mask; tiles that
    are entirely NoData are flagged (`all_nodata`) so the inference loop can
    skip + manifest-log them (§5.3) without crashing the batch.
    """

    def __init__(
        self,
        tile_list: pd.DataFrame,
        quad_index: pd.DataFrame,
        mean: np.ndarray,
        std: np.ndarray,
        tile_size_px: int = TILE_SIZE_PX,
        scale: float = 1.0,
    ) -> None:
        """tile_list needs columns: tile_id, minx, miny, maxx, maxy."""
        required = {"tile_id", "minx", "miny", "maxx", "maxy"}
        missing = required - set(tile_list.columns)
        if missing:
            raise ValueError(f"tile list missing columns {sorted(missing)}")
        self.tiles = tile_list.reset_index(drop=True)
        self.quad_index = quad_index
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.tile_size_px = tile_size_px
        self.scale = scale

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i: int) -> dict:
        row = self.tiles.iloc[i]
        bbox = (row["minx"], row["miny"], row["maxx"], row["maxy"])
        rgb, nodata = read_tile(bbox, self.quad_index, self.tile_size_px,
                                scale=self.scale)
        all_nodata = bool(nodata.all())
        if not all_nodata:
            # Mean-substitute NoData before z-scoring (training.md §4.4 parity);
            # those pixels are masked to -1.0 in the output afterwards (§5.3).
            rgb[:, nodata] = self.mean[:, None]
            rgb = (rgb - self.mean[:, None, None]) / self.std[:, None, None]
        return {
            "tile_id": row["tile_id"],
            "image": rgb,
            "nodata_mask": nodata,
            "all_nodata": all_nodata,
            "bounds": np.array(bbox, dtype=np.float64),
        }
