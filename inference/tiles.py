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
from collections import OrderedDict

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely import STRtree, box
from torch.utils.data import Dataset

from data.normalization import apply_norm, build_norm_arrays, fill_nodata_with_mean
from inference.quad_index import QUAD_SIZE_M, WORLD_MIN

logger = logging.getLogger(__name__)

TILE_SIZE_PX = 512  # CLAUDE.md technical constraint; matches training tiles

# Band indices (1-based) in the bulk S2 composite COGs — export order B4,B3,B2,B8
# (scripts/export_s2_composites.py DEFAULT_BANDS). Red=B4, NIR=B8 → NDVI.
S2_RED_BAND = 1   # B4
S2_NIR_BAND = 4   # B8

# Transient-GCS-read retry (same rationale as data/dataset.py).
_READ_ATTEMPTS = 4
_RETRY_BASE_DELAY_S = 1.0

# A quad/cell listed in the index but ABSENT from the bucket is a real data gap
# (e.g. pdg-planet-data quad 1459-1437, whose neighbours all exist) — GDAL raises
# an open error whose message says the object "does not exist". Such a gap must
# degrade to NoData for its footprint (§5.3), NOT crash the whole run, and must
# not burn the transient-error retry budget. Distinct from a transient GCS/network
# error, which still gets retried.
_MISSING_OBJECTS: set[str] = set()


def _is_missing_object(exc: Exception) -> bool:
    """True if a rasterio error means the object does not exist (a data gap)."""
    msg = str(exc).lower()
    return "does not exist" in msg or "no such file" in msg


def _note_missing_object(path: str) -> None:
    """Log each absent object once per worker (one quad spans ~36 overlapping tiles)."""
    if path not in _MISSING_OBJECTS:
        _MISSING_OBJECTS.add(path)
        logger.warning("Object absent from bucket — treating as no-coverage (NoData) "
                       "for its footprint: %s", path)

# --- Per-worker open-dataset cache (inference.md §11.3 quad-cache) -----------
# At stride 344 each quad is re-opened by ~36 overlapping tiles; over /vsigs/
# every open is a GCS auth + COG-header round-trip that dominated throughput
# (10.5 tiles/s, GPU idle). Caching the open handle removes the reopen and lets
# GDAL's per-dataset block cache serve the overlapping windows. The cache is
# module-level so each DataLoader worker process gets its own; handles never
# cross the fork because the parent process never reads tiles.
_OPEN_CACHE_SIZE = 16  # quads/cells kept open per worker; spatial tile ordering
#                        keeps the working set well under this.


class _OpenDatasetCache:
    """LRU of open rasterio datasets keyed by path, evictable on read error."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._cache: "OrderedDict[str, rasterio.DatasetReader]" = OrderedDict()

    def get(self, path: str) -> rasterio.DatasetReader:
        ds = self._cache.get(path)
        if ds is not None:
            self._cache.move_to_end(path)
            return ds
        ds = self._open_with_retry(path)
        self._cache[path] = ds
        if len(self._cache) > self.maxsize:
            _, evicted = self._cache.popitem(last=False)
            evicted.close()
        return ds

    def evict(self, path: str) -> None:
        """Drop (and close) a handle so a possibly-stale one is reopened."""
        ds = self._cache.pop(path, None)
        if ds is not None:
            ds.close()

    def clear(self) -> None:
        """Close and drop every cached handle (test isolation / shutdown)."""
        for ds in self._cache.values():
            ds.close()
        self._cache.clear()

    @staticmethod
    def _open_with_retry(path: str) -> rasterio.DatasetReader:
        last_exc: Exception | None = None
        for attempt in range(_READ_ATTEMPTS):
            try:
                return rasterio.open(path)
            except rasterio.errors.RasterioIOError as exc:
                if _is_missing_object(exc):
                    raise  # real gap: don't spend the retry budget; the caller skips it
                last_exc = exc
                delay = _RETRY_BASE_DELAY_S * 2 ** attempt
                logger.warning("Open failed (%s) attempt %d/%d: %s; retrying in %.0fs",
                               path, attempt + 1, _READ_ATTEMPTS, exc, delay)
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]


_DATASET_CACHE = _OpenDatasetCache(_OPEN_CACHE_SIZE)


def _read_with_cache(path: str, read_fn):
    """Run ``read_fn(dataset)`` on a cached open dataset, retrying transient GCS
    errors (evicting the handle between attempts so a stale one is reopened)."""
    last_exc: Exception | None = None
    for attempt in range(_READ_ATTEMPTS):
        try:
            return read_fn(_DATASET_CACHE.get(path))
        except rasterio.errors.RasterioIOError as exc:
            if _is_missing_object(exc):
                raise  # real gap: surface immediately so the caller can skip it
            last_exc = exc
            _DATASET_CACHE.evict(path)
            delay = _RETRY_BASE_DELAY_S * 2 ** attempt
            logger.warning("Read failed (%s) attempt %d/%d: %s; retrying in %.0fs",
                           path, attempt + 1, _READ_ATTEMPTS, exc, delay)
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _read_window_with_retry(
    path: str,
    bounds: tuple[float, float, float, float],
    out_size: int | None = None,
) -> np.ndarray:
    """Read all bands of `path` within `bounds`, boundless with 0-fill, retried.

    With `out_size`, the window is decimated/resampled to (out_size, out_size):
    RGB bands bilinear, alpha band (if present) nearest — so the NoData mask
    stays crisp instead of blending validity at coverage edges. The dataset is
    served from the per-worker open-handle cache (§11.3).
    """
    def _read(src: rasterio.DatasetReader) -> np.ndarray:
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

    return _read_with_cache(path, _read)


class _BBoxIndex:
    """STRtree over row bounding boxes for O(log N) tile→row lookup.

    Replaces the O(N) per-tile boolean mask over the full quad/S2 index
    (inference.md §11.3 spatial hit-test) — at 41.57M tiles × ~309k quads the
    linear scan is itself a real cost. The tree is built lazily on first query
    so each DataLoader worker builds its own (read-only GEOS state never crosses
    the fork). Candidates are re-filtered with the exact strict-inequality
    overlap test, in original row order, so the result is identical to the mask.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        self._tree: STRtree | None = None

    def _ensure_tree(self) -> None:
        if self._tree is None:
            self._tree = STRtree(
                [box(r.minx, r.miny, r.maxx, r.maxy) for r in self.df.itertuples()])

    def hits(self, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
        self._ensure_tree()
        minx, miny, maxx, maxy = bbox
        cand = self._tree.query(box(minx, miny, maxx, maxy))
        if len(cand) == 0:
            return self.df.iloc[:0]
        sub = self.df.iloc[sorted(cand)]
        return sub[(sub["minx"] < maxx) & (sub["maxx"] > minx)
                   & (sub["miny"] < maxy) & (sub["maxy"] > miny)]


def _spatial_sort(tiles: pd.DataFrame) -> pd.DataFrame:
    """Order tiles so spatially-adjacent ones are processed consecutively.

    Tiles overlapping the same quad then fall in the same batch and hit the
    per-worker open-handle cache (§11.3) instead of each re-opening the quad.
    Sort is by Planet quad grid cell (row then column), then position within the
    cell; output is unaffected — each tile is written independently by tile_id.
    """
    qy = ((tiles["miny"] - WORLD_MIN) // QUAD_SIZE_M).astype("int64")
    qx = ((tiles["minx"] - WORLD_MIN) // QUAD_SIZE_M).astype("int64")
    order = np.lexsort((tiles["minx"].to_numpy(), tiles["miny"].to_numpy(),
                        qx.to_numpy(), qy.to_numpy()))
    return tiles.iloc[order]


def read_tile(
    bbox: tuple[float, float, float, float],
    quad_index: pd.DataFrame,
    tile_size_px: int = TILE_SIZE_PX,
    scale: float = 1.0,
    hits: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one inference tile by mosaicking the intersecting quads.

    Args:
        bbox: (minx, miny, maxx, maxy) in EPSG:3857.
        quad_index: DataFrame from inference.quad_index (bounds + gcs_path).
        tile_size_px: output tile edge in pixels.
        scale: inference scale per inference.md §6.2 — at scale s the bbox
            covers tile_size_px/s native pixels and the read is decimated to
            tile_size_px (e.g. s=0.5 → 2× GSD, 4× ground area, same 512²).
        hits: pre-filtered intersecting quads (``InferenceTileDataset`` supplies
            these from a spatial index, §11.3); when None the intersecting quads
            are found by scanning ``quad_index`` directly.

    Returns:
        (rgb, nodata_mask): rgb float32 (3, H, W) raw values [0, 255];
        nodata_mask bool (H, W), True where no valid imagery exists.
    """
    minx, miny, maxx, maxy = bbox
    if hits is None:
        hits = quad_index[(quad_index["minx"] < maxx) & (quad_index["maxx"] > minx)
                          & (quad_index["miny"] < maxy) & (quad_index["maxy"] > miny)]

    rgb = np.zeros((3, tile_size_px, tile_size_px), dtype=np.float32)
    valid = np.zeros((tile_size_px, tile_size_px), dtype=bool)
    res_x = (maxx - minx) / tile_size_px
    res_y = (maxy - miny) / tile_size_px

    for _, quad in hits.iterrows():
        try:
            data = _read_window_with_retry(quad["gcs_path"], bbox,
                                           out_size=None if scale == 1.0 else tile_size_px)
        except rasterio.errors.RasterioIOError as exc:
            if _is_missing_object(exc):  # §5.3: absent quad → no coverage here (stays NoData)
                _note_missing_object(quad["gcs_path"])
                continue
            raise
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


def read_ndvi_tile(
    bbox: tuple[float, float, float, float],
    s2_index: pd.DataFrame,
    tile_size_px: int = TILE_SIZE_PX,
    hits: pd.DataFrame | None = None,
) -> np.ndarray:
    """Window NDVI from the bulk S2 composites onto one inference tile.

    Mirrors ``read_tile`` for the EXTRA=NDVI channel: mosaics every intersecting
    S2 composite cell, computing ``NDVI = (B8 - B4) / (B8 + B4)`` — the same
    formula training derives server-side from the same ``s2_sr_composite`` recipe
    (data/extra_channels.s2_image; NDVI is scale-invariant so the /10000
    reflectance cancels — CLAUDE Rule 3). The 10 m composite is resampled
    (bilinear) onto the tile's projected grid.

    No-coverage pixels (outside every cell, or cloud/edge gaps the export left as
    0) yield NaN — the NoData contract honoured downstream by ``apply_norm``
    (non-finite → 0), matching training's EXTRA handling.

    Args:
        bbox: (minx, miny, maxx, maxy) in EPSG:3857.
        s2_index: DataFrame from inference.s2_index (bounds + gcs_path).
        tile_size_px: output tile edge in pixels.

    Returns:
        ndvi float32 (H, W); NaN where no S2 coverage / invalid.
    """
    from rasterio.enums import Resampling

    minx, miny, maxx, maxy = bbox
    if hits is None:
        hits = s2_index[(s2_index["minx"] < maxx) & (s2_index["maxx"] > minx)
                        & (s2_index["miny"] < maxy) & (s2_index["maxy"] > miny)]

    def _read(src: rasterio.DatasetReader) -> np.ndarray:
        # Read red (B4) + NIR (B8), resampled to the tile grid; 0-fill outside.
        window = from_bounds(*bbox, transform=src.transform)
        return src.read(
            indexes=[S2_RED_BAND, S2_NIR_BAND], window=window,
            boundless=True, fill_value=0,
            out_shape=(2, tile_size_px, tile_size_px),
            resampling=Resampling.bilinear).astype(np.float32)

    ndvi = np.full((tile_size_px, tile_size_px), np.nan, dtype=np.float32)
    for _, cell in hits.iterrows():
        try:
            bands = _read_with_cache(cell["gcs_path"], _read)
        except rasterio.errors.RasterioIOError as exc:
            if _is_missing_object(exc):  # absent S2 cell → no NDVI coverage (stays NaN)
                _note_missing_object(cell["gcs_path"])
                continue
            raise
        red, nir = bands[0], bands[1]
        denom = nir + red
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_ndvi = np.where(denom > 0, (nir - red) / denom, np.nan).astype(np.float32)
        # First-valid-wins mosaic: fill only pixels still uncovered.
        take = np.isnan(ndvi) & np.isfinite(cell_ndvi)
        ndvi[take] = cell_ndvi[take]

    return ndvi


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
        stats: dict,
        tile_size_px: int = TILE_SIZE_PX,
        scale: float = 1.0,
        s2_index: pd.DataFrame | None = None,
        extra_bands: list[dict] | None = None,
        scales: list[float] | None = None,
    ) -> None:
        """tile_list needs columns: tile_id, minx, miny, maxx, maxy.

        ``stats`` is the deployment ``normalization_stats.json`` dict; normalization
        runs through the shared ``apply_norm`` (CLAUDE Rule 3) so RGB(+EXTRA) z-score
        / clip / NoData-neutralization match training exactly.

        EXTRA=NDVI (the locked v2 channel) is sourced on the fly from the bulk S2
        composites: pass ``s2_index`` (inference.s2_index) + ``extra_bands`` (the
        deployment ``model_config.channels.extra`` list). RGB-only when both are None.

        ``scales``: when None (default), the dataset is single-scale (``scale``) and
        each item is one image + NoData mask — the deploy path. When a list is given
        (e.g. ``[1.0, 0.5]``, inference.md §6.3), each item carries a per-scale image
        + valid mask so ``inference.runner`` can fuse them (§7.3); scale s<1 reads the
        tile's bbox expanded 1/s× (2× ground at 0.5×) decimated to ``tile_size_px``.
        """
        required = {"tile_id", "minx", "miny", "maxx", "maxy"}
        missing = required - set(tile_list.columns)
        if missing:
            raise ValueError(f"tile list missing columns {sorted(missing)}")
        self.scales = scales
        self.with_extra = bool(extra_bands)
        if self.with_extra:
            names = [c["name"] for c in extra_bands]
            if names != ["ndvi"]:
                raise NotImplementedError(
                    f"inference EXTRA reader supports ndvi only, got {names}")
            if s2_index is None:
                raise ValueError("extra_bands=[ndvi] requires an s2_index to window NDVI")
            if scales is None and scale != 1.0:
                raise NotImplementedError("NDVI EXTRA reader supports scale=1.0 only")
        # Spatial-sort so adjacent tiles share quads within a batch → the
        # per-worker open-handle cache hits instead of re-opening (§11.3).
        self.tiles = _spatial_sort(tile_list).reset_index(drop=True)
        self.quad_index = quad_index
        self.s2_index = s2_index
        self._quad_bbox_index = _BBoxIndex(quad_index)
        self._s2_bbox_index = _BBoxIndex(s2_index) if s2_index is not None else None
        self.norm_params = build_norm_arrays(stats, with_extra=self.with_extra)
        self.rgb_mean = self.norm_params["mean"][:3]
        self.tile_size_px = tile_size_px
        self.scale = scale

    def __len__(self) -> int:
        return len(self.tiles)

    def _expand_bbox(self, bbox: tuple, scale: float) -> tuple:
        """Bbox for a scale-s read: expanded 1/s× about its centre (§6.3 context)."""
        minx, miny, maxx, maxy = bbox
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        hx, hy = (maxx - minx) / 2 / scale, (maxy - miny) / 2 / scale
        return (cx - hx, cy - hy, cx + hx, cy + hy)

    def _read_and_norm(self, bbox: tuple, scale: float) -> tuple[np.ndarray, np.ndarray]:
        """Read + normalize one tile at ``scale``; returns (image (C,H,W), valid (H,W)).

        For scale<1 the read covers the 1/s×-expanded bbox decimated to tile_size_px
        (2× ground at 0.5×), so the image is the model input on the context-expanded
        grid; the returned valid mask is on that same grid (the runner center-crops it
        back to the 1× footprint when fusing).
        """
        read_bbox = bbox if scale == 1.0 else self._expand_bbox(bbox, scale)
        rgb, nodata = read_tile(read_bbox, self.quad_index, self.tile_size_px,
                                scale=scale, hits=self._quad_bbox_index.hits(read_bbox))
        valid = ~nodata
        n_ch = 4 if self.with_extra else 3
        if nodata.all():
            return np.zeros((n_ch, self.tile_size_px, self.tile_size_px), np.float32), valid
        rgb = fill_nodata_with_mean(rgb, np.broadcast_to(nodata, rgb.shape),
                                    self.rgb_mean, channel_axis=0)
        if self.with_extra:
            ndvi = read_ndvi_tile(read_bbox, self.s2_index, self.tile_size_px,
                                  hits=self._s2_bbox_index.hits(read_bbox))
            stack = np.concatenate([rgb, ndvi[None]], axis=0)
        else:
            stack = rgb
        return apply_norm(stack, self.norm_params), valid

    def _getitem_multiscale(self, i: int) -> dict:
        """Per-scale image + valid mask for §7.3 fusion (self.scales set)."""
        row = self.tiles.iloc[i]
        bbox = (row["minx"], row["miny"], row["maxx"], row["maxy"])
        images, valid = {}, {}
        for s in self.scales:
            img, val = self._read_and_norm(bbox, s)
            images[s], valid[s] = img, val
        return {
            "tile_id": row["tile_id"],
            "images": images,
            "valid": valid,
            "all_nodata": bool(not valid[1.0].any()),  # 1× footprint fully NoData
            "bounds": np.array(bbox, dtype=np.float64),
        }

    def __getitem__(self, i: int) -> dict:
        if self.scales is not None:
            return self._getitem_multiscale(i)
        row = self.tiles.iloc[i]
        bbox = (row["minx"], row["miny"], row["maxx"], row["maxy"])
        rgb, nodata = read_tile(bbox, self.quad_index, self.tile_size_px,
                                scale=self.scale,
                                hits=self._quad_bbox_index.hits(bbox))
        all_nodata = bool(nodata.all())
        if all_nodata:
            # Discarded by the inference loop (§5.3); emit a correctly-shaped zero
            # tensor so the batch collate (np.stack) doesn't trip on a 3-vs-4
            # channel mismatch when EXTRA=NDVI is stacked on the kept tiles.
            n_ch = 4 if self.with_extra else 3
            image = np.zeros((n_ch, self.tile_size_px, self.tile_size_px), dtype=np.float32)
        else:
            # Mean-substitute RGB NoData before z-scoring via the shared helper so
            # training and inference neutralise NoData identically (Rule 3,
            # training.md §4.4); those pixels are masked to -1.0 afterwards (§5.3).
            rgb = fill_nodata_with_mean(rgb, np.broadcast_to(nodata, rgb.shape),
                                        self.rgb_mean, channel_axis=0)
            if self.with_extra:
                # NDVI from the S2 composites; no-coverage stays NaN → apply_norm
                # neutralizes to 0 (the channel mean), exactly as in training.
                ndvi = read_ndvi_tile(bbox, self.s2_index, self.tile_size_px,
                                      hits=self._s2_bbox_index.hits(bbox))
                stack = np.concatenate([rgb, ndvi[None]], axis=0)
            else:
                stack = rgb
            image = apply_norm(stack, self.norm_params)
        return {
            "tile_id": row["tile_id"],
            "image": image,
            "nodata_mask": nodata,
            "all_nodata": all_nodata,
            "bounds": np.array(bbox, dtype=np.float64),
        }
