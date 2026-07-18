"""Sample + materialize the v2.1 SSL pretraining corpus (spec: pretraining/pretraining.md).

Unlabeled 512x512 4-ch (RGB+NDVI) tiles over the S2-covered south footprint. Tile
reads reuse the inference readers verbatim (``inference.tiles.read_tile`` /
``read_ndvi_tile``, CLAUDE Rule 3) so pretraining pixels are byte-identical to what
inference/fine-tune would see. This module holds the pure sampling/exclusion/quality
logic; ``scripts/build_pretrain_corpus.py`` is the entry point that does the GCS I/O.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import STRtree, box
from shapely.geometry import shape

logger = logging.getLogger(__name__)

# Stratification grid, in EPSG:3857 metres. 2deg lat x 20deg lon at the equator;
# the exact cell size is immaterial — it only balances the spatial sample so one
# dense longitude band can't dominate.
_LAT_BAND_M = 2.0 / 360.0 * (2 * 20037508.34)
_LON_SECTOR_M = 20.0 / 360.0 * (2 * 20037508.34)


def _tile_centers(tiles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cx = (tiles["minx"].to_numpy() + tiles["maxx"].to_numpy()) / 2
    cy = (tiles["miny"].to_numpy() + tiles["maxy"].to_numpy()) / 2
    return cx, cy


def stratum_labels(tiles: pd.DataFrame) -> np.ndarray:
    """Integer stratum id per tile = (lat_band, lon_sector) of the tile centre."""
    cx, cy = _tile_centers(tiles)
    band = np.floor(cy / _LAT_BAND_M).astype(np.int64)
    sector = np.floor(cx / _LON_SECTOR_M).astype(np.int64)
    # Cantor-ish pairing into one label (both can be negative → offset).
    return (band + 1_000_000) * 10_000 + (sector + 5_000)


def _envelope(index: pd.DataFrame) -> tuple[float, float, float, float]:
    return (index["minx"].min(), index["miny"].min(),
            index["maxx"].max(), index["maxy"].max())


def _boxes(df: pd.DataFrame) -> np.ndarray:
    """Vectorized shapely box geometries from a bounds DataFrame.

    ``shapely.box`` broadcasts over numpy arrays in C — orders of magnitude faster
    than a Python ``[box(...) for r in df.itertuples()]`` at the 40M-row domain scale.
    """
    return box(df["minx"].to_numpy(), df["miny"].to_numpy(),
               df["maxx"].to_numpy(), df["maxy"].to_numpy())


def filter_to_s2_footprint(
    tiles: pd.DataFrame, s2_index: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only tiles whose bbox intersects an S2 cell (NDVI is available there).

    Two stages: a cheap vectorized envelope pre-filter (drops the pan-arctic tiles
    far outside the south S2 extent), then an exact STRtree intersection test. This
    is what defines the "south-covered" corpus domain.
    """
    ex = _envelope(s2_index)
    pre = tiles[(tiles["maxx"] > ex[0]) & (tiles["minx"] < ex[2])
                & (tiles["maxy"] > ex[1]) & (tiles["miny"] < ex[3])]
    if pre.empty:
        return pre
    tree = STRtree(_boxes(s2_index))
    left, _ = tree.query(_boxes(pre), predicate="intersects")
    keep = np.unique(left)
    return pre.iloc[keep]


def _geojson_epsg(gj: dict) -> int:
    """EPSG code from a GeoJSON ``crs`` member; GeoJSON default is 4326 if absent."""
    name = gj.get("crs", {}).get("properties", {}).get("name")
    if not name:
        return 4326
    return int(name.rsplit(":", 1)[-1].rsplit("::", 1)[-1])


def load_exclusion_polygons(
    regions_geojson: str | Path,
    split_names: list[str],
    name_key: str = "ECO_NAME",
    target_epsg: int = 3857,
) -> STRtree:
    """STRtree (in EPSG:3857) over the val/test region polygons to exclude.

    ``split_names`` are the region names listed under val/test in splits.yaml; the
    GeoJSON features are matched on ``name_key`` (subregions use ECO_NAME).

    The subregions GeoJSON is **EPSG:3413** (polar stereographic), not the 3857 the
    tile bounds use — so the polygons are reprojected to ``target_epsg`` before the
    tree is built. Skipping this makes every intersection test miss (mismatched
    coordinate spaces) and silently leaks eval-region tiles into the corpus.
    """
    with _open_text(regions_geojson) as f:
        gj = json.load(f)
    src_epsg = _geojson_epsg(gj)
    wanted = set(split_names)
    geoms = [shape(feat["geometry"]) for feat in gj["features"]
             if feat.get("properties", {}).get(name_key) in wanted]
    if src_epsg != target_epsg:
        from pyproj import Transformer
        from shapely.ops import transform
        tf = Transformer.from_crs(src_epsg, target_epsg, always_xy=True).transform
        geoms = [transform(tf, g) for g in geoms]
    logger.info("Exclusion polygons: %d regions matched of %d requested names "
                "(reprojected EPSG:%d → %d)", len(geoms), len(wanted), src_epsg, target_epsg)
    return STRtree(geoms)


def drop_excluded(tiles: pd.DataFrame, exclusion: STRtree) -> pd.DataFrame:
    """Drop tiles whose bbox intersects any val/test exclusion polygon."""
    if len(exclusion.geometries) == 0:
        return tiles
    left, _ = exclusion.query(_boxes(tiles), predicate="intersects")
    excluded = np.unique(left)
    mask = np.ones(len(tiles), dtype=bool)
    mask[excluded] = False
    return tiles.iloc[mask]


def stratified_sample(
    tiles: pd.DataFrame,
    n_target: int,
    seed: int = 42,
    oversample_mask: np.ndarray | None = None,
    oversample_factor: float = 2.0,
) -> pd.DataFrame:
    """Draw ``n_target`` tiles balanced across strata (round-robin per stratum).

    ``oversample_mask`` (bool, aligned to ``tiles``) marks tiles near the training
    label footprint; those strata get ``oversample_factor``x their fair share before
    balancing. Sampling is without replacement; if a stratum runs dry its quota
    spills to the others (round-robin naturally handles this).
    """
    rng = np.random.default_rng(seed)
    tiles = tiles.reset_index(drop=True)
    strata = stratum_labels(tiles)
    weight = np.ones(len(tiles), dtype=np.float64)
    if oversample_mask is not None:
        weight[oversample_mask] = oversample_factor

    # Group row indices by stratum, shuffled within each, ordered by ascending
    # weight so oversampled (high-weight) tiles sit at the list end and are drawn
    # first by the pop() below.
    groups: dict[int, list[int]] = {}
    for s in np.unique(strata):
        idx = np.nonzero(strata == s)[0]
        order = np.lexsort((rng.random(len(idx)), weight[idx]))
        groups[int(s)] = list(idx[order])

    if n_target >= len(tiles):
        return tiles

    chosen: list[int] = []
    keys = list(groups)
    rng.shuffle(keys)
    while len(chosen) < n_target and any(groups[k] for k in keys):
        for k in keys:
            if groups[k]:
                chosen.append(groups[k].pop())
                if len(chosen) >= n_target:
                    break
    return tiles.iloc[sorted(chosen)]


def quality_ok(
    rgb: np.ndarray, nodata: np.ndarray, ndvi: np.ndarray,
    max_nodata_frac: float = 0.5,
) -> bool:
    """Reject a tile with >``max_nodata_frac`` RGB NoData or all-NaN NDVI."""
    if nodata.mean() > max_nodata_frac:
        return False
    if not np.isfinite(ndvi).any():
        return False
    return True


def _open_text(path: str | Path):
    """Open a local or gs:// text path (mirrors data/splits._open_text)."""
    path = str(path)
    if path.startswith("gs://"):
        from google.cloud import storage  # deferred
        import io
        bucket, _, key = path[len("gs://"):].partition("/")
        blob = storage.Client().bucket(bucket).blob(key)
        return io.StringIO(blob.download_as_text())
    return open(path)
