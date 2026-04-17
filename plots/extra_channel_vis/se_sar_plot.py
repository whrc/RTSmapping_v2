"""
SE feasibility and extended SAR channel visualization for RTS segmentation.

Produces two figures per polygon (7 polygons = 14 figures total):
  1. SE Feasibility (4x3 grid): per-tile PCA, global PCA, prototype cosine
     similarity of AlphaEarth Satellite Embeddings.
  2. Extended SAR (3x3 grid): Sentinel-1 VV+VH channels and derived indices.

GEE products:
  - GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL  (64 bands, 10 m, 2024)
  - COPERNICUS/S1_GRD                      (VV+VH, 10 m, 2024 Jul-Sep)

GCS paths:
  - gs://abrupt_thaw/RTS_MODEL_V2/DATA/labels/{tile_id}.tif

Outputs:
  - plots/extra_channel_vis/se_derived/feasibility/oid{OID}_se_feasibility.png
  - plots/extra_channel_vis/sentinel_derived/sar_extended/oid{OID}_sar_extended.png

Environment: conda env rts_dataset
GEE auth: must run `earthengine authenticate` first
GCS auth: service account or ADC configured for gs://abrupt_thaw/

Expected runtime: ~20-40 min cold (GEE + GCS); <2 min cached.
"""

import argparse
import io
import logging
import math
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from google.cloud import storage
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add parent to path so we can import from extra_channel_plot
sys.path.insert(0, str(Path(__file__).parent))
from extra_channel_plot import (  # noqa: E402
    SCRIPT_DIR,
    TILE_SIZE,
    add_polygon_overlay,
    fetch_pixels,
    initialize_gee,
    load_mapping,
    load_planetscope_rgb,
    make_gee_grid,
    make_tile_bbox,
    polygon_to_pixel_coords,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = SCRIPT_DIR / ".cache"

SE_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
SE_YEAR = 2024
SE_N_BANDS = 64

GLOBAL_PCA_N_PIXELS = 10_000
PROTOTYPE_MAX_POINTS = 50_000  # statistically sufficient for mean prototype
PROTOTYPE_CHUNK_SIZE = 5_000  # stay well under GEE 10 MB payload limit
SEED = 42

GCS_BUCKET = "abrupt_thaw"
GCS_LABELS_PREFIX = "RTS_MODEL_V2/DATA/labels"

# Will be set by parse_args()
USE_CACHE = True

# Earth radius for EPSG:3857 <-> EPSG:4326 conversion
_R = 6378137.0


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def ensure_cache_dir() -> Path:
    """Create .cache/ directory if needed. Return its path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def cache_path(key: str, ext: str = ".npz") -> Path:
    """Return cache file path for a given key."""
    return CACHE_DIR / f"{key}{ext}"


def load_cache(key: str, ext: str = ".npz") -> dict | None:
    """Load cached numpy arrays. Returns None if --no-cache or missing."""
    if not USE_CACHE:
        return None
    p = cache_path(key, ext)
    if not p.exists():
        return None
    logger.info("  Cache hit: %s", p.name)
    return dict(np.load(p, allow_pickle=True))


def save_cache(key: str, ext: str = ".npz", **arrays) -> None:
    """Save numpy arrays to cache."""
    p = cache_path(key, ext)
    np.savez_compressed(p, **arrays)
    logger.info("  Cached: %s", p.name)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def epsg3857_to_4326(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert EPSG:3857 (x, y) arrays to EPSG:4326 (lon, lat)."""
    lon = x * 180.0 / (math.pi * _R)
    lat = (2.0 * np.arctan(np.exp(y / _R)) - math.pi / 2.0) * 180.0 / math.pi
    return lon, lat


# ---------------------------------------------------------------------------
# SE image
# ---------------------------------------------------------------------------

def get_se_collection(year: int = SE_YEAR) -> ee.ImageCollection:
    """Get the SE annual ImageCollection for the given year.

    The collection is tiled (one image per UTM zone). Callers must
    filterBounds() to their AOI and mosaic() before fetching pixels.
    GEE default resampling is nearest-neighbour, which preserves 10 m
    semantic boundaries.
    """
    return (
        ee.ImageCollection(SE_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    )


def get_se_image_for_region(
    se_col: ee.ImageCollection, region: ee.Geometry,
) -> ee.Image:
    """Mosaic SE tiles covering a region. Returns a single ee.Image."""
    return se_col.filterBounds(region).mosaic().toFloat()


def fetch_se_tile(
    se_col: ee.ImageCollection, bbox: ee.Geometry,
    grid: dict, tile_name: str,
) -> np.ndarray:
    """Fetch 64-band SE for a tile. Returns (64, 512, 512) float32. Cached.

    The full 64-band tile (67 MB) exceeds GEE's 50 MB computePixels limit,
    so we split into two 32-band requests.
    """
    key = f"se_tile_{tile_name}_{SE_YEAR}"
    cached = load_cache(key)
    if cached is not None:
        return cached["se_data"]

    logger.info("  Fetching SE tile from GEE (2 x 32 bands)...")
    se_image = get_se_image_for_region(se_col, bbox)
    band_names = se_image.bandNames().getInfo()
    band_names_sorted = sorted(band_names)

    # Split into two halves to stay under 50 MB limit
    mid = len(band_names_sorted) // 2
    arrays = []
    for chunk_idx, chunk_bands in enumerate(
        [band_names_sorted[:mid], band_names_sorted[mid:]],
    ):
        sub_image = se_image.select(chunk_bands)
        data = fetch_pixels(sub_image, grid, f"SE {tile_name} part {chunk_idx+1}")
        arrays.extend(data[b] for b in chunk_bands)

    se_data = np.stack(arrays, axis=0).astype(np.float32)
    save_cache(key, se_data=se_data)
    return se_data


# ---------------------------------------------------------------------------
# Percentile stretch helper
# ---------------------------------------------------------------------------

def percentile_stretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Stretch array values to [0, 1] using lo-hi percentile range."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    vmin = np.percentile(valid, lo)
    vmax = np.percentile(valid, hi)
    if vmax - vmin < 1e-12:
        return np.zeros_like(arr)
    return np.clip((arr - vmin) / (vmax - vmin), 0, 1)


# ---------------------------------------------------------------------------
# Per-tile PCA (Diagnostic A)
# ---------------------------------------------------------------------------

def compute_per_tile_pca(
    se_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run PCA(n_components=3) on a single tile's SE embeddings.

    Args:
        se_data: (64, 512, 512) float32

    Returns:
        pca_rgb:  (512, 512, 3) float32, 2-98% stretched to [0, 1]
        pca_components: (3, 512, 512) individual PC maps
        explained_variance_ratios: (3,) array
    """
    n_bands, h, w = se_data.shape
    flat = se_data.reshape(n_bands, -1).T  # (n_pixels, 64)

    pca = PCA(n_components=3, random_state=SEED)
    projected = pca.fit_transform(flat)  # (n_pixels, 3)
    components = projected.T.reshape(3, h, w)  # (3, h, w)

    # PCA-RGB: per-component stretch
    rgb = np.stack(
        [percentile_stretch(components[i]) for i in range(3)], axis=-1,
    )  # (h, w, 3)

    return rgb, components, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Global PCA (Diagnostic B)
# ---------------------------------------------------------------------------

def sample_arctic_land_se(
    se_col: ee.ImageCollection,
    n_pixels: int = GLOBAL_PCA_N_PIXELS,
    seed: int = SEED,
) -> np.ndarray:
    """Sample n_pixels SE vectors from 60-74N land. Returns (n, 64). Cached."""
    key = f"global_pca_sample_{n_pixels}_{seed}_{SE_YEAR}"
    cached = load_cache(key)
    if cached is not None:
        return cached["sample"]

    logger.info("Sampling %d SE pixels from 60-74N land...", n_pixels)

    # The full 60-74N band is too large for a single GEE sample call.
    # Split into longitude strips and sample proportionally from each.
    lon_strips = [
        (-180, -120), (-120, -60), (-60, 0), (0, 60), (60, 120), (120, 180),
    ]
    n_strips = len(lon_strips)
    per_strip = n_pixels // n_strips
    remainder = n_pixels - per_strip * n_strips

    features: list[dict] = []
    for idx, (lon_min, lon_max) in tqdm(
        list(enumerate(lon_strips)), desc="Global PCA sampling", unit="strip",
    ):
        strip_n = per_strip + (1 if idx < remainder else 0)
        if strip_n == 0:
            continue
        strip_aoi = ee.Geometry.Rectangle(
            [lon_min, 60, lon_max, 74], proj="EPSG:4326",
        )
        se_mosaic = se_col.filterBounds(strip_aoi).mosaic().toFloat()
        strip_fc = se_mosaic.sample(
            region=strip_aoi, numPixels=strip_n, scale=10,
            seed=seed + idx, geometries=False,
        )
        strip_feats = strip_fc.getInfo()["features"]
        features.extend(strip_feats)

    print(f"  Total sample features from GEE: {len(features)}", flush=True)
    if not features:
        raise RuntimeError("Global PCA sampling returned 0 features from GEE")

    rows = []
    for feat in features:
        props = feat["properties"]
        # Band names are whatever GEE returns; sort for consistency
        band_names = sorted(props.keys())
        rows.append([props[b] for b in band_names])

    sample = np.array(rows, dtype=np.float32)
    save_cache(key, sample=sample)
    return sample


def fit_global_pca(sample: np.ndarray) -> PCA:
    """Fit PCA(3) on the global sample. Caches components+mean."""
    key = f"global_pca_model_{GLOBAL_PCA_N_PIXELS}_{SEED}_{SE_YEAR}"
    cached = load_cache(key, ext=".pkl")
    if cached is not None:
        return cached  # type: ignore[return-value]

    pca = PCA(n_components=3, random_state=SEED)
    pca.fit(sample)
    logger.info(
        "Global PCA explained variance: %s",
        np.round(pca.explained_variance_ratio_, 4),
    )

    p = cache_path(key, ext=".pkl")
    with open(p, "wb") as f:
        pickle.dump(pca, f)
    logger.info("  Cached: %s", p.name)
    return pca


def _load_global_pca_cache() -> PCA | None:
    """Try to load cached global PCA model."""
    if not USE_CACHE:
        return None
    key = f"global_pca_model_{GLOBAL_PCA_N_PIXELS}_{SEED}_{SE_YEAR}"
    p = cache_path(key, ext=".pkl")
    if not p.exists():
        return None
    logger.info("  Cache hit: %s", p.name)
    with open(p, "rb") as f:
        return pickle.load(f)  # noqa: S301


def project_global_pca(
    se_data: np.ndarray, pca: PCA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project tile's SE through global PCA.

    Returns same format as compute_per_tile_pca.
    """
    n_bands, h, w = se_data.shape
    flat = se_data.reshape(n_bands, -1).T  # (n_pixels, 64)
    projected = pca.transform(flat)  # (n_pixels, 3)
    components = projected.T.reshape(3, h, w)

    rgb = np.stack(
        [percentile_stretch(components[i]) for i in range(3)], axis=-1,
    )
    return rgb, components, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Prototype construction (Diagnostic C)
# ---------------------------------------------------------------------------

def _list_label_tiles() -> list[str]:
    """List all tile IDs from GCS labels directory."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blobs = bucket.list_blobs(prefix=f"{GCS_LABELS_PREFIX}/")
    tile_ids = []
    for blob in blobs:
        name = blob.name.split("/")[-1]
        if name.endswith(".tif"):
            tile_ids.append(name.replace(".tif", ""))
    return sorted(tile_ids)


def _load_label_rts_coords(
    tile_id: str, max_per_tile: int = 100,
) -> np.ndarray | None:
    """Download one label tif from GCS, return EPSG:4326 coords of RTS pixels.

    To avoid accumulating tens of millions of coordinates in memory,
    subsamples to at most max_per_tile pixels per tile.

    Returns (n, 2) array of [lon, lat] or None if no RTS pixels.
    """
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"{GCS_LABELS_PREFIX}/{tile_id}.tif")

    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)

    with rasterio.open(buf) as ds:
        label = ds.read(1)
        transform = ds.transform

    rts_rows, rts_cols = np.where(label == 1)
    if rts_rows.size == 0:
        return None

    # Subsample per-tile to cap memory usage
    n_rts = rts_rows.size
    if n_rts > max_per_tile:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n_rts, max_per_tile, replace=False)
        rts_rows = rts_rows[idx]
        rts_cols = rts_cols[idx]

    # Pixel (row, col) -> EPSG:3857 (x, y) via affine transform
    x_3857 = transform.c + rts_cols * transform.a + rts_rows * transform.b
    y_3857 = transform.f + rts_cols * transform.d + rts_rows * transform.e

    lon, lat = epsg3857_to_4326(x_3857, y_3857)
    return np.column_stack([lon, lat]).astype(np.float64)


def _sample_se_at_points(
    se_col: ee.ImageCollection, points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample SE values at EPSG:4326 points via GEE sampleRegions.

    Each input feature is tagged with its row index via an `idx` property so
    returned rows can be aligned with the original input order. Rows dropped
    by GEE (e.g. points falling outside any image) are reported via the
    ``kept_indices`` return value, which is monotonically increasing.

    Args:
        se_col: the SE ImageCollection (tiled, will be mosaicked per chunk)
        points: (n, 2) array of [lon, lat]

    Returns:
        (se_vectors, kept_indices):
          - se_vectors: (m, 64) float32 array of SE values (m <= n), sorted so
            that row i corresponds to original index kept_indices[i]
          - kept_indices: (m,) int64 array of input-row indices that survived
    """
    n = len(points)
    all_rows: list[list[float]] = []
    all_idx: list[int] = []
    band_names_ref: list[str] | None = None

    n_chunks = math.ceil(n / PROTOTYPE_CHUNK_SIZE)
    pbar = tqdm(total=n_chunks, desc="GEE sampleRegions", unit="chunk")

    for start in range(0, n, PROTOTYPE_CHUNK_SIZE):
        end = min(start + PROTOTYPE_CHUNK_SIZE, n)
        chunk = points[start:end]

        features = [
            ee.Feature(
                ee.Geometry.Point([float(lon), float(lat)]),
                {"idx": int(start + i)},
            )
            for i, (lon, lat) in enumerate(chunk)
        ]
        fc = ee.FeatureCollection(features)

        # Mosaic SE tiles covering this chunk's extent
        chunk_bounds = fc.geometry().bounds()
        se_image = get_se_image_for_region(se_col, chunk_bounds)

        try:
            result = se_image.sampleRegions(
                collection=fc,
                properties=["idx"],
                scale=10,
                geometries=False,
            )
            info = result.getInfo()
        except ee.ee_exception.EEException as exc:
            logger.warning("  sampleRegions failed at chunk %d: %s", start, exc)
            pbar.update(1)
            continue

        for feat in info["features"]:
            props = feat["properties"]
            if band_names_ref is None:
                band_names_ref = sorted(
                    k for k in props.keys()
                    if k not in ("system:index", "idx")
                )
            all_rows.append([props[b] for b in band_names_ref])
            all_idx.append(int(props["idx"]))
        pbar.update(1)

    pbar.close()

    if not all_rows:
        raise RuntimeError("sampleRegions returned no data for any chunk")

    rows_arr = np.array(all_rows, dtype=np.float32)
    idx_arr = np.array(all_idx, dtype=np.int64)
    order = np.argsort(idx_arr)
    return rows_arr[order], idx_arr[order]


def build_rts_prototype(
    se_col: ee.ImageCollection,
) -> tuple[np.ndarray, float, np.ndarray, dict]:
    """Build the RTS prototype vector from all labeled positive tiles.

    Returns:
        prototype: (64,) unit vector
        threshold: float (5th percentile of RTS cosine similarities)
        rts_cosine_sims: (n_sampled,) cosine similarities of RTS pixels
        metadata: dict with n_tiles, n_pixels_total, n_pixels_sampled
    """
    key = f"prototype_{SE_YEAR}"
    cached = load_cache(key)
    if cached is not None:
        return (
            cached["prototype"],
            float(cached["threshold"]),
            cached["rts_cosine_sims"],
            dict(cached["metadata"].item()),
        )

    logger.info("Building RTS prototype vector...")

    # Step 1: discover all label tiles
    tile_ids = _list_label_tiles()
    logger.info("  Found %d label tiles on GCS", len(tile_ids))

    # Step 2: load RTS pixel coordinates in parallel
    all_coords: list[np.ndarray] = []
    n_tiles_with_rts = 0

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {
            pool.submit(_load_label_rts_coords, tid): tid
            for tid in tile_ids
        }
        pbar = tqdm(
            as_completed(futures), total=len(futures),
            desc="Loading labels from GCS", unit="tile",
        )
        for future in pbar:
            tid = futures[future]
            try:
                coords = future.result()
            except Exception as exc:
                logger.warning("  Failed to load %s: %s", tid, exc)
                continue
            if coords is not None:
                all_coords.append(coords)
                n_tiles_with_rts += 1
        pbar.close()

    if not all_coords:
        raise RuntimeError("No RTS pixels found in any label tile")

    all_points = np.concatenate(all_coords, axis=0)
    n_total = len(all_points)
    logger.info(
        "  Total RTS pixels: %d across %d tiles", n_total, n_tiles_with_rts,
    )

    # Step 3: subsample if needed
    rng = np.random.RandomState(SEED)
    if n_total > PROTOTYPE_MAX_POINTS:
        idx = rng.choice(n_total, PROTOTYPE_MAX_POINTS, replace=False)
        all_points = all_points[idx]
        logger.info(
            "  Subsampled to %d points (%.1f%%)",
            PROTOTYPE_MAX_POINTS, 100 * PROTOTYPE_MAX_POINTS / n_total,
        )
    n_sampled = len(all_points)

    # Step 4: sample SE values at those points
    se_values, _kept = _sample_se_at_points(se_col, all_points)
    logger.info("  Got SE values for %d points", len(se_values))

    # Step 5: compute prototype (mean, re-normalize to unit length)
    mean_vec = se_values.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm < 1e-12:
        raise RuntimeError("Prototype vector has near-zero norm")
    prototype = (mean_vec / norm).astype(np.float32)

    # Step 6: threshold T = 5th percentile of cosine similarities
    rts_cosine_sims = (se_values @ prototype).astype(np.float32)
    threshold = float(np.percentile(rts_cosine_sims, 5))
    logger.info(
        "  Prototype norm=%.4f, threshold T=%.4f (5th pct)",
        norm, threshold,
    )

    metadata = {
        "n_tiles": n_tiles_with_rts,
        "n_pixels_total": n_total,
        "n_pixels_sampled": n_sampled,
        "n_se_sampled": len(se_values),
    }

    save_cache(
        key,
        prototype=prototype,
        threshold=np.array(threshold),
        rts_cosine_sims=rts_cosine_sims,
        metadata=np.array(metadata),
    )
    return prototype, threshold, rts_cosine_sims, metadata


# ---------------------------------------------------------------------------
# Prototype scoring
# ---------------------------------------------------------------------------

def score_tile_prototype(
    se_data: np.ndarray, prototype: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity of each pixel to prototype.

    Both SE embeddings and prototype are unit-length, so dot = cosine.

    Returns: (512, 512) similarity map in [-1, 1].
    """
    n_bands, h, w = se_data.shape
    flat = se_data.reshape(n_bands, -1).T  # (n_pixels, 64)
    sims = flat @ prototype  # (n_pixels,)
    return sims.reshape(h, w)


def compute_polygon_mean_sim(
    sim_map: np.ndarray,
    pixel_coords: list[tuple[float, float]],
) -> float:
    """Compute mean cosine similarity within the polygon."""
    h, w = sim_map.shape
    poly_path = mpath.Path(pixel_coords)
    yy, xx = np.mgrid[0:h, 0:w]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    mask = poly_path.contains_points(points).reshape(h, w)
    if mask.sum() == 0:
        return float("nan")
    return float(sim_map[mask].mean())


# ---------------------------------------------------------------------------
# SAR data fetching
# ---------------------------------------------------------------------------

def _db_to_linear(image: ee.Image) -> ee.Image:
    """Convert dB to linear power: 10^(dB/10)."""
    return ee.Image(10).pow(image.divide(10))


def _linear_to_db(image: ee.Image) -> ee.Image:
    """Convert linear power to dB: 10 * log10(linear)."""
    return image.log10().multiply(10)


def _detect_s1_mode(bbox: ee.Geometry) -> tuple[str, str, str]:
    """Detect which S1 mode/polarization is available at this location.

    Over Arctic Siberia, S1 uses EW mode with HH+HV polarization.
    Over Alaska/Canada/Europe, S1 uses IW mode with VV+VH polarization.

    Returns:
        (instrument_mode, copol_band, crosspol_band)
        e.g. ("IW", "VV", "VH") or ("EW", "HH", "HV")
    """
    # Try IW VV+VH first (standard land mode)
    iw_count = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(bbox)
        .filterDate("2024-07-01", "2024-10-01")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .size().getInfo()
    )
    if iw_count > 0:
        return "IW", "VV", "VH"

    # Fall back to EW HH+HV (Arctic Siberia)
    ew_count = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(bbox)
        .filterDate("2024-07-01", "2024-10-01")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "HH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "HV"))
        .filter(ee.Filter.eq("instrumentMode", "EW"))
        .size().getInfo()
    )
    if ew_count > 0:
        return "EW", "HH", "HV"

    return "NONE", "", ""


def fetch_extended_sar(
    bbox: ee.Geometry, grid: dict, tile_name: str,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fetch all extended SAR channels for a tile.

    Automatically detects S1 mode: IW VV+VH (Alaska/Canada) or
    EW HH+HV (Arctic Siberia). Band names in the output dict are
    generic: copol_db, crosspol_db, etc.

    S1 GRD in GEE stores values in dB. We convert to linear before
    aggregating, then back to dB.

    Returns:
        sar_data: dict of channel_name -> (512, 512) array, or None if no coverage
        sar_meta: dict with acquisition counts, orbit, mode, and band info
    """
    key = f"sar_tile_{tile_name}_{SE_YEAR}"
    cached = load_cache(key)
    if cached is not None:
        data = {k: cached[k] for k in cached if k != "meta"}
        meta = dict(cached["meta"].item()) if "meta" in cached else {}
        return data, meta

    logger.info("  Fetching extended SAR channels...")

    # Detect mode and polarization
    mode, copol, crosspol = _detect_s1_mode(bbox)
    if mode == "NONE":
        meta = {
            "mode": "NONE", "copol": "", "crosspol": "",
            "orbit": "N/A", "n_scenes_total": 0,
            "n_asc": 0, "n_desc": 0,
            "n_scenes_filtered": 0, "n_jul": 0, "n_augsep": 0,
        }
        logger.warning("  No S1 coverage at %s for 2024 Jul-Sep", tile_name)
        return None, meta

    logger.info("  S1 mode: %s %s+%s", mode, copol, crosspol)

    # Base collection with detected mode and polarization
    s1_full = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(bbox)
        .filterDate("2024-07-01", "2024-10-01")
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", copol))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", crosspol))
        .filter(ee.Filter.eq("instrumentMode", mode))
    )

    # Check orbit direction counts, pick dominant
    asc_count = s1_full.filter(
        ee.Filter.eq("orbitProperties_pass", "ASCENDING"),
    ).size().getInfo()
    desc_count = s1_full.filter(
        ee.Filter.eq("orbitProperties_pass", "DESCENDING"),
    ).size().getInfo()
    total_count = asc_count + desc_count

    if asc_count >= desc_count:
        orbit = "ASCENDING"
    else:
        orbit = "DESCENDING"
    logger.info(
        "  S1 acquisitions: %d total (%d asc, %d desc) -> using %s",
        total_count, asc_count, desc_count, orbit,
    )

    s1 = s1_full.filter(ee.Filter.eq("orbitProperties_pass", orbit))

    # Sub-period collections
    s1_jul = s1.filterDate("2024-07-01", "2024-08-01")
    s1_augsep = s1.filterDate("2024-08-01", "2024-10-01")

    jul_count = s1_jul.size().getInfo()
    augsep_count = s1_augsep.size().getInfo()
    full_count = s1.size().getInfo()
    logger.info(
        "  S1 scenes (filtered orbit): full=%d, Jul=%d, Aug-Sep=%d",
        full_count, jul_count, augsep_count,
    )

    meta = {
        "mode": mode,
        "copol": copol,
        "crosspol": crosspol,
        "orbit": orbit,
        "n_scenes_total": total_count,
        "n_asc": asc_count,
        "n_desc": desc_count,
        "n_scenes_filtered": full_count,
        "n_jul": jul_count,
        "n_augsep": augsep_count,
    }

    if full_count == 0:
        logger.warning("  No S1 coverage at %s for 2024 Jul-Sep", tile_name)
        return None, meta

    # Convert to linear for aggregation
    s1_lin = s1.map(lambda img: _db_to_linear(img.select([copol, crosspol])))

    # Full-window linear mean -> dB
    copol_lin_mean = s1_lin.select(copol).mean()
    crosspol_lin_mean = s1_lin.select(crosspol).mean()
    copol_db = _linear_to_db(copol_lin_mean).rename("copol_db")
    crosspol_db = _linear_to_db(crosspol_lin_mean).rename("crosspol_db")

    # Copol - Crosspol (in dB)
    pol_diff = copol_db.subtract(crosspol_db).rename("pol_diff")

    # RVI in linear space: 4 * crosspol / (copol + crosspol)
    rvi = (
        crosspol_lin_mean.multiply(4)
        .divide(copol_lin_mean.add(crosspol_lin_mean))
        .rename("rvi")
    )

    # Temporal std on dB scenes
    copol_std = s1.select(copol).reduce(ee.Reducer.stdDev()).rename("copol_std")
    crosspol_std = s1.select(crosspol).reduce(ee.Reducer.stdDev()).rename("crosspol_std")

    # Mid-season copol shift: dB(Aug-Sep mean) - dB(Jul mean)
    if jul_count > 0 and augsep_count > 0:
        jul_lin = s1_jul.map(
            lambda img: _db_to_linear(img.select(copol)),
        )
        augsep_lin = s1_augsep.map(
            lambda img: _db_to_linear(img.select(copol)),
        )
        copol_jul_db = _linear_to_db(jul_lin.mean())
        copol_augsep_db = _linear_to_db(augsep_lin.mean())
        mid_shift = copol_augsep_db.subtract(copol_jul_db).rename("mid_shift")
    else:
        mid_shift = ee.Image.constant(0).rename("mid_shift").toFloat()
        logger.warning(
            "  Missing sub-period data (Jul=%d, Aug-Sep=%d); "
            "mid-season shift will be zero",
            jul_count, augsep_count,
        )

    # Stack and fetch in one call
    stacked = (
        ee.Image.cat([
            copol_db, crosspol_db, pol_diff, rvi,
            copol_std, crosspol_std, mid_shift,
        ])
        .toFloat()
    )

    raw = fetch_pixels(stacked, grid, f"SAR extended {tile_name}")

    sar_data = {
        "copol_db": raw["copol_db"],
        "crosspol_db": raw["crosspol_db"],
        "pol_diff": raw["pol_diff"],
        "rvi": raw["rvi"],
        "copol_std": raw["copol_std"],
        "crosspol_std": raw["crosspol_std"],
        "mid_shift": raw["mid_shift"],
    }

    save_cache(key, meta=np.array(meta), **sar_data)
    return sar_data, meta


# ---------------------------------------------------------------------------
# Plotting — SE feasibility (4x3)
# ---------------------------------------------------------------------------

def plot_se_feasibility(
    rgb: np.ndarray,
    per_tile_pca_rgb: np.ndarray,
    per_tile_pcs: np.ndarray,
    per_tile_evr: np.ndarray,
    global_pca_rgb: np.ndarray,
    global_pcs: np.ndarray,
    global_evr: np.ndarray,
    similarity_map: np.ndarray,
    threshold: float,
    rts_cosine_sims: np.ndarray,
    pixel_coords: list[tuple[float, float]],
    polygon_mean_sim: float,
    proto_meta: dict,
    title_prefix: str,
    output_path: Path,
) -> None:
    """Create the SE feasibility figure (4x3 grid)."""
    fig, axes = plt.subplots(4, 3, figsize=(16, 20))

    # Top annotation
    fig.suptitle(
        f"SE Feasibility \u2014 {title_prefix}\n"
        f"Year: {SE_YEAR}  |  "
        f"Global sample: {GLOBAL_PCA_N_PIXELS:,}  |  "
        f"Prototype: {proto_meta.get('n_tiles', '?')} tiles, "
        f"{proto_meta.get('n_se_sampled', '?')} SE pixels sampled",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # --- Row 0: RGB, Per-tile PCA-RGB, Global PCA-RGB ---
    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title("RGB (PlanetScope)", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    evr_str = ", ".join(f"{v:.2f}" for v in per_tile_evr)
    ax = axes[0, 1]
    ax.imshow(per_tile_pca_rgb)
    ax.set_title(f"Per-tile PCA-RGB ({evr_str})", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    evr_str = ", ".join(f"{v:.2f}" for v in global_evr)
    ax = axes[0, 2]
    ax.imshow(global_pca_rgb)
    ax.set_title(f"Global PCA-RGB ({evr_str})", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    # --- Row 1: Per-tile PC1, PC2, PC3 ---
    for j in range(3):
        ax = axes[1, j]
        data = per_tile_pcs[j]
        vmin, vmax = np.percentile(data[np.isfinite(data)], [2, 98])
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ax.set_title(
            f"Per-tile PC{j+1} ({per_tile_evr[j]:.2%})", fontsize=10,
        )
        add_polygon_overlay(ax, pixel_coords)

    # --- Row 2: Global PC1, PC2, PC3 ---
    for j in range(3):
        ax = axes[2, j]
        data = global_pcs[j]
        vmin, vmax = np.percentile(data[np.isfinite(data)], [2, 98])
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ax.set_title(
            f"Global PC{j+1} ({global_evr[j]:.2%})", fontsize=10,
        )
        add_polygon_overlay(ax, pixel_coords)

    # --- Row 3: Prototype similarity, threshold mask, histogram ---

    # Similarity map
    ax = axes[3, 0]
    im = ax.imshow(
        similarity_map, cmap="RdBu_r", vmin=-1, vmax=1,
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("cos similarity", fontsize=8)
    ax.set_title("Prototype cosine similarity", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    # Threshold mask
    ax = axes[3, 1]
    mask = (similarity_map >= threshold).astype(np.uint8)
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Threshold mask (T={threshold:.3f})", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    # Similarity histogram
    ax = axes[3, 2]
    flat_sim = similarity_map.ravel()
    ax.hist(flat_sim, bins=100, color="steelblue", alpha=0.7, log=True)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"T={threshold:.3f}")
    frac_above = float((flat_sim >= threshold).sum()) / flat_sim.size
    ax.set_title("Similarity histogram", fontsize=10)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count (log)")
    ax.legend(fontsize=8)
    ax.annotate(
        f"Tile above T: {frac_above:.1%}\n"
        f"Polygon mean: {polygon_mean_sim:.3f}",
        xy=(0.02, 0.95), xycoords="axes fraction",
        fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # Remove ticks from spatial panels
    for row in range(4):
        for col in range(3):
            if not (row == 3 and col == 2):  # skip histogram
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Plotting — Extended SAR (3x3)
# ---------------------------------------------------------------------------

def plot_sar_extended(
    rgb: np.ndarray,
    sar_data: dict[str, np.ndarray],
    pixel_coords: list[tuple[float, float]],
    sar_meta: dict,
    title_prefix: str,
    output_path: Path,
) -> None:
    """Create the extended SAR figure (3x3 grid)."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))

    no_data = sar_data is None
    orbit = sar_meta.get("orbit", "?")
    mode = sar_meta.get("mode", "?")
    copol = sar_meta.get("copol", "?") if not no_data else "?"
    crosspol = sar_meta.get("crosspol", "?") if not no_data else "?"
    n_full = sar_meta.get("n_scenes_filtered", 0)
    n_jul = sar_meta.get("n_jul", 0)
    n_augsep = sar_meta.get("n_augsep", 0)

    if no_data:
        fig.suptitle(
            f"Extended SAR \u2014 {title_prefix}\n"
            f"2024 Jul\u2013Sep  |  No S1 data available",
            fontsize=14, fontweight="bold", y=0.98,
        )
    else:
        fig.suptitle(
            f"Extended SAR \u2014 {title_prefix}\n"
            f"2024 Jul\u2013Sep  |  {mode} {copol}+{crosspol}  |  "
            f"Orbit: {orbit}  |  "
            f"Scenes: {n_full} total ({n_jul} Jul, {n_augsep} Aug\u2013Sep)",
            fontsize=14, fontweight="bold", y=0.98,
        )

    # --- Row 0: RGB, copol dB, crosspol dB ---
    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title("RGB (PlanetScope)", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    if no_data:
        # Fill all non-RGB panels with "No S1 data" message
        panel_labels = [
            "Co-pol (dB)", "Cross-pol (dB)",
            "Co-pol \u2212 Cross-pol (dB)", "RVI", "Dual-pol composite",
            "Co-pol temporal std (dB)", "Cross-pol temporal std (dB)",
            "Mid-season shift",
        ]
        panel_positions = [
            (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
        ]
        for (r, c), lbl in zip(panel_positions, panel_labels):
            ax = axes[r, c]
            ax.set_facecolor("#f0f0f0")
            ax.text(
                0.5, 0.5, "No S1 data\n(Jul\u2013Sep 2024)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="#888888", fontweight="bold",
            )
            ax.set_title(lbl, fontsize=10)
    else:
        for j, (key, label) in enumerate(
            [
                ("copol_db", f"{copol} (dB)"),
                ("crosspol_db", f"{crosspol} (dB)"),
            ],
            start=1,
        ):
            ax = axes[0, j]
            data = sar_data[key]
            valid = data[np.isfinite(data)]
            vmin, vmax = np.percentile(valid, [2, 98]) if valid.size > 0 else (0, 1)
            im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
            cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            cb.set_label("dB", fontsize=8)
            ax.set_title(label, fontsize=10)
            add_polygon_overlay(ax, pixel_coords)

        # --- Row 1: copol-crosspol, RVI, Dual-pol composite ---

        # Copol - Crosspol
        ax = axes[1, 0]
        data = sar_data["pol_diff"]
        valid = data[np.isfinite(data)]
        vmin, vmax = np.percentile(valid, [2, 98]) if valid.size > 0 else (0, 1)
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
        cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label("dB", fontsize=8)
        ax.set_title(f"{copol} \u2212 {crosspol} (dB)", fontsize=10)
        add_polygon_overlay(ax, pixel_coords)

        # RVI
        ax = axes[1, 1]
        im = ax.imshow(sar_data["rvi"], cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ax.set_title("RVI", fontsize=10)
        add_polygon_overlay(ax, pixel_coords)

        # Dual-pol composite: R=copol_dB, G=crosspol_dB, B=pol_diff_dB
        ax = axes[1, 2]
        r = percentile_stretch(sar_data["copol_db"])
        g = percentile_stretch(sar_data["crosspol_db"])
        b = percentile_stretch(sar_data["pol_diff"])
        dual_rgb = np.stack([r, g, b], axis=-1)
        ax.imshow(dual_rgb)
        ax.set_title("Dual-pol composite", fontsize=10)
        ax.text(
            0.02, 0.02,
            f"R={copol}  G={crosspol}  B={copol}\u2212{crosspol}",
            transform=ax.transAxes, fontsize=7, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
        )
        add_polygon_overlay(ax, pixel_coords)

        # --- Row 2: copol std, crosspol std, Mid-season copol shift ---

        for j, (key, label) in enumerate(
            [
                ("copol_std", f"{copol} temporal std (dB)"),
                ("crosspol_std", f"{crosspol} temporal std (dB)"),
            ],
        ):
            ax = axes[2, j]
            data = sar_data[key]
            valid = data[np.isfinite(data)]
            vmin, vmax = np.percentile(valid, [2, 98]) if valid.size > 0 else (0, 1)
            im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
            cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            cb.set_label("dB", fontsize=8)
            ax.set_title(label, fontsize=10)
            add_polygon_overlay(ax, pixel_coords)

        # Mid-season copol shift
        ax = axes[2, 2]
        data = sar_data["mid_shift"]
        valid = data[np.isfinite(data)]
        if valid.size > 0:
            abs_pct = np.percentile(np.abs(valid), 98)
            vmin_ms, vmax_ms = -abs_pct, abs_pct
        else:
            vmin_ms, vmax_ms = -1, 1
        im = ax.imshow(data, cmap="RdBu_r", vmin=vmin_ms, vmax=vmax_ms)
        cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label("dB", fontsize=8)
        ax.set_title(
            f"Mid-season {copol} shift (Aug\u2013Sep \u2212 Jul)", fontsize=10,
        )
        add_polygon_overlay(ax, pixel_coords)

    # Remove ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI + Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="SE feasibility and extended SAR visualization.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip reading cached data (still writes cache for next run).",
    )
    return parser.parse_args()


def main() -> None:
    global USE_CACHE  # noqa: PLW0603

    args = parse_args()
    USE_CACHE = not args.no_cache

    ensure_cache_dir()
    entries = load_mapping()
    print(f"Loaded {len(entries)} polygon-tile pairs", flush=True)

    initialize_gee()

    # ---------------------------------------------------------------
    # Phase 1: Global precomputation (one-time)
    # ---------------------------------------------------------------
    se_col = get_se_collection()

    print("--- Global PCA ---", flush=True)
    global_sample = sample_arctic_land_se(se_col)
    global_pca = _load_global_pca_cache()
    if global_pca is None:
        global_pca = fit_global_pca(global_sample)

    print("--- Prototype construction ---", flush=True)
    prototype, threshold, rts_cosine_sims, proto_meta = build_rts_prototype(
        se_col,
    )

    # ---------------------------------------------------------------
    # Phase 2: Per-tile loop
    # ---------------------------------------------------------------
    for i, entry in enumerate(
        tqdm(entries, desc="Processing tiles", unit="tile"), 1,
    ):
        oid = entry["object_id"]
        tile_name = entry["tile_name"]
        label = f"OID {oid} ({tile_name})"
        print(f"\n=== [{i}/{len(entries)}] {label} ===", flush=True)
        t_start = time.time()

        rgb = load_planetscope_rgb(entry["tile_path"])
        pixel_coords = polygon_to_pixel_coords(
            entry["polygon_coords"],
            entry["origin_x"], entry["origin_y"], entry["pixel_scale"],
        )
        grid = make_gee_grid(
            entry["origin_x"], entry["origin_y"], entry["pixel_scale"],
        )
        bbox = make_tile_bbox(
            entry["origin_x"], entry["origin_y"], entry["pixel_scale"],
        )

        # --- SE feasibility ---
        se_data = fetch_se_tile(se_col, bbox, grid, tile_name)

        per_tile_pca_rgb, per_tile_pcs, per_tile_evr = compute_per_tile_pca(
            se_data,
        )
        global_pca_rgb, global_pcs, global_evr = project_global_pca(
            se_data, global_pca,
        )
        sim_map = score_tile_prototype(se_data, prototype)
        polygon_mean_sim = compute_polygon_mean_sim(sim_map, pixel_coords)

        out_prefix = f"oid{oid}"
        plot_se_feasibility(
            rgb=rgb,
            per_tile_pca_rgb=per_tile_pca_rgb,
            per_tile_pcs=per_tile_pcs,
            per_tile_evr=per_tile_evr,
            global_pca_rgb=global_pca_rgb,
            global_pcs=global_pcs,
            global_evr=global_evr,
            similarity_map=sim_map,
            threshold=threshold,
            rts_cosine_sims=rts_cosine_sims,
            pixel_coords=pixel_coords,
            polygon_mean_sim=polygon_mean_sim,
            proto_meta=proto_meta,
            title_prefix=label,
            output_path=(SCRIPT_DIR / "se_derived" / "feasibility" / f"{out_prefix}_se_feasibility.png"),
        )

        # --- Extended SAR ---
        sar_data, sar_meta = fetch_extended_sar(bbox, grid, tile_name)
        plot_sar_extended(
            rgb=rgb,
            sar_data=sar_data,
            pixel_coords=pixel_coords,
            sar_meta=sar_meta,
            title_prefix=label,
            output_path=(SCRIPT_DIR / "sentinel_derived" / "sar_extended" / f"{out_prefix}_sar_extended.png"),
        )

        logger.info("  Total for %s: %.1fs", label, time.time() - t_start)

    logger.info("Done \u2014 all %d polygons processed.", len(entries))


if __name__ == "__main__":
    main()
