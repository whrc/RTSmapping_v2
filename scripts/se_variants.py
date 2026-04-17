"""
Phase 2 — SE-cosine variant evaluation for SE investigation.

Two approaches to make SE-cosine a complementary signal for RTS segmentation:
  Approach 1 (multi-prototype): k-means on reference pool, max-cosine scoring.
  Approach 2 (contrastive): positive minus negative prototype cosine.

Both approaches operate directly on raw 64-band SE unit vectors (cosine = dot
product). No preprocessing is applied to the embeddings.

Evaluates on 7 test tiles (OID 93, 113, 136, 144, 169, 187, 262) with figures,
sign-consistency tables, and correlation against S2 channels.

Usage:
  python scripts/se_variants.py --config configs/se_investigation.yaml --approach approach1
  python scripts/se_variants.py --config configs/se_investigation.yaml --approach all
"""

import argparse
import functools
import io
import json
import logging
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from google.cloud import storage
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add vis directory to path for reuse of existing functions
# ---------------------------------------------------------------------------
_VIS_DIR = Path(__file__).resolve().parent.parent / "plots" / "extra_channel_vis"
sys.path.insert(0, str(_VIS_DIR))

from extra_channel_plot import (  # noqa: E402
    TILE_SIZE,
    add_polygon_overlay,
    fetch_sentinel2_channels,
    initialize_gee,
    load_mapping,
    load_planetscope_rgb,
    make_gee_grid,
    make_tile_bbox,
    polygon_to_pixel_coords,
)

import se_sar_plot  # noqa: E402
from se_sar_plot import (  # noqa: E402
    CACHE_DIR,
    GCS_BUCKET,
    GCS_LABELS_PREFIX,
    SEED,
    SE_YEAR,
    _list_label_tiles,
    _load_label_rts_coords,
    _sample_se_at_points,
    build_rts_prototype,
    compute_polygon_mean_sim,
    ensure_cache_dir,
    epsg3857_to_4326,
    fetch_se_tile,
    get_se_collection,
    load_cache,
    save_cache,
    score_tile_prototype,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Earth radius for coordinate conversion
_R = 6378137.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reference pool construction
# ---------------------------------------------------------------------------

def _load_label_rts_coords_with_rowcol(
    tile_id: str, max_per_tile: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Download a label tif from GCS, return RTS pixel (lon, lat, row, col).

    Mirrors `se_sar_plot._load_label_rts_coords` but also returns the
    pixel-space (row, col) aligned with each (lon, lat). Used by
    :func:`build_reference_pool` to preserve per-sample provenance through
    GEE sampling.

    Returns:
        (points, rows, cols) where
          - points: (n, 2) float64 [lon, lat] in EPSG:4326
          - rows, cols: (n,) int arrays of pixel indices in the 512x512 tile
        or None if the tile has no RTS pixels.
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

    n_rts = rts_rows.size
    if n_rts > max_per_tile:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(n_rts, max_per_tile, replace=False)
        rts_rows = rts_rows[idx]
        rts_cols = rts_cols[idx]

    x_3857 = transform.c + rts_cols * transform.a + rts_rows * transform.b
    y_3857 = transform.f + rts_cols * transform.d + rts_rows * transform.e
    lon, lat = epsg3857_to_4326(x_3857, y_3857)
    points = np.column_stack([lon, lat]).astype(np.float64)
    return points, rts_rows.astype(np.int32), rts_cols.astype(np.int32)


def build_reference_pool(
    se_col,
    reference_tile_ids: list[str],
    max_points: int,
    seed: int,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, list[str] | None, np.ndarray | None, np.ndarray | None]:
    """Build reference pool of raw SE vectors from RTS pixels.

    Tracks per-point provenance (tile_id, row, col) through GEE sampling by
    tagging each feature with an `idx` property and aligning returned rows
    with the input order. This lets callers render the actual nearest-pixel
    crops (not tile-center fallbacks).

    Args:
        se_col: GEE SE ImageCollection.
        reference_tile_ids: Tile IDs to sample from.
        max_points: Maximum number of reference points.
        seed: Random seed.
        force_rebuild: Ignore cache and rebuild from scratch.

    Returns:
        se_vectors: (N, 64) float32 unit vectors.
        tile_ids: List of N tile IDs, or None on alignment failure.
        rows, cols: (N,) int arrays of pixel indices, or None on failure.
    """
    key_v2 = f"reference_pool_se_{SE_YEAR}_v2"
    key_v1 = f"reference_pool_se_{SE_YEAR}"
    if not force_rebuild:
        cached = load_cache(key_v2)
        if cached is not None:
            se_vectors = cached["se_vectors"]
            tile_ids = cached.get("tile_ids")
            if tile_ids is not None:
                tile_ids = tile_ids.tolist()
            rows = cached.get("rows")
            cols = cached.get("cols")
            logger.info(
                "Loaded reference pool (v2) from cache: %d vectors",
                len(se_vectors),
            )
            return se_vectors, tile_ids, rows, cols
        cached = load_cache(key_v1)
        if cached is not None:
            se_vectors = cached["se_vectors"]
            tile_ids = cached.get("tile_ids")
            if tile_ids is not None:
                tile_ids = tile_ids.tolist()
            logger.info(
                "Loaded legacy reference pool (v1) from cache: %d vectors "
                "(no row/col metadata — rerun with --rebuild-reference-pool "
                "to regenerate prototype_samples crops)",
                len(se_vectors),
            )
            return se_vectors, tile_ids, None, None

    logger.info("Building reference pool from %d tiles...", len(reference_tile_ids))

    # Step 1: collect RTS coords + (row, col) with tile IDs
    all_coords: list[np.ndarray] = []
    all_tile_ids: list[str] = []
    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    n_tiles_with_rts = 0

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {
            pool.submit(_load_label_rts_coords_with_rowcol, tid): tid
            for tid in reference_tile_ids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Loading RTS coords from GCS", unit="tile",
        ):
            tid = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("Failed to load %s: %s", tid, exc)
                continue
            if result is None:
                continue
            coords, rts_rows, rts_cols = result
            all_coords.append(coords)
            all_rows.append(rts_rows)
            all_cols.append(rts_cols)
            all_tile_ids.extend([tid] * len(coords))
            n_tiles_with_rts += 1

    if not all_coords:
        raise RuntimeError("No RTS pixels found in reference tiles")

    all_points = np.concatenate(all_coords, axis=0)
    all_rows_arr = np.concatenate(all_rows, axis=0)
    all_cols_arr = np.concatenate(all_cols, axis=0)
    n_total = len(all_points)
    logger.info(
        "Total RTS pixels: %d across %d tiles", n_total, n_tiles_with_rts,
    )

    # Step 2: subsample
    rng = np.random.RandomState(seed)
    if n_total > max_points:
        idx = rng.choice(n_total, max_points, replace=False)
        all_points = all_points[idx]
        all_rows_arr = all_rows_arr[idx]
        all_cols_arr = all_cols_arr[idx]
        all_tile_ids = [all_tile_ids[i] for i in idx]
        logger.info("Subsampled to %d points", max_points)

    # Step 3: sample SE with per-feature index tracking
    se_vectors, kept = _sample_se_at_points(se_col, all_points)
    logger.info(
        "Got SE values for %d / %d points (%d dropped by GEE)",
        len(se_vectors), len(all_points), len(all_points) - len(se_vectors),
    )

    # Step 4: slice provenance arrays by kept indices
    tile_ids_out = [all_tile_ids[i] for i in kept]
    rows_out = all_rows_arr[kept]
    cols_out = all_cols_arr[kept]

    save_cache(
        key_v2,
        se_vectors=se_vectors,
        tile_ids=np.array(tile_ids_out, dtype=object),
        rows=rows_out,
        cols=cols_out,
    )

    return se_vectors, tile_ids_out, rows_out, cols_out


def build_negative_pool(
    se_col,
    tile_ids: list[str],
    n_tiles: int,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Sample SE vectors from background (label=0) pixels.

    Args:
        se_col: GEE SE ImageCollection.
        tile_ids: All tile IDs to sample from.
        n_tiles: Number of tiles to use.
        max_points: Maximum number of points.
        seed: Random seed.

    Returns:
        (N, 64) float32 array of SE vectors.
    """
    key = f"negative_pool_se_{SE_YEAR}"
    cached = load_cache(key)
    if cached is not None:
        logger.info(
            "Loaded negative pool from cache: %d vectors",
            len(cached["se_vectors"]),
        )
        return cached["se_vectors"]

    logger.info("Building negative pool from %d tiles...", n_tiles)

    # Sample a subset of tiles
    rng = np.random.RandomState(seed)
    selected = rng.choice(tile_ids, min(n_tiles, len(tile_ids)), replace=False)

    all_coords: list[np.ndarray] = []

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {
            pool.submit(_load_neg_coords, tid): tid for tid in selected
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Loading background coords", unit="tile",
        ):
            try:
                coords = future.result()
            except Exception as exc:
                logger.warning("Negative pool tile failed: %s", exc)
                continue
            if coords is not None:
                all_coords.append(coords)

    if not all_coords:
        raise RuntimeError("No background pixels found")

    all_points = np.concatenate(all_coords, axis=0)
    logger.info("Total background pixels: %d", len(all_points))

    if len(all_points) > max_points:
        idx = rng.choice(len(all_points), max_points, replace=False)
        all_points = all_points[idx]
        logger.info("Subsampled to %d points", max_points)

    se_vectors, _kept = _sample_se_at_points(se_col, all_points)
    logger.info("Got SE values for %d background points", len(se_vectors))

    save_cache(key, se_vectors=se_vectors)
    return se_vectors


def _load_neg_coords(
    tile_id: str, max_per_tile: int = 50,
) -> np.ndarray | None:
    """Load background (label=0) pixel coordinates from one tile.

    Returns (n, 2) array of [lon, lat] in EPSG:4326, or None.
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

    bg_rows, bg_cols = np.where(label == 0)
    if bg_rows.size == 0:
        return None

    # Subsample aggressively (background is vast)
    rng = np.random.RandomState(SEED)
    if bg_rows.size > max_per_tile:
        idx = rng.choice(bg_rows.size, max_per_tile, replace=False)
        bg_rows = bg_rows[idx]
        bg_cols = bg_cols[idx]

    x_3857 = transform.c + bg_cols * transform.a + bg_rows * transform.b
    y_3857 = transform.f + bg_cols * transform.d + bg_rows * transform.e

    lon, lat = epsg3857_to_4326(x_3857, y_3857)
    return np.column_stack([lon, lat]).astype(np.float64)


# ---------------------------------------------------------------------------
# k-means and scoring
# ---------------------------------------------------------------------------

def run_kmeans_sweep(
    se_vectors: np.ndarray,
    k_values: list[int],
    seed: int,
) -> dict[int, dict]:
    """Run k-means for multiple k values, compute silhouette scores.

    Args:
        se_vectors: (N, 64) reference pool.
        k_values: List of k to try.
        seed: Random seed.

    Returns:
        Dict mapping k to {model, labels, centroids, silhouette}.
    """
    results = {}
    # Subsample for silhouette computation (expensive for large N)
    rng = np.random.RandomState(seed)
    n_sil = min(10000, len(se_vectors))
    sil_idx = rng.choice(len(se_vectors), n_sil, replace=False)
    sil_data = se_vectors[sil_idx]

    for k in k_values:
        logger.info("k-means k=%d ...", k)
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        labels = km.fit_predict(se_vectors)
        centroids = km.cluster_centers_.copy()

        # Re-normalize centroids to unit length
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        centroids = centroids / norms

        # Silhouette on subsample
        sil_labels = km.predict(sil_data)
        sil = silhouette_score(sil_data, sil_labels)
        logger.info("  k=%d: silhouette=%.4f", k, sil)

        results[k] = {
            "model": km,
            "labels": labels,
            "centroids": centroids,
            "silhouette": sil,
        }

    return results


def select_best_k(results: dict[int, dict]) -> int:
    """Select k with highest silhouette score."""
    return max(results, key=lambda k: results[k]["silhouette"])


def score_tile_multiprototype(
    se_data: np.ndarray, centroids: np.ndarray,
) -> np.ndarray:
    """Compute max-cosine across multiple prototypes.

    Args:
        se_data: (64, H, W) float32 unit vectors.
        centroids: (k, 64) unit vectors.

    Returns:
        (H, W) float32 max-cosine similarity map.
    """
    n_bands, h, w = se_data.shape
    flat = se_data.reshape(n_bands, -1).T  # (n_pixels, 64)
    sims = flat @ centroids.T  # (n_pixels, k)
    return sims.max(axis=1).reshape(h, w)


def score_tile_contrastive(
    se_data: np.ndarray,
    pos_centroids: np.ndarray,
    neg_centroids: np.ndarray,
) -> np.ndarray:
    """Compute contrastive cosine: max-positive minus max-negative.

    Args:
        se_data: (64, H, W) float32.
        pos_centroids: (k_pos, 64) unit vectors.
        neg_centroids: (k_neg, 64) unit vectors.

    Returns:
        (H, W) float32 in approximately [-2, 2].
    """
    n_bands, h, w = se_data.shape
    flat = se_data.reshape(n_bands, -1).T  # (n_pixels, 64)
    pos_sims = (flat @ pos_centroids.T).max(axis=1)  # (n_pixels,)
    neg_sims = (flat @ neg_centroids.T).max(axis=1)  # (n_pixels,)
    return (pos_sims - neg_sims).reshape(h, w)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_variant_tile(
    rgb: np.ndarray,
    sim_map: np.ndarray,
    threshold: float,
    pixel_coords: list[tuple[float, float]],
    polygon_mean_sim: float,
    title_prefix: str,
    output_path: Path,
) -> None:
    """Create the SE variant per-tile figure (2x2 grid).

    Layout:
      [RGB]              [Similarity map]
      [Threshold mask]   [Histogram]
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    fig.suptitle(title_prefix, fontsize=14, fontweight="bold", y=0.98)

    # --- (0,0) RGB ---
    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title("RGB (PlanetScope)", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    # --- (0,1) Similarity map ---
    ax = axes[0, 1]
    # Auto vmin/vmax from 2-98 percentile for better contrast
    valid = sim_map[np.isfinite(sim_map)]
    if valid.size > 0:
        vmin, vmax = np.percentile(valid, [2, 98])
        # Ensure symmetric around 0 if range straddles 0
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -1, 1
    im = ax.imshow(sim_map, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("cosine similarity", fontsize=8)
    ax.set_title("SE variant similarity", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    # --- (1,0) Threshold mask ---
    ax = axes[1, 0]
    mask = (sim_map >= threshold).astype(np.uint8)
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Threshold mask (T={threshold:.3f})", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    # --- (1,1) Histogram ---
    ax = axes[1, 1]
    flat_sim = sim_map[np.isfinite(sim_map)]
    if flat_sim.size > 0:
        ax.hist(flat_sim, bins=100, color="steelblue", alpha=0.7, log=True)
        ax.axvline(
            threshold, color="red", linestyle="--", linewidth=1.5,
            label=f"T={threshold:.3f}",
        )
        frac_above = float((flat_sim >= threshold).sum()) / flat_sim.size
        ax.annotate(
            f"Tile above T: {frac_above:.1%}\n"
            f"Polygon mean: {polygon_mean_sim:.3f}",
            xy=(0.02, 0.95), xycoords="axes fraction",
            fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        ax.legend(fontsize=8)
    ax.set_title("Similarity histogram", fontsize=10)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count (log)")

    # Remove ticks from spatial panels
    for r in range(2):
        for c in range(2):
            if not (r == 1 and c == 1):
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_prototype_samples(
    se_vectors: np.ndarray,
    tile_ids: list[str] | None,
    rows: np.ndarray | None,
    cols: np.ndarray | None,
    centroids: np.ndarray,
    labels: np.ndarray,
    silhouette_scores: dict[int, float],
    best_k: int,
    config: dict,
    output_path: Path,
    n_nearest: int = 6,
) -> None:
    """Render nearest-pixel SE samples per cluster.

    For each centroid, finds the N nearest reference SE vectors and renders
    RGB crops centered on the actual nearest pixel (row, col). Falls back
    to text-only summary if provenance metadata is missing.

    Also shows silhouette scores for each k.
    """
    k = centroids.shape[0]

    fig_height = 4 + k * 2
    fig, axes = plt.subplots(
        k + 1, 1, figsize=(12, fig_height),
        gridspec_kw={"height_ratios": [2] + [2] * k},
    )

    # Silhouette bar chart
    ax = axes[0]
    ks = sorted(silhouette_scores.keys())
    sils = [silhouette_scores[ki] for ki in ks]
    colors = ["darkorange" if ki == best_k else "steelblue" for ki in ks]
    ax.bar([str(ki) for ki in ks], sils, color=colors)
    ax.set_ylabel("Silhouette score")
    ax.set_xlabel("k")
    ax.set_title(
        f"k-means Silhouette Scores (best k={best_k})",
        fontsize=12, fontweight="bold",
    )

    have_rowcol = (
        tile_ids is not None and rows is not None and cols is not None
    )

    for ci in range(k):
        ax = axes[ci + 1]
        dists = 1 - se_vectors @ centroids[ci]
        nearest_idx = np.argsort(dists)[:n_nearest]
        nearest_sims = 1 - dists[nearest_idx]

        cluster_size = int((labels == ci).sum())
        cluster_pct = 100 * cluster_size / len(labels)

        if have_rowcol:
            samples = [
                (tile_ids[i], int(rows[i]), int(cols[i]), float(nearest_sims[j]))
                for j, i in enumerate(nearest_idx)
            ]
            _render_nearest_crops(ax, samples, config)
            ax.set_title(
                f"Cluster {ci} — {cluster_size} vectors ({cluster_pct:.1f}%)",
                fontsize=10,
            )
        else:
            text = (
                f"Cluster {ci}: {cluster_size} vectors ({cluster_pct:.1f}%)\n"
                f"Nearest cosine sims: {nearest_sims[:5].round(4)}\n"
                f"(Provenance unavailable — rebuild reference pool to enable crops)"
            )
            ax.text(
                0.5, 0.5, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=9,
                fontfamily="monospace",
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


@functools.lru_cache(maxsize=64)
def _download_rgb_tile(bucket_name: str, rgb_prefix: str, tile_id: str) -> np.ndarray:
    """Download and decode a PlanetScope RGB tile from GCS (cached)."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{rgb_prefix}/{tile_id}.tif")
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    with rasterio.open(buf) as ds:
        data = ds.read()
    return np.moveaxis(data, 0, -1).astype(np.float32) / 255.0


def _render_nearest_crops(
    ax: plt.Axes,
    samples: list[tuple[str, int, int, float]],
    config: dict,
    crop_size: int = 64,
) -> None:
    """Render RGB crops centered on each sample's actual (row, col).

    Args:
        ax: parent axes to populate with inset crops.
        samples: list of (tile_id, row, col, cosine_similarity) per nearest
            pixel, already ordered most-similar first.
        config: top-level YAML config (used to resolve GCS paths).
        crop_size: side length of each crop in pixels.
    """
    gcs = config["gcs"]
    half = crop_size // 2
    crops: list[tuple[str, float, np.ndarray]] = []

    for tid, row, col, sim in samples:
        try:
            rgb = _download_rgb_tile(gcs["bucket"], gcs["rgb_prefix"], tid)
        except Exception as exc:
            logger.warning("  RGB download failed for %s: %s", tid, exc)
            continue

        h, w = rgb.shape[:2]
        r0 = max(0, row - half)
        r1 = min(h, row + half)
        c0 = max(0, col - half)
        c1 = min(w, col + half)
        crop = rgb[r0:r1, c0:c1]

        # Pad to crop_size if clipped at edge
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            padded = np.zeros((crop_size, crop_size, crop.shape[2]), dtype=crop.dtype)
            padded[: crop.shape[0], : crop.shape[1]] = crop
            crop = padded
        crops.append((tid, sim, crop))

    if crops:
        n = len(crops)
        for i, (tid, sim, crop) in enumerate(crops):
            left = i / n
            width = 1.0 / n
            inset = ax.inset_axes([left, 0.0, width * 0.95, 0.85])
            inset.imshow(np.clip(crop, 0, 1))
            inset.set_title(f"{tid[-12:]}\ncos={sim:.3f}", fontsize=6)
            inset.set_xticks([])
            inset.set_yticks([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    else:
        ax.text(
            0.5, 0.5, "(No crops available)",
            transform=ax.transAxes, ha="center", va="center", fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])


# ---------------------------------------------------------------------------
# Sign consistency & correlation tables
# ---------------------------------------------------------------------------

def compute_background_mode(sim_map: np.ndarray) -> float:
    """Estimate background mode from similarity map histogram."""
    valid = sim_map[np.isfinite(sim_map)]
    if valid.size == 0:
        return float("nan")
    counts, edges = np.histogram(valid, bins=200)
    mode_idx = np.argmax(counts)
    return float((edges[mode_idx] + edges[mode_idx + 1]) / 2)


def write_sign_consistency(
    results: list[dict],
    output_path: Path,
) -> None:
    """Write sign-consistency table to markdown.

    Args:
        results: List of dicts with keys: oid, polygon_mean, background_mode.
    """
    lines = [
        "# Sign Consistency",
        "",
        "| OID | Polygon Mean | Background Mode | Direction Correct |",
        "|-----|-------------|-----------------|-------------------|",
    ]
    n_correct = 0
    for r in results:
        correct = r["polygon_mean"] > r["background_mode"]
        n_correct += int(correct)
        mark = "Yes" if correct else "**No**"
        lines.append(
            f"| {r['oid']} | {r['polygon_mean']:.4f} | "
            f"{r['background_mode']:.4f} | {mark} |",
        )

    n_total = len(results)
    n_inversion = n_total - n_correct
    lines.extend([
        "",
        f"**{n_correct}/{n_total}** tiles with correct direction. "
        f"**{n_inversion}** inversions.",
        f"C2 target: inversions <= 1.",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved: %s", output_path)


def write_correlation_vs_s2(
    variant_values: np.ndarray,
    s2_channels: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Write Spearman correlation of SE variant vs NIR, NDVI, NBR.

    Args:
        variant_values: Flat array of SE variant pixel values.
        s2_channels: Dict mapping channel name to flat array.
    """
    lines = [
        "# Correlation vs Sentinel-2 Channels",
        "",
        "| Channel | Spearman r | |r| | C1 Pass (|r| <= 0.5) |",
        "|---------|-----------|-----|----------------------|",
    ]
    for name in ["NIR", "NDVI", "NBR"]:
        key = name.lower()
        if key not in s2_channels or len(s2_channels[key]) == 0:
            lines.append(f"| {name} | N/A | N/A | N/A |")
            continue
        r, _ = spearmanr(variant_values, s2_channels[key])
        passed = "Yes" if abs(r) <= 0.5 else "**No**"
        lines.append(f"| {name} | {r:.4f} | {abs(r):.4f} | {passed} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# OID 169 membership (Approach 1)
# ---------------------------------------------------------------------------

def plot_oid169_membership(
    se_data: np.ndarray,
    centroids: np.ndarray,
    pixel_coords: list[tuple[float, float]],
    rgb: np.ndarray,
    output_path: Path,
) -> None:
    """Show which prototype fires inside OID 169's polygon.

    Renders per-prototype cosine maps and a winner-take-all membership map.
    """
    import matplotlib.path as mpath

    n_bands, h, w = se_data.shape
    k = centroids.shape[0]
    flat = se_data.reshape(n_bands, -1).T  # (n_pixels, 64)
    sims = flat @ centroids.T  # (n_pixels, k)
    sims_maps = sims.T.reshape(k, h, w)  # (k, h, w)
    winner = np.argmax(sims, axis=1).reshape(h, w)

    # Polygon mask
    poly_path = mpath.Path(pixel_coords)
    yy, xx = np.mgrid[0:h, 0:w]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    poly_mask = poly_path.contains_points(points).reshape(h, w)

    # Which prototype has highest mean cosine in polygon?
    polygon_means = []
    for ci in range(k):
        if poly_mask.sum() > 0:
            polygon_means.append(float(sims_maps[ci][poly_mask].mean()))
        else:
            polygon_means.append(float("nan"))

    best_proto = int(np.nanargmax(polygon_means))

    n_cols = min(k + 2, 6)
    n_rows = math.ceil((k + 2) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)

    # RGB
    ax = axes.flat[0]
    ax.imshow(rgb)
    ax.set_title("RGB", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    # Winner map
    ax = axes.flat[1]
    im = ax.imshow(winner, cmap="tab10", vmin=0, vmax=max(k - 1, 1))
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title("Winner-take-all", fontsize=10)
    add_polygon_overlay(ax, pixel_coords)

    # Per-prototype cosine
    for ci in range(k):
        ax = axes.flat[ci + 2]
        im = ax.imshow(sims_maps[ci], cmap="RdBu_r", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.75)
        star = " *" if ci == best_proto else ""
        ax.set_title(
            f"Proto {ci} (poly={polygon_means[ci]:.3f}){star}", fontsize=9,
        )
        add_polygon_overlay(ax, pixel_coords)

    # Hide unused axes
    for idx in range(k + 2, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"OID 169 Prototype Membership — Best: proto {best_proto}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Three-way comparison
# ---------------------------------------------------------------------------

def _hist_panel(
    ax: plt.Axes,
    sim_map: np.ndarray,
    polygon_mean: float,
    background_mode: float,
    title: str,
) -> None:
    """Histogram with polygon-mean and background-mode annotations."""
    valid = sim_map[np.isfinite(sim_map)]
    if valid.size == 0:
        ax.text(0.5, 0.5, "(empty)", transform=ax.transAxes, ha="center")
        return
    ax.hist(valid, bins=80, color="steelblue", alpha=0.7, log=True)
    ax.axvline(
        polygon_mean, color="red", linestyle="--", linewidth=1.5,
        label=f"polygon mean = {polygon_mean:.4f}",
    )
    ax.axvline(
        background_mode, color="black", linestyle=":", linewidth=1.2,
        label=f"background mode = {background_mode:.4f}",
    )
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("score", fontsize=8)
    ax.set_ylabel("pixel count (log)", fontsize=8)


def _sim_panel(
    ax: plt.Axes,
    sim_map: np.ndarray,
    pixel_coords: list[tuple[float, float]],
    title: str,
) -> None:
    """Spatial similarity panel with polygon overlay and per-panel vmin/vmax."""
    valid = sim_map[np.isfinite(sim_map)]
    if valid.size > 0:
        vmin, vmax = np.percentile(valid, [2, 98])
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -1, 1
    im = ax.imshow(sim_map, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    ax.set_title(title, fontsize=10)
    add_polygon_overlay(ax, pixel_coords)


def plot_three_way_comparison(
    rgb: np.ndarray,
    pixel_coords: list[tuple[float, float]],
    sim_original: np.ndarray,
    sim_a1: np.ndarray,
    sim_a2: np.ndarray,
    suptitle: str,
    output_path: Path,
) -> None:
    """2x4 comparison figure: RGB + 3 sim maps on top, histograms on bottom."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)

    # --- Top row: RGB + 3 similarity maps ---
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB (PlanetScope)", fontsize=10)
    add_polygon_overlay(axes[0, 0], pixel_coords)

    _sim_panel(axes[0, 1], sim_original, pixel_coords, "Original single-prototype SE-cosine")
    _sim_panel(axes[0, 2], sim_a1, pixel_coords, "Approach 1 — multi-prototype max-cos")
    _sim_panel(axes[0, 3], sim_a2, pixel_coords, "Approach 2 — contrastive")

    # --- Bottom row: RGB (reference) + 3 histograms ---
    axes[1, 0].imshow(rgb)
    axes[1, 0].set_title("RGB (reference)", fontsize=10)
    add_polygon_overlay(axes[1, 0], pixel_coords)

    for ax, sim, title in (
        (axes[1, 1], sim_original, "Histogram — original"),
        (axes[1, 2], sim_a1, "Histogram — A1 max-cos"),
        (axes[1, 3], sim_a2, "Histogram — A2 contrastive"),
    ):
        pm = compute_polygon_mean_sim(sim, pixel_coords)
        bm = compute_background_mode(sim)
        _hist_panel(ax, sim, pm, bm, title)

    for ax in axes[0, :]:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_summary_metrics(
    oids: list[int],
    c1: dict[str, dict[str, float]],
    c2: dict[str, list[tuple[int, float, float]]],
    dynamic_range: dict[str, list[np.ndarray]],
    output_path: Path,
) -> None:
    """Three-panel summary across variants.

    Args:
        oids: OIDs in the order used across C2 columns.
        c1: variant -> channel -> |r| on the pooled 7-tile pixel sample.
        c2: variant -> list of (oid, polygon_mean, background_mode).
        dynamic_range: variant -> list of per-tile valid-pixel 1-D arrays.
    """
    variants = list(c1.keys())
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel A — C1 bars
    channels = ["NIR", "NDVI", "NBR"]
    width = 0.25
    x = np.arange(len(channels))
    ax = axes[0]
    for i, v in enumerate(variants):
        vals = [c1[v].get(ch, float("nan")) for ch in channels]
        ax.bar(x + (i - 1) * width, vals, width, label=v)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="|r|=0.5 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("|Spearman r|")
    ax.set_title("C1 — correlation vs Sentinel-2 channels", fontsize=11)
    ax.legend(fontsize=8)

    # Panel B — polygon_mean minus background_mode (per OID, per variant)
    ax = axes[1]
    xb = np.arange(len(oids))
    for i, v in enumerate(variants):
        by_oid = {oid: (pm, bm) for oid, pm, bm in c2[v]}
        diffs = [by_oid[o][0] - by_oid[o][1] for o in oids]
        ax.bar(xb + (i - 1) * width, diffs, width, label=v)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(xb)
    ax.set_xticklabels([f"OID {o}" for o in oids], rotation=30, ha="right")
    ax.set_ylabel("polygon_mean − background_mode")
    ax.set_title("C2 — sign-consistency margin per test tile", fontsize=11)
    ax.legend(fontsize=8)

    # Panel C — dynamic range (boxplot of pooled valid pixels per variant)
    ax = axes[2]
    data = [np.concatenate(dynamic_range[v]) for v in variants]
    ax.boxplot(data, labels=variants, showfliers=False)
    ax.set_ylabel("score")
    ax.set_title("Dynamic range (pooled across 7 tiles)", fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _safe_abs_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Return |Spearman r|, or nan on failure."""
    try:
        r, _ = spearmanr(a, b)
        return float(abs(r)) if np.isfinite(r) else float("nan")
    except Exception:
        return float("nan")


def write_comparison_summary_md(
    oids: list[int],
    c1: dict[str, dict[str, float]],
    c2: dict[str, list[tuple[int, float, float]]],
    output_path: Path,
) -> None:
    """Markdown sibling of plot_summary_metrics, for quick text reference."""
    lines = [
        "# SE Variant Comparison — Summary",
        "",
        "## C1 — |Spearman r| vs Sentinel-2 (pooled 7-tile pixel sample)",
        "",
        "| Variant | NIR | NDVI | NBR |",
        "|---------|-----|------|-----|",
    ]
    for v, by_ch in c1.items():
        lines.append(
            f"| {v} | {by_ch.get('NIR', float('nan')):.3f}"
            f" | {by_ch.get('NDVI', float('nan')):.3f}"
            f" | {by_ch.get('NBR', float('nan')):.3f} |"
        )

    lines.extend([
        "",
        "## C2 — polygon_mean − background_mode (higher = stronger RTS contrast)",
        "",
        "| OID | " + " | ".join(c1.keys()) + " |",
        "|-----|" + "|".join(["-----"] * len(c1)) + "|",
    ])
    by_oid_per_variant = {
        v: {oid: (pm, bm) for oid, pm, bm in c2[v]} for v in c1.keys()
    }
    for oid in oids:
        row = [f"{oid}"]
        for v in c1.keys():
            pm, bm = by_oid_per_variant[v][oid]
            row.append(f"{pm - bm:+.4f}")
        lines.append("| " + " | ".join(row) + " |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2: SE-cosine variant evaluation.",
    )
    parser.add_argument(
        "--config", required=True, help="Path to se_investigation.yaml",
    )
    parser.add_argument(
        "--approach", required=True,
        choices=["approach1", "approach2", "compare", "all"],
        help=(
            "Which approach to run. 'compare' produces a three-way comparison "
            "(original SE-cosine + approach1 + approach2) across the 7 test "
            "tiles plus a summary figure; requires both approaches' "
            "centroids to be reachable (it re-runs the cheap centroid "
            "computation from the cached reference pool)."
        ),
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip reading cached data (still writes cache).",
    )
    parser.add_argument(
        "--rebuild-reference-pool", action="store_true",
        help=(
            "Ignore the cached reference pool and rebuild it via GEE. Use "
            "this to pick up the per-feature index fix in _sample_se_at_points."
        ),
    )
    return parser.parse_args()


def get_held_out_tile_ids(cache_dir: str) -> list[str]:
    """Load held-out tile IDs written by Phase 1."""
    path = Path(cache_dir) / "held_out_tile_ids.json"
    if not path.exists():
        logger.warning(
            "No held_out_tile_ids.json found. Run channel_correlation.py first."
            " Using empty held-out set.",
        )
        return []
    with open(path) as f:
        return json.load(f)


def get_positive_tile_ids() -> list[str]:
    """Get all tile IDs that have RTS pixels. Cached."""
    cache_path = CACHE_DIR / "positive_tile_ids.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    all_ids = _list_label_tiles()
    positive = []

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {
            pool.submit(_load_label_rts_coords, tid, 1): tid
            for tid in all_ids
        }
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Finding positive tiles", unit="tile",
        ):
            tid = futures[future]
            try:
                coords = future.result()
                if coords is not None:
                    positive.append(tid)
            except Exception:
                pass

    positive.sort()
    with open(cache_path, "w") as f:
        json.dump(positive, f, indent=2)
    logger.info("Found %d positive tiles", len(positive))
    return positive


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    se_cfg = config["se_variants"]
    run_a1 = args.approach in ("approach1", "all")
    run_a2 = args.approach in ("approach2", "all")
    run_compare = args.approach in ("compare", "all")
    force_rebuild = args.rebuild_reference_pool

    # Propagate cache flag
    se_sar_plot.USE_CACHE = not args.no_cache

    ensure_cache_dir()
    initialize_gee()
    se_col = get_se_collection()

    # --- Load 7 test tiles ---
    entries = load_mapping()
    logger.info("Loaded %d test polygon-tile pairs", len(entries))

    # --- Build reference pool ---
    held_out = get_held_out_tile_ids(config["paths"]["cache_dir"])
    positive_ids = get_positive_tile_ids()
    reference_ids = sorted(set(positive_ids) - set(held_out))
    logger.info(
        "Reference pool: %d tiles (%d positive - %d held-out)",
        len(reference_ids), len(positive_ids), len(held_out),
    )

    se_vectors, tile_ids, ref_rows, ref_cols = build_reference_pool(
        se_col, reference_ids,
        max_points=se_cfg["prototype_max_points"],
        seed=config["seed"],
        force_rebuild=force_rebuild,
    )

    # ===================================================================
    # Approach 1 — Multi-Prototype
    # ===================================================================
    pos_centroids = None
    neg_centroids = None
    best_k = None
    a1_sims_by_oid: dict[int, dict] = {}
    a2_sims_by_oid: dict[int, dict] = {}

    if run_a1:
        logger.info("=== Approach 1: Multi-Prototype ===")
        out_dir = Path(config["paths"]["variants_output"]) / "approach1_multiprototype"
        out_dir.mkdir(parents=True, exist_ok=True)

        # k-means sweep
        km_results = run_kmeans_sweep(
            se_vectors, se_cfg["k_sweep"], config["seed"],
        )
        best_k = select_best_k(km_results)
        pos_centroids = km_results[best_k]["centroids"]
        logger.info("Best k=%d (silhouette=%.4f)", best_k,
                     km_results[best_k]["silhouette"])

        # Threshold: 5th percentile of reference pool max-cosine
        ref_max_cos = (se_vectors @ pos_centroids.T).max(axis=1)
        threshold = float(np.percentile(ref_max_cos, 5))
        logger.info("Multi-prototype threshold T=%.4f", threshold)

        # --- Per-tile figures ---
        sign_results = []
        all_variant = []
        all_s2 = {"nir": [], "ndvi": [], "nbr": []}

        for entry in tqdm(entries, desc="Approach 1 tiles", unit="tile"):
            oid = entry["object_id"]
            tile_name = entry["tile_name"]

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

            se_data = fetch_se_tile(se_col, bbox, grid, tile_name)
            sim_map = score_tile_multiprototype(se_data, pos_centroids)

            polygon_mean = compute_polygon_mean_sim(sim_map, pixel_coords)
            bg_mode = compute_background_mode(sim_map)

            sign_results.append({
                "oid": oid,
                "polygon_mean": polygon_mean,
                "background_mode": bg_mode,
            })

            plot_variant_tile(
                rgb, sim_map, threshold, pixel_coords, polygon_mean,
                f"SE variant — Multi-Prototype (k={best_k}) \u2014 OID {oid} ({tile_name})",
                out_dir / f"oid{oid}_variant.png",
            )

            # Collect for correlation
            flat_sim = sim_map.ravel()
            all_variant.append(flat_sim)
            try:
                s2 = fetch_sentinel2_channels(bbox, grid)
                for ch in ("nir", "ndvi", "nbr"):
                    all_s2[ch].append(s2[ch].ravel())
            except Exception as exc:
                logger.warning("S2 fetch failed for %s: %s", tile_name, exc)

            # OID 169 membership
            if oid == 169:
                plot_oid169_membership(
                    se_data, pos_centroids, pixel_coords, rgb,
                    out_dir / "oid169_membership.png",
                )

        # Sign consistency
        write_sign_consistency(sign_results, out_dir / "sign_consistency.md")

        # Correlation vs S2
        if all_variant and all_s2["nir"]:
            variant_flat = np.concatenate(all_variant)
            s2_flat = {ch: np.concatenate(v) for ch, v in all_s2.items() if v}
            # Subsample for speed
            rng = np.random.RandomState(config["seed"])
            n = min(500000, len(variant_flat))
            idx = rng.choice(len(variant_flat), n, replace=False)
            write_correlation_vs_s2(
                variant_flat[idx],
                {ch: v[idx] for ch, v in s2_flat.items()},
                out_dir / "correlation_vs_s2.md",
            )

        # Prototype samples
        sil_scores = {k: r["silhouette"] for k, r in km_results.items()}
        plot_prototype_samples(
            se_vectors, tile_ids, ref_rows, ref_cols,
            pos_centroids, km_results[best_k]["labels"],
            sil_scores, best_k, config,
            out_dir / "prototype_samples.png",
        )

        logger.info("Approach 1 complete. Outputs in %s", out_dir)

    # ===================================================================
    # Approach 2 — Contrastive
    # ===================================================================
    if run_a2:
        logger.info("=== Approach 2: Contrastive ===")
        out_dir = Path(config["paths"]["variants_output"]) / "approach2_contrastive"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Need positive prototypes from Approach 1
        if pos_centroids is None:
            logger.info("Running Approach 1 k-means for positive prototypes...")
            km_results = run_kmeans_sweep(
                se_vectors, se_cfg["k_sweep"], config["seed"],
            )
            best_k = select_best_k(km_results)
            pos_centroids = km_results[best_k]["centroids"]

        # Build negative pool
        all_tile_ids = _list_label_tiles()
        neg_vectors = build_negative_pool(
            se_col, all_tile_ids,
            n_tiles=se_cfg["negative_sample_tiles"],
            max_points=se_cfg["negative_max_points"],
            seed=config["seed"],
        )

        # k-means on negative pool with same k
        logger.info("k-means on negative pool (k=%d)...", best_k)
        km_neg = KMeans(
            n_clusters=best_k, random_state=config["seed"],
            n_init=10, max_iter=300,
        )
        km_neg.fit(neg_vectors)
        neg_centroids = km_neg.cluster_centers_.copy()
        norms = np.linalg.norm(neg_centroids, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        neg_centroids = neg_centroids / norms

        # Threshold: 5th percentile of reference pool contrastive score
        ref_pos = (se_vectors @ pos_centroids.T).max(axis=1)
        ref_neg = (se_vectors @ neg_centroids.T).max(axis=1)
        ref_contrastive = ref_pos - ref_neg
        threshold = float(np.percentile(ref_contrastive, 5))
        logger.info("Contrastive threshold T=%.4f", threshold)

        # --- Per-tile figures ---
        sign_results = []
        all_variant = []
        all_s2 = {"nir": [], "ndvi": [], "nbr": []}

        for entry in tqdm(entries, desc="Approach 2 tiles", unit="tile"):
            oid = entry["object_id"]
            tile_name = entry["tile_name"]

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

            se_data = fetch_se_tile(se_col, bbox, grid, tile_name)
            sim_map = score_tile_contrastive(
                se_data, pos_centroids, neg_centroids,
            )

            polygon_mean = compute_polygon_mean_sim(sim_map, pixel_coords)
            bg_mode = compute_background_mode(sim_map)

            sign_results.append({
                "oid": oid,
                "polygon_mean": polygon_mean,
                "background_mode": bg_mode,
            })

            plot_variant_tile(
                rgb, sim_map, threshold, pixel_coords, polygon_mean,
                f"SE variant — Contrastive (k={best_k}) \u2014 OID {oid} ({tile_name})",
                out_dir / f"oid{oid}_variant.png",
            )

            # Collect for correlation
            flat_sim = sim_map.ravel()
            all_variant.append(flat_sim)
            try:
                s2 = fetch_sentinel2_channels(bbox, grid)
                for ch in ("nir", "ndvi", "nbr"):
                    all_s2[ch].append(s2[ch].ravel())
            except Exception as exc:
                logger.warning("S2 fetch failed for %s: %s", tile_name, exc)

        # Sign consistency
        write_sign_consistency(sign_results, out_dir / "sign_consistency.md")

        # Correlation vs S2
        if all_variant and all_s2["nir"]:
            variant_flat = np.concatenate(all_variant)
            s2_flat = {ch: np.concatenate(v) for ch, v in all_s2.items() if v}
            rng = np.random.RandomState(config["seed"])
            n = min(500000, len(variant_flat))
            idx = rng.choice(len(variant_flat), n, replace=False)
            write_correlation_vs_s2(
                variant_flat[idx],
                {ch: v[idx] for ch, v in s2_flat.items()},
                out_dir / "correlation_vs_s2.md",
            )

        logger.info("Approach 2 complete. Outputs in %s", out_dir)

    # ===================================================================
    # Three-way comparison — original SE-cosine + Approach 1 + Approach 2
    # ===================================================================
    if run_compare:
        logger.info("=== Comparison: three-way ===")
        out_dir = Path(config["paths"]["comparison_output"])
        out_dir.mkdir(parents=True, exist_ok=True)

        # Lazily compute centroids if the user invoked `compare` alone.
        if pos_centroids is None:
            logger.info("Computing positive centroids from reference pool...")
            km_results = run_kmeans_sweep(
                se_vectors, se_cfg["k_sweep"], config["seed"],
            )
            best_k = select_best_k(km_results)
            pos_centroids = km_results[best_k]["centroids"]
        if neg_centroids is None:
            logger.info("Building negative pool + centroids...")
            all_tile_ids = _list_label_tiles()
            neg_vectors = build_negative_pool(
                se_col, all_tile_ids,
                n_tiles=se_cfg["negative_sample_tiles"],
                max_points=se_cfg["negative_max_points"],
                seed=config["seed"],
            )
            km_neg = KMeans(
                n_clusters=best_k, random_state=config["seed"],
                n_init=10, max_iter=300,
            )
            km_neg.fit(neg_vectors)
            neg_centroids = km_neg.cluster_centers_.copy()
            norms = np.linalg.norm(neg_centroids, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            neg_centroids = neg_centroids / norms

        # Single-prototype baseline = mean of reference pool, re-normalized.
        proto_mean = se_vectors.mean(axis=0)
        proto = proto_mean / max(np.linalg.norm(proto_mean), 1e-12)

        variants = ["original", "approach1", "approach2"]
        c1_acc: dict[str, dict[str, list[np.ndarray]]] = {
            v: {"sim": [], "nir": [], "ndvi": [], "nbr": []} for v in variants
        }
        c2_acc: dict[str, list[tuple[int, float, float]]] = {v: [] for v in variants}
        range_acc: dict[str, list[np.ndarray]] = {v: [] for v in variants}
        oid_order: list[int] = []

        for entry in tqdm(entries, desc="Three-way tiles", unit="tile"):
            oid = entry["object_id"]
            tile_name = entry["tile_name"]
            oid_order.append(oid)

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

            se_data = fetch_se_tile(se_col, bbox, grid, tile_name)
            sim_orig = score_tile_prototype(se_data, proto)
            sim_a1 = score_tile_multiprototype(se_data, pos_centroids)
            sim_a2 = score_tile_contrastive(se_data, pos_centroids, neg_centroids)

            plot_three_way_comparison(
                rgb, pixel_coords,
                sim_orig, sim_a1, sim_a2,
                f"SE variant comparison \u2014 OID {oid} ({tile_name})",
                out_dir / f"oid{oid}_three_way.png",
            )

            # Per-variant aggregations for the summary panels
            try:
                s2 = fetch_sentinel2_channels(bbox, grid)
                nir = s2["nir"].ravel()
                ndvi = s2["ndvi"].ravel()
                nbr = s2["nbr"].ravel()
            except Exception as exc:
                logger.warning("S2 fetch failed for %s: %s", tile_name, exc)
                nir = ndvi = nbr = None

            for v, sim in (
                ("original", sim_orig), ("approach1", sim_a1), ("approach2", sim_a2),
            ):
                pm = compute_polygon_mean_sim(sim, pixel_coords)
                bm = compute_background_mode(sim)
                c2_acc[v].append((oid, pm, bm))

                flat = sim.ravel()
                c1_acc[v]["sim"].append(flat)
                if nir is not None:
                    c1_acc[v]["nir"].append(nir)
                    c1_acc[v]["ndvi"].append(ndvi)
                    c1_acc[v]["nbr"].append(nbr)

                valid = sim[np.isfinite(sim)]
                if valid.size > 0:
                    range_acc[v].append(valid)

        # Compute C1 on pooled pixels (subsampled for speed).
        rng = np.random.RandomState(config["seed"])
        c1_summary: dict[str, dict[str, float]] = {}
        for v in variants:
            sim_flat = np.concatenate(c1_acc[v]["sim"]) if c1_acc[v]["sim"] else np.empty(0)
            nir_flat = np.concatenate(c1_acc[v]["nir"]) if c1_acc[v]["nir"] else np.empty(0)
            ndvi_flat = np.concatenate(c1_acc[v]["ndvi"]) if c1_acc[v]["ndvi"] else np.empty(0)
            nbr_flat = np.concatenate(c1_acc[v]["nbr"]) if c1_acc[v]["nbr"] else np.empty(0)
            n = min(500000, len(sim_flat), len(nir_flat))
            if n == 0:
                c1_summary[v] = {"NIR": float("nan"), "NDVI": float("nan"), "NBR": float("nan")}
                continue
            idx = rng.choice(len(sim_flat), n, replace=False)
            c1_summary[v] = {
                "NIR": _safe_abs_spearman(sim_flat[idx], nir_flat[idx]),
                "NDVI": _safe_abs_spearman(sim_flat[idx], ndvi_flat[idx]),
                "NBR": _safe_abs_spearman(sim_flat[idx], nbr_flat[idx]),
            }

        plot_summary_metrics(
            oid_order, c1_summary, c2_acc, range_acc,
            out_dir / "summary.png",
        )
        write_comparison_summary_md(
            oid_order, c1_summary, c2_acc,
            out_dir / "summary.md",
        )
        logger.info("Comparison complete. Outputs in %s", out_dir)

    logger.info("Phase 2 done.")


if __name__ == "__main__":
    main()
