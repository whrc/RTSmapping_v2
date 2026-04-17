"""
Phase 1 — Channel correlation analysis for SE investigation.

Quantifies pairwise Spearman/Pearson correlations across 15 channels on a
held-out set of tiles to assess how redundant SE-cosine and SE global-PCA
components are with existing Sentinel-2 auxiliary channels.

15 channels:
  PlanetScope RGB (R, G, B)
  Sentinel-2 spectral (NIR, SWIR)
  Sentinel-2 indices (NBR, NDVI, NDWI, NDMI)
  Sentinel-2 tasseled cap (TCB, TCW)
  SE reference-anchored (SE-cosine)
  SE global PCA (PC1, PC2, PC3)

Three correlation regimes: all pixels, positive (label=1), near-boundary.

Outputs to plots/extra_channel_vis/correlation/:
  heatmap_all.png, heatmap_positive.png, heatmap_boundary.png,
  heatmap_all_pearson.png, cluster_dendrogram.png,
  distance_from_ndvi.png, distance_from_nir.png, interpretations.md

Usage:
  python scripts/channel_correlation.py --config configs/se_investigation.yaml
  python scripts/channel_correlation.py --config configs/se_investigation.yaml --n-tiles 3  # dry run
"""

import argparse
import io
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import seaborn as sns
import yaml
from google.cloud import storage
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.stats import spearmanr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add vis directory to path for reuse of existing functions
# ---------------------------------------------------------------------------
_VIS_DIR = Path(__file__).resolve().parent.parent / "plots" / "extra_channel_vis"
sys.path.insert(0, str(_VIS_DIR))

from extra_channel_plot import (  # noqa: E402
    TILE_SIZE,
    fetch_pixels,
    fetch_sentinel2_channels,
    initialize_gee,
    make_gee_grid,
    make_tile_bbox,
)

import se_sar_plot  # noqa: E402  — need module ref to set USE_CACHE
from se_sar_plot import (  # noqa: E402
    SEED,
    SE_YEAR,
    _list_label_tiles,
    _load_global_pca_cache,
    _sample_se_at_points,
    build_rts_prototype,
    ensure_cache_dir,
    fetch_se_tile,
    fit_global_pca,
    get_se_collection,
    load_cache,
    project_global_pca,
    sample_arctic_land_se,
    save_cache,
    score_tile_prototype,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel names (fixed order, 15 total)
# ---------------------------------------------------------------------------
CHANNEL_NAMES = [
    "R", "G", "B",
    "NIR", "SWIR",
    "NBR", "NDVI", "NDWI", "NDMI",
    "TCB", "TCW",
    "SE-cosine",
    "PC1", "PC2", "PC3",
]
N_CHANNELS = len(CHANNEL_NAMES)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def load_label_from_gcs(
    tile_id: str, bucket_name: str, prefix: str,
) -> tuple[np.ndarray, rasterio.Affine]:
    """Download label GeoTIFF from GCS, return (label_array, affine_transform).

    Args:
        tile_id: Tile identifier (no extension).
        bucket_name: GCS bucket name.
        prefix: Blob prefix before tile_id.

    Returns:
        label: (512, 512) uint8 array with values {0, 1, 255}.
        transform: Rasterio affine transform for the tile.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/{tile_id}.tif")
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    with rasterio.open(buf) as ds:
        return ds.read(1), ds.transform


def load_rgb_from_gcs(
    tile_id: str, bucket_name: str, prefix: str,
) -> np.ndarray:
    """Download PlanetScope RGB GeoTIFF from GCS.

    Returns:
        (512, 512, 3) float32 array in [0, 1].
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/{tile_id}.tif")
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    with rasterio.open(buf) as ds:
        data = ds.read()  # (3, 512, 512) uint8
    return np.moveaxis(data, 0, -1).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Boundary mask
# ---------------------------------------------------------------------------

def make_boundary_mask(label: np.ndarray, distance_px: int = 10) -> np.ndarray:
    """Create a ring mask around RTS polygon edges.

    Dilate label==1 by distance_px, subtract erosion by distance_px.
    The result is a band of pixels straddling the polygon boundary.

    Args:
        label: (H, W) uint8 label array.
        distance_px: Width of the boundary band in pixels.

    Returns:
        (H, W) bool mask.
    """
    rts = label == 1
    struct = np.ones((3, 3), dtype=bool)
    dilated = binary_dilation(rts, structure=struct, iterations=distance_px)
    eroded = binary_erosion(rts, structure=struct, iterations=distance_px)
    return dilated & ~eroded


# ---------------------------------------------------------------------------
# Tile selection
# ---------------------------------------------------------------------------

def _check_tile_has_rts(tile_id: str) -> tuple[str, bool]:
    """Check if a tile has any RTS pixels. Returns (tile_id, has_rts)."""
    from se_sar_plot import _load_label_rts_coords  # noqa: E402
    try:
        coords = _load_label_rts_coords(tile_id, max_per_tile=1)
        return tile_id, coords is not None
    except Exception:
        return tile_id, False


def select_held_out_tiles(
    config: dict, n_override: int | None = None,
) -> list[str]:
    """Select held-out tiles from the positive tile pool.

    Checks cache first; if missing, discovers positive tiles from GCS
    and randomly samples n_held_out of them.

    Args:
        config: Full config dict.
        n_override: Override n_held_out from config (for dry runs).

    Returns:
        Sorted list of held-out tile IDs.
    """
    cache_dir = Path(config["paths"]["cache_dir"])
    cache_file = cache_dir / "held_out_tile_ids.json"
    pos_cache_file = cache_dir / "positive_tile_ids.json"
    seed = config["seed"]
    n_held_out = n_override or config["correlation"]["n_held_out"]

    # Check cache
    if cache_file.exists() and n_override is None:
        with open(cache_file) as f:
            cached = json.load(f)
        if len(cached) == n_held_out:
            logger.info("Loaded %d held-out tile IDs from cache", len(cached))
            return cached

    # Reuse positive-tile cache if present (shared with se_variants.py)
    if pos_cache_file.exists():
        with open(pos_cache_file) as f:
            positive_ids = sorted(json.load(f))
        logger.info(
            "Loaded %d positive tile IDs from cache", len(positive_ids),
        )
    else:
        logger.info("Discovering label tiles on GCS...")
        all_tile_ids = _list_label_tiles()
        logger.info("Found %d label tiles", len(all_tile_ids))

        logger.info("Checking which tiles have RTS pixels (parallel)...")
        positive_ids = []
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = {
                pool.submit(_check_tile_has_rts, tid): tid
                for tid in all_tile_ids
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Filtering positive tiles", unit="tile",
            ):
                tid, has_rts = future.result()
                if has_rts:
                    positive_ids.append(tid)

        positive_ids.sort()
        logger.info("Found %d positive tiles", len(positive_ids))
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(pos_cache_file, "w") as f:
            json.dump(positive_ids, f, indent=2)

    # Random sample
    rng = np.random.RandomState(seed)
    n = min(n_held_out, len(positive_ids))
    held_out = sorted(rng.choice(positive_ids, n, replace=False).tolist())
    logger.info("Selected %d held-out tiles", len(held_out))

    # Cache (only if not overridden)
    if n_override is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(held_out, f, indent=2)
        logger.info("Cached held-out tile IDs to %s", cache_file)

    return held_out


# ---------------------------------------------------------------------------
# Per-tile channel collection
# ---------------------------------------------------------------------------

def collect_tile_channels(
    tile_id: str,
    se_col,
    prototype: np.ndarray,
    global_pca,
    config: dict,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Collect 15 channels + label for one tile.

    Args:
        tile_id: Tile identifier.
        se_col: GEE SE ImageCollection.
        prototype: (64,) unit vector.
        global_pca: Fitted PCA(3) model.
        config: Full config dict.

    Returns:
        (channels, label) where channels is (15, 512, 512) float32
        and label is (512, 512) uint8. Returns None on failure.
    """
    gcs = config["gcs"]
    try:
        # 1. Label + affine transform
        label, transform = load_label_from_gcs(
            tile_id, gcs["bucket"], gcs["labels_prefix"],
        )

        # 2. GEE grid from affine
        origin_x = transform.c
        origin_y = transform.f
        pixel_scale = transform.a
        grid = make_gee_grid(origin_x, origin_y, pixel_scale)
        bbox = make_tile_bbox(origin_x, origin_y, pixel_scale)

        # 3. PlanetScope RGB
        rgb = load_rgb_from_gcs(tile_id, gcs["bucket"], gcs["rgb_prefix"])

        # 4. Sentinel-2
        s2 = fetch_sentinel2_channels(bbox, grid)

        # 5. SE
        se_data = fetch_se_tile(se_col, bbox, grid, tile_id)

        # 6. SE-cosine
        sim_map = score_tile_prototype(se_data, prototype)

        # 7. Global PCA
        _, global_pcs, _ = project_global_pca(se_data, global_pca)

        # 8. Stack 15 channels
        channels = np.stack([
            rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2],
            s2["nir"], s2["swir"],
            s2["nbr"], s2["ndvi"], s2["ndwi"], s2["ndmi"],
            s2["tcb"], s2["tcw"],
            sim_map,
            global_pcs[0], global_pcs[1], global_pcs[2],
        ], axis=0).astype(np.float32)

        return channels, label

    except Exception as exc:
        logger.warning("Failed to collect tile %s: %s", tile_id, exc)
        return None


# ---------------------------------------------------------------------------
# Pixel extraction
# ---------------------------------------------------------------------------

def extract_regime_pixels(
    channels: np.ndarray,
    label: np.ndarray,
    boundary_distance_px: int,
) -> dict[str, np.ndarray]:
    """Extract valid pixels for each correlation regime.

    Args:
        channels: (15, 512, 512) float32.
        label: (512, 512) uint8.
        boundary_distance_px: Width of boundary band.

    Returns:
        Dict mapping regime name to (N, 15) float32 arrays.
    """
    valid = ~np.isnan(channels).any(axis=0) & (label != 255)

    regimes = {
        "all": valid,
        "positive": valid & (label == 1),
        "boundary": valid & make_boundary_mask(label, boundary_distance_px),
    }

    result = {}
    for name, mask in regimes.items():
        if mask.sum() > 0:
            result[name] = channels[:, mask].T  # (N, 15)
        else:
            result[name] = np.empty((0, N_CHANNELS), dtype=np.float32)

    return result


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def compute_correlations(
    data: np.ndarray, method: str = "spearman",
) -> np.ndarray:
    """Compute pairwise correlation matrix.

    Args:
        data: (N, 15) array of pixel values.
        method: "spearman" or "pearson".

    Returns:
        (15, 15) correlation matrix.
    """
    if data.shape[0] < 3:
        return np.full((N_CHANNELS, N_CHANNELS), np.nan)

    if method == "spearman":
        corr, _ = spearmanr(data, axis=0)
        # spearmanr returns a scalar if only 2 columns; ensure matrix
        if np.ndim(corr) == 0:
            corr = np.array([[1.0, corr], [corr, 1.0]])
        return corr
    # Pearson: mask zero-variance columns so they produce NaN cells instead of
    # poisoning the whole matrix via divide-by-zero in np.corrcoef.
    stds = data.std(axis=0)
    valid = stds > 0
    n = data.shape[1]
    corr = np.full((n, n), np.nan)
    if valid.sum() >= 2:
        sub = np.corrcoef(data[:, valid], rowvar=False)
        idx = np.where(valid)[0]
        corr[np.ix_(idx, idx)] = sub
    np.fill_diagonal(corr, 1.0)
    return corr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(
    corr: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Plot annotated correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.zeros_like(corr, dtype=bool)
    sns.heatmap(
        corr,
        mask=mask,
        xticklabels=CHANNEL_NAMES,
        yticklabels=CHANNEL_NAMES,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "correlation"},
        ax=ax,
        annot_kws={"fontsize": 7},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_dendrogram(
    corr: np.ndarray,
    output_path: Path,
) -> None:
    """Plot hierarchical clustering dendrogram from correlation matrix."""
    # Distance = 1 - |r|
    dist = 1 - np.abs(corr)
    # Convert to condensed form
    n = dist.shape[0]
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist[i, j])
    condensed = np.array(condensed)

    Z = linkage(condensed, method="average")

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(
        Z,
        labels=CHANNEL_NAMES,
        leaf_rotation=45,
        leaf_font_size=10,
        ax=ax,
        color_threshold=0.5,
    )
    ax.set_ylabel("Distance (1 - |r|)", fontsize=12)
    ax.set_title(
        "Channel Clustering (average linkage, Spearman)",
        fontsize=14, fontweight="bold",
    )
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="|r| = 0.5")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def plot_distance_from_channel(
    corr: np.ndarray,
    ref_channel: str,
    output_path: Path,
) -> None:
    """Bar chart of |r| distance from a reference channel."""
    ref_idx = CHANNEL_NAMES.index(ref_channel)
    distances = 1 - np.abs(corr[ref_idx])

    # Sort descending (most independent first)
    order = np.argsort(-distances)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [CHANNEL_NAMES[i] for i in order]
    vals = distances[order]
    colors = ["steelblue" if n not in ("SE-cosine", "PC1", "PC2", "PC3")
              else "darkorange" for n in names]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel(f"1 - |r| vs {ref_channel}")
    ax.set_title(
        f"Channel Independence from {ref_channel} (Spearman)",
        fontsize=14, fontweight="bold",
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="|r| = 0.5")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Interpretations table
# ---------------------------------------------------------------------------

def generate_interpretations_md(
    corr_matrices: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Auto-generate interpretations.md for high-correlation pairs.

    Lists all pairs with |r| > 0.7 in any regime.
    """
    rows: list[tuple[str, str, float, str]] = []

    for regime, corr in corr_matrices.items():
        if np.any(np.isnan(corr)):
            continue
        for i in range(N_CHANNELS):
            for j in range(i + 1, N_CHANNELS):
                r = corr[i, j]
                if abs(r) > 0.7:
                    rows.append((CHANNEL_NAMES[i], CHANNEL_NAMES[j], r, regime))

    # Deduplicate by pair, keeping the highest |r|
    pair_best: dict[tuple[str, str], tuple[float, str]] = {}
    for a, b, r, regime in rows:
        key = (a, b)
        if key not in pair_best or abs(r) > abs(pair_best[key][0]):
            pair_best[key] = (r, regime)

    lines = [
        "# Channel Correlation Interpretations",
        "",
        "Pairs with |r| > 0.7 in at least one regime.",
        "Fill in the **Physical Reason** column.",
        "",
        "| Channel A | Channel B | |r| | Regime | Physical Reason |",
        "|-----------|-----------|-----|--------|-----------------|",
    ]
    for (a, b), (r, regime) in sorted(
        pair_best.items(), key=lambda x: -abs(x[1][0]),
    ):
        lines.append(f"| {a} | {b} | {abs(r):.3f} | {regime} | |")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1: channel correlation analysis for SE investigation.",
    )
    parser.add_argument(
        "--config", required=True, help="Path to se_investigation.yaml",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip reading cached data (still writes cache).",
    )
    parser.add_argument(
        "--n-tiles", type=int, default=None,
        help="Override n_held_out for dry-run testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # Propagate cache flag to se_sar_plot module
    se_sar_plot.USE_CACHE = not args.no_cache

    ensure_cache_dir()
    output_dir = Path(config["paths"]["correlation_output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Init GEE ---
    initialize_gee()
    se_col = get_se_collection()

    # --- Precomputation (cached) ---
    logger.info("--- Prototype construction ---")
    prototype, threshold, rts_cosine_sims, proto_meta = build_rts_prototype(
        se_col,
    )

    logger.info("--- Global PCA ---")
    global_pca = _load_global_pca_cache()
    if global_pca is None:
        global_sample = sample_arctic_land_se(se_col)
        global_pca = fit_global_pca(global_sample)

    # --- Tile selection ---
    held_out = select_held_out_tiles(config, n_override=args.n_tiles)
    logger.info("Processing %d held-out tiles", len(held_out))

    # --- Collect pixels per regime ---
    boundary_px = config["correlation"]["boundary_distance_px"]
    max_px = config["correlation"]["max_pixels_per_regime"]

    accum: dict[str, list[np.ndarray]] = {
        "all": [], "positive": [], "boundary": [],
    }

    for i, tile_id in enumerate(
        tqdm(held_out, desc="Collecting channels", unit="tile"), 1,
    ):
        logger.info("[%d/%d] Tile %s", i, len(held_out), tile_id)
        t0 = time.time()

        result = collect_tile_channels(
            tile_id, se_col, prototype, global_pca, config,
        )
        if result is None:
            continue

        channels, label = result
        regime_pixels = extract_regime_pixels(channels, label, boundary_px)

        for regime, pixels in regime_pixels.items():
            if pixels.shape[0] > 0:
                accum[regime].append(pixels)

        logger.info("  Tile %s done in %.1fs", tile_id, time.time() - t0)

    # --- Concatenate and subsample ---
    rng = np.random.RandomState(config["seed"])
    regime_data: dict[str, np.ndarray] = {}

    for regime, pixel_list in accum.items():
        if not pixel_list:
            logger.warning("No pixels for regime '%s'", regime)
            regime_data[regime] = np.empty((0, N_CHANNELS), dtype=np.float32)
            continue

        data = np.concatenate(pixel_list, axis=0)
        logger.info(
            "Regime '%s': %d pixels across %d tiles",
            regime, data.shape[0], len(pixel_list),
        )

        if data.shape[0] > max_px:
            idx = rng.choice(data.shape[0], max_px, replace=False)
            data = data[idx]
            logger.info("  Subsampled to %d pixels", max_px)

        regime_data[regime] = data

    # --- Compute correlations ---
    logger.info("Computing correlations...")
    spearman_matrices: dict[str, np.ndarray] = {}
    for regime, data in regime_data.items():
        spearman_matrices[regime] = compute_correlations(data, "spearman")
        logger.info(
            "  %s: Spearman done (%d pixels)", regime, data.shape[0],
        )

    pearson_all = compute_correlations(regime_data["all"], "pearson")

    # --- Generate plots ---
    logger.info("Generating plots...")

    # Spearman heatmaps
    for regime, corr in spearman_matrices.items():
        if np.any(np.isnan(corr)):
            logger.warning("Skipping heatmap for '%s' (NaN values)", regime)
            continue
        plot_heatmap(
            corr,
            f"Spearman Correlation — {regime} pixels",
            output_dir / f"heatmap_{regime}.png",
        )

    # Pearson heatmap: allow NaN cells (zero-variance columns); skip only if
    # the full matrix is unusable.
    if np.any(~np.isnan(pearson_all)):
        n_nan = int(np.isnan(pearson_all).sum())
        if n_nan:
            logger.warning(
                "Pearson heatmap has %d NaN cells (zero-variance cols); "
                "rendering with NaN masked",
                n_nan,
            )
        plot_heatmap(
            pearson_all,
            "Pearson Correlation — all pixels",
            output_dir / "heatmap_all_pearson.png",
        )

    # Dendrogram (from all-pixels Spearman)
    corr_all = spearman_matrices.get("all")
    if corr_all is not None and not np.any(np.isnan(corr_all)):
        plot_dendrogram(corr_all, output_dir / "cluster_dendrogram.png")

        # Distance bar charts
        plot_distance_from_channel(
            corr_all, "NDVI", output_dir / "distance_from_ndvi.png",
        )
        plot_distance_from_channel(
            corr_all, "NIR", output_dir / "distance_from_nir.png",
        )

    # Interpretations table
    generate_interpretations_md(spearman_matrices, output_dir / "interpretations.md")

    logger.info("Phase 1 complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
