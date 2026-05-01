"""Matplotlib figures logged per validation.

Per plan Step 5:
    - prediction_preview_grid: fixed 3 positive + 3 negative val tiles,
      3-column strip each (RGB | GT overlay | predicted probability heatmap).
    - pr_curves_at_ratios: PR curves for the three prevalence ratios.
    - probability_histogram: log-scale histogram of predicted probabilities
      (catches mode collapse and saturation).
    - confusion_matrix_pixel: pixel-level confusion at threshold 0.5, on
      a subsampled 1:500 neg-vs-pos pixel pool for readability.

Each function takes its inputs + an output path, writes a PNG, returns the
path. The caller (scripts/train.py) logs the PNG as an MLflow artifact and
applies the keep-last-N rotation per figure type (plan risk #19).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display required

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preview grid
# ---------------------------------------------------------------------------


def _denormalize_rgb(
    rgb_chw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    max_value: int = 255,
) -> np.ndarray:
    """Invert mean/std normalization and clip to [0, max_value] uint8.

    Assumes the source dataset is uint8 (PlanetScope basemap RGB). For higher
    bit-depth or non-RGB previews, pass `max_value` explicitly.

    Args:
        rgb_chw: (3, H, W) float32 — normalised tensor as stored in the batch.
        mean, std: length-3 arrays of the training stats.
        max_value: upper clip value matching the source dataset's bit-depth.
    """
    denorm = rgb_chw * std[:, None, None] + mean[:, None, None]
    return np.clip(denorm.transpose(1, 2, 0), 0, max_value).astype(np.uint8)


def prediction_preview_grid(
    tiles: list[dict],
    mean: np.ndarray,
    std: np.ndarray,
    out_path: Path,
    *,
    threshold: float = 0.5,
    ignore_index: int = 255,
) -> Path:
    """Render one row per tile with [RGB | GT overlay | prob heatmap].

    Args:
        tiles: List of dicts with keys:
            'tile_id' (str), 'image' (3, H, W) float32 normalised,
            'label' (H, W) int {0,1,255}, 'prob' (H, W) float32 in [0, 1].
        mean, std: RGB normalization stats (length 3) for denormalising image.
        out_path: PNG path to write.
        threshold: Binary decision threshold for overlay (visual only; metrics
            already live elsewhere).
        ignore_index: Label value to render as grey.
    """
    n = len(tiles)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for i, t in enumerate(tiles):
        rgb = _denormalize_rgb(t["image"][:3], mean, std)
        label = t["label"]
        prob = t["prob"]

        ax = axes[i][0]
        ax.imshow(rgb)
        ax.set_title(f"{t['tile_id']} (RGB)", fontsize=8)
        ax.axis("off")

        ax = axes[i][1]
        ax.imshow(rgb)
        # GT overlay: red = positive, grey = ignore. Set RGB channels per
        # class so ignore renders neutral grey rather than transparent red.
        gt_positive = (label == 1).astype(np.float32)
        gt_ignore = (label == ignore_index).astype(np.float32)
        overlay = np.zeros((*label.shape, 4), dtype=np.float32)
        overlay[..., 0] = gt_positive + 0.5 * gt_ignore     # R
        overlay[..., 1] = 0.5 * gt_ignore                    # G
        overlay[..., 2] = 0.5 * gt_ignore                    # B
        overlay[..., 3] = 0.5 * gt_positive + 0.4 * gt_ignore
        ax.imshow(overlay)
        ax.set_title("GT (red=pos, grey=ignore)", fontsize=8)
        ax.axis("off")

        ax = axes[i][2]
        im = ax.imshow(prob, vmin=0.0, vmax=1.0, cmap="viridis")
        # Contour at threshold to mark the binary decision boundary.
        if prob.max() > threshold:
            ax.contour(prob, levels=[threshold], colors="white", linewidths=0.5)
        ax.set_title(f"pred prob (thresh={threshold})", fontsize=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# PR curves
# ---------------------------------------------------------------------------


def pr_curves_at_ratios(
    logits_by_ratio: dict[int, tuple[np.ndarray, np.ndarray]],
    out_path: Path,
) -> Path:
    """Plot PR curves for each 1:N ratio on a single axes.

    Args:
        logits_by_ratio: {N: (logits_flat, labels_flat)} for each ratio.
        out_path: PNG path.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for ratio, (logits, labels) in sorted(logits_by_ratio.items()):
        if labels.max() == 0:
            continue
        prec, rec, _ = precision_recall_curve(labels, logits)
        ax.plot(rec, prec, label=f"1:{ratio}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Val-Realistic PR curves (negative-subsampled)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Probability histogram
# ---------------------------------------------------------------------------


def probability_histogram(
    probs_flat: np.ndarray,
    out_path: Path,
    *,
    bins: int = 50,
) -> Path:
    """Log-scale histogram of predicted probabilities across valid pixels.

    Mode collapse shows as everything piled at 0 or 1. Healthy training is
    bimodal at 0 and 1 with a sparse middle.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs_flat, bins=bins, range=(0.0, 1.0), log=True)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Pixel count (log)")
    ax.set_title("Prediction probability histogram (valid pixels)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def confusion_matrix_pixel(
    tp: int, fp: int, fn: int, tn: int,
    out_path: Path,
    *,
    subsample_ratio: int = 500,
) -> Path:
    """Render a pixel-level confusion matrix at threshold 0.5.

    Args:
        tp, fp, fn, tn: Raw pixel counts.
        out_path: PNG path.
        subsample_ratio: Displayed negatives are scaled to a 1:subsample_ratio
            pos:neg ratio so the matrix is legible; the counts are scaled
            proportionally and rounded.
    """
    if tp + fn > 0:
        target_neg = subsample_ratio * (tp + fn)
        actual_neg = max(1, fp + tn)
        scale = target_neg / actual_neg
        fp_s = int(round(fp * scale))
        tn_s = int(round(tn * scale))
    else:
        fp_s, tn_s = fp, tn

    mat = np.array([[tp, fn], [fp_s, tn_s]], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred=1", "pred=0"])
    ax.set_yticklabels(["gt=1", "gt=0"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(mat[i, j]):,}", ha="center", va="center",
                    color="black" if mat[i, j] < mat.max() / 2 else "white")
    ax.set_title(f"Pixel confusion (neg subsampled to 1:{subsample_ratio})")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Preview tile selection (pass 1 heuristic)
# ---------------------------------------------------------------------------


def pick_preview_tiles_pass1(
    metadata,  # pd.DataFrame with columns Tile_id, TrainClass, bbox centroid
    val_tile_ids: list[str],
    n_positive: int = 3,
    n_negative: int = 3,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Select fixed preview tiles by heuristic (plan risk #4, pass 1).

    Positives: 3 tiles from the 25-75th percentile of positive-pixel fraction.
    Negatives: 3 tiles from the 25-75th percentile of distance-to-nearest-
        positive-centroid (ambiguous tundra near known RTS).

    Pass 2 (post-baseline) replaces these with "interesting failure" tiles
    via scripts/pick_preview_tiles.py. See plan Step 5.

    Args:
        metadata: The repo's metadata.csv as a DataFrame, indexed by Tile_id.
        val_tile_ids: Tile IDs in the validation split.
        n_positive: Target positive-tile count.
        n_negative: Target negative-tile count.
        seed: RNG seed for deterministic selection within the percentile band.

    Returns:
        {"positive": [...], "negative": [...]}
    """
    md = metadata.loc[val_tile_ids].copy()
    rng = np.random.default_rng(seed)

    pos = md[md["TrainClass"] == "Positive"]
    # Per-tile positive-pixel fraction is typically stored as "Pos_frac" or
    # computed from the label raster at split creation; fall back to 1.0 if
    # absent, which effectively ranks by whatever column is available.
    if "Pos_frac" in pos.columns:
        pos_sorted = pos.sort_values("Pos_frac")
    else:
        pos_sorted = pos.sort_values(pos.columns[0])
    lo, hi = int(len(pos_sorted) * 0.25), int(len(pos_sorted) * 0.75)
    mid_pos = pos_sorted.iloc[lo:hi] if hi > lo else pos_sorted
    picked_pos = (
        rng.choice(mid_pos.index.to_numpy(), size=min(n_positive, len(mid_pos)), replace=False)
        if len(mid_pos) > 0 else np.array([], dtype=object)
    )

    neg = md[md["TrainClass"] != "Positive"]
    # Distance-to-nearest-positive: if centroid columns exist, compute; else random
    # selection from the negative pool.
    if {"centroid_x", "centroid_y"}.issubset(neg.columns) and len(pos) > 0:
        pos_xy = pos[["centroid_x", "centroid_y"]].to_numpy()
        neg_xy = neg[["centroid_x", "centroid_y"]].to_numpy()
        # (n_neg, n_pos) distance matrix -> min per row
        d = np.linalg.norm(neg_xy[:, None, :] - pos_xy[None, :, :], axis=-1)
        min_d = d.min(axis=1)
        order = np.argsort(min_d)
        neg_sorted = neg.iloc[order]
    else:
        neg_sorted = neg
    lo, hi = int(len(neg_sorted) * 0.25), int(len(neg_sorted) * 0.75)
    mid_neg = neg_sorted.iloc[lo:hi] if hi > lo else neg_sorted
    picked_neg = (
        rng.choice(mid_neg.index.to_numpy(), size=min(n_negative, len(mid_neg)), replace=False)
        if len(mid_neg) > 0 else np.array([], dtype=object)
    )

    return {
        "positive": [str(t) for t in picked_pos],
        "negative": [str(t) for t in picked_neg],
    }
