"""Validation metrics: pixel IoU/F1, object P/R/F1, PR-AUC at ratios.

The ValidationAccumulator consumes batches across the val loop, holds per-tile
logits + labels in memory, and produces a single metrics dict at the end.

Memory footprint (reference): N_val tiles * 512 * 512 * 5 bytes
(float32 logits + uint8 labels) ~= 1.3 MB per tile. At N_val = 5000 tiles,
peak ~6.5 GB. Acceptable for Phase 1; revisit if val sets grow.

PR-AUC at ratios 1:200/500/1000 (training.md §10.3, "Efficient Multi-Ratio
Evaluation") subsamples negatives *without replacement* when the val set has
enough; falls back to bootstrap with replacement (with a log-warning) when
the pool is insufficient. This is a **prevalence-conditional deployment
estimate**, not a prevalence-free quality score — absolute numbers across
ratios are not directly comparable (training.md §4.2 comment in §12.2 and
plan risk #2).

Edge cases for object metrics (training.md §6.2 + plan adversarial review):
  - Predictions smaller than min_blob_size are dropped (speckle-FP guard).
  - Empty prediction on a positive tile -> all GTs count as FN.
  - Empty GT on a negative tile -> all surviving preds count as FP.
  - One-pred-many-GT and many-pred-one-GT resolve naturally via confidence-
    sorted greedy 1-to-1 matching (sorted by mean predicted probability).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-tile record kept in memory
# ---------------------------------------------------------------------------


@dataclass
class _TileRecord:
    tile_id: str
    is_positive_tile: bool
    # Flattened valid-pixel-only (ignore pixels dropped) arrays, for PR-AUC.
    valid_logits: np.ndarray   # (n_valid,) float32
    valid_labels: np.ndarray   # (n_valid,) uint8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_small_blobs(binary: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than `min_size` pixels.

    Args:
        binary: 2-D uint8 array of {0, 1}.
        min_size: Minimum blob size in pixels (inclusive threshold).

    Returns:
        Copy of `binary` with undersized blobs zeroed out.
    """
    if min_size <= 1 or binary.sum() == 0:
        return binary
    labels, n = ndimage.label(binary)
    if n == 0:
        return binary
    sizes = ndimage.sum(binary, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_size
    return keep[labels].astype(np.uint8)


def _match_objects(
    pred_labels: np.ndarray,
    n_pred: int,
    gt_labels: np.ndarray,
    n_gt: int,
    pred_conf_per_blob: np.ndarray,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Greedy 1-to-1 matching by descending confidence. Returns (tp, fp, fn)."""
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    order = np.argsort(pred_conf_per_blob)[::-1]  # highest-confidence first
    matched_gt: set[int] = set()
    tp = 0
    for p_idx in order:
        p_mask = pred_labels == (p_idx + 1)
        best_iou = 0.0
        best_gt = -1
        for g_idx in range(n_gt):
            if g_idx in matched_gt:
                continue
            g_mask = gt_labels == (g_idx + 1)
            inter = np.logical_and(p_mask, g_mask).sum()
            if inter == 0:
                continue
            union = np.logical_or(p_mask, g_mask).sum()
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_gt = g_idx
        if best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            tp += 1

    fp = n_pred - tp
    fn = n_gt - len(matched_gt)
    return tp, fp, fn


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


class ValidationAccumulator:
    """Accumulate per-batch validation predictions, compute metrics at the end.

    Usage:
        acc = ValidationAccumulator(cfg)
        for batch in val_loader:
            logits = model(batch["image"])
            acc.update(logits, batch["label"], batch["tile_id"])
        metrics = acc.compute()

    Args:
        cfg: Parsed YAML config (uses data.label_ignore_index and the metrics.*
            block from configs/baseline.yaml).
        ratios: Prevalence ratios for PR-AUC. Defaults to [200, 500, 1000].
        seed: Seed for negative-subsampling RNG. Defaults to 42.
    """

    def __init__(
        self,
        cfg: dict,
        ratios: list[int] | None = None,
        seed: int = 42,
    ):
        self.ignore_index = int(cfg["data"]["label_ignore_index"])
        metrics_cfg = cfg.get("metrics", {})
        self.threshold = float(metrics_cfg.get("reporting_threshold", 0.5))
        self.min_blob_size = int(metrics_cfg.get("min_blob_size_px", 10))
        self.obj_iou_threshold = float(metrics_cfg.get("object_iou_threshold", 0.3))
        self.ratios: list[int] = list(ratios) if ratios is not None else [200, 500, 1000]
        self._rng = np.random.default_rng(seed)

        # Pixel-level TP/FP/FN (ignore pixels excluded).
        self.pixel_tp = 0
        self.pixel_fp = 0
        self.pixel_fn = 0

        # Object-level counters aggregated over tiles.
        self.obj_tp = 0
        self.obj_fp = 0
        self.obj_fn = 0

        # Per-tile records for PR-AUC at ratios.
        self._tiles: list[_TileRecord] = []

    # ------------------------------------------------------------------ update

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tile_ids: list[str],
    ) -> None:
        """Consume one batch.

        Args:
            logits: (B, 1, H, W) float — model outputs, no sigmoid applied.
            labels: (B, H, W) long — values in {0, 1, ignore_index}.
            tile_ids: length-B list of string tile IDs.
        """
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"Expected (B, 1, H, W), got {tuple(logits.shape)}")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("Batch size mismatch between logits and labels")

        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        logits_np = logits.squeeze(1).detach().cpu().float().numpy()
        labels_np = labels.detach().cpu().numpy()

        for b in range(probs.shape[0]):
            self._accumulate_tile(probs[b], logits_np[b], labels_np[b], tile_ids[b])

    def _accumulate_tile(
        self,
        prob: np.ndarray,
        logit: np.ndarray,
        label: np.ndarray,
        tile_id: str,
    ) -> None:
        valid = label != self.ignore_index
        gt = (label == 1) & valid
        pred = (prob >= self.threshold) & valid

        # Pixel-level
        self.pixel_tp += int(np.logical_and(pred, gt).sum())
        self.pixel_fp += int(np.logical_and(pred, np.logical_not(gt) & valid).sum())
        self.pixel_fn += int(np.logical_and(np.logical_not(pred) & valid, gt).sum())

        # Object-level: filter small-blob predictions, label components.
        pred_bin = pred.astype(np.uint8)
        pred_filt = _filter_small_blobs(pred_bin, self.min_blob_size)
        gt_bin = gt.astype(np.uint8)

        pred_labels, n_pred = ndimage.label(pred_filt)
        gt_labels, n_gt = ndimage.label(gt_bin)

        if n_pred > 0:
            # Mean predicted probability per blob for confidence-sorted matching.
            conf = np.array(
                ndimage.mean(prob, pred_labels, index=np.arange(1, n_pred + 1)),
                dtype=np.float64,
            )
        else:
            conf = np.zeros(0)

        tp, fp, fn = _match_objects(
            pred_labels, n_pred, gt_labels, n_gt, conf, self.obj_iou_threshold,
        )
        self.obj_tp += tp
        self.obj_fp += fp
        self.obj_fn += fn

        # PR-AUC data: keep valid-pixel-only logits + labels.
        self._tiles.append(_TileRecord(
            tile_id=tile_id,
            is_positive_tile=bool(gt.any()),
            valid_logits=logit[valid].astype(np.float32),
            valid_labels=label[valid].astype(np.uint8),
        ))

    # ----------------------------------------------------------------- compute

    def compute(self) -> dict[str, float]:
        """Compute all metrics. Safe to call exactly once at validation end."""
        result: dict[str, float] = {
            "pixel_iou": _safe_div(self.pixel_tp, self.pixel_tp + self.pixel_fp + self.pixel_fn),
            "pixel_f1": _safe_div(2 * self.pixel_tp, 2 * self.pixel_tp + self.pixel_fp + self.pixel_fn),
            "object_precision": _safe_div(self.obj_tp, self.obj_tp + self.obj_fp),
            "object_recall": _safe_div(self.obj_tp, self.obj_tp + self.obj_fn),
        }
        obj_p, obj_r = result["object_precision"], result["object_recall"]
        result["object_f1"] = _safe_div(2 * obj_p * obj_r, obj_p + obj_r)

        pr_aucs = self._pr_auc_at_ratios()
        result.update(pr_aucs)

        # Geometric mean across ratios — the early-stopping metric
        # (config: training.early_stopping.metric = val_realistic_pr_auc_geomean).
        ratio_values = [pr_aucs[f"pr_auc_ratio_{r}"] for r in self.ratios]
        result["val_realistic_pr_auc_geomean"] = _geomean(ratio_values)

        # Global val prevalence (surfaces alongside per-ratio APs — see
        # plan risk #2).
        n_pos_tiles = sum(1 for t in self._tiles if t.is_positive_tile)
        result["val_n_positive_tiles"] = float(n_pos_tiles)
        result["val_n_negative_tiles"] = float(len(self._tiles) - n_pos_tiles)

        return result

    def _pr_auc_at_ratios(self) -> dict[str, float]:
        pos = [t for t in self._tiles if t.is_positive_tile]
        neg = [t for t in self._tiles if not t.is_positive_tile]
        n_pos = len(pos)
        out: dict[str, float] = {}
        if n_pos == 0:
            logger.warning("No positive tiles in val set; PR-AUC defined as 0.0")
            return {f"pr_auc_ratio_{r}": 0.0 for r in self.ratios}

        for r in self.ratios:
            needed = r * n_pos
            if len(neg) == 0:
                out[f"pr_auc_ratio_{r}"] = 0.0
                continue
            if len(neg) >= needed:
                idx = self._rng.choice(len(neg), size=needed, replace=False)
            else:
                logger.warning(
                    "Only %d negative val tiles for ratio 1:%d (need %d); "
                    "falling back to bootstrap with replacement (plan risk #2, "
                    "Step 3 revisit gate).", len(neg), r, needed,
                )
                idx = self._rng.choice(len(neg), size=needed, replace=True)
            sub = [neg[i] for i in idx]
            logits = np.concatenate([t.valid_logits for t in pos + sub])
            labels = np.concatenate([t.valid_labels for t in pos + sub])
            # average_precision_score with integer {0, 1} labels and
            # real-valued scores (logits are monotonic with sigmoid(logits)
            # so AP is identical whether we pass logits or sigmoid(logits)).
            if labels.max() == 0:
                out[f"pr_auc_ratio_{r}"] = 0.0
            else:
                out[f"pr_auc_ratio_{r}"] = float(average_precision_score(labels, logits))
        return out


def _geomean(values: list[float]) -> float:
    """Geometric mean with a small floor to survive zeros."""
    floored = [max(float(v), 1e-12) for v in values]
    if not floored:
        return 0.0
    return math.exp(sum(math.log(v) for v in floored) / len(floored))
