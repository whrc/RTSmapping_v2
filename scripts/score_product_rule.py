"""Score the SHIPPED adaptive product rule on a frozen prediction cache — report-only.

The manuscript's two available test numbers are both *fixed-minimum-size* rules at the
0.65 contour (ledger J: thr 0.65 / min_blob 80; ledger K: thr 0.65 / min_blob 2000).
The delivered South map uses neither: it is vectorised at the **0.30** contour
(``vectorize_region.py --threshold 0.30``), given a per-polygon ``max_prob``, and banded
by ``export_south_products.assign_conf_class`` / ``assign_rts_class`` — the high tier
carrying **no size floor at all**. This script scores that rule, from cached predictions
only, so the product's held-out number exists.

Five rows, one table, all IoU>=0.3, all on the same cache, all with the J/K
``valid``/``gt`` conventions:

  anchor_A_thr065_mb80       ledger J  — PARITY GATE
  anchor_B_thr065_mb2000     ledger K  — PARITY GATE
  product_candidates_all     south_rts_candidates.gpkg      (0.30 blobs, no tier filter)
  product_high_confidence    south_rts_high_confidence.gpkg (0.30 blobs, blob-max >= tier)
  product_high_conf_geom065  geometry sensitivity: same blobs, outline re-cut at the tier

Both anchors must reproduce their published values exactly or nothing is written.

This is a reporting re-cut of frozen cached predictions — the same category as the
ledger-K min_blob-2000 recompute (``configs/deployment.yaml`` min_blob note: "a different
point on the *same* probabilities"). No model run, no imagery read, no config touched.
**It does not supersede the frozen ledger-J number.** FACTS ONLY: counts, metrics, CIs —
no recommendations, no "primary" row.

Run:
    python scripts/score_product_rule.py \
        --cache /mnt/outputs/v1.0/diagnostics/test_probs.npz \
        --metadata /mnt/outputs/v1.0/data_local/metadata.csv \
        --out /mnt/outputs/v1.0/diagnostics \
        --tag frozen_test
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.splits import load_metadata  # noqa: E402
from inference.writer import (  # noqa: E402
    NODATA_SCALED_U8,
    SCALE_U8,
    _encode_scaled_uint8,
)
from scripts.analyze_residual_errors import (  # noqa: E402
    bootstrap_region_object_ci,
    object_detail_counts,
    score_by_region,
    _geometry_summary,
    _prf_counts,
)
from training.metrics import _filter_small_blobs, _object_match_detail  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCT_CONSTANTS_FILE = REPO_ROOT / "scripts" / "export_south_products.py"

LOW_SAMPLE_MIN_OBJECTS = 5  # mirrors scripts/object_scorecard.py

# The shipped vectorisation threshold. Unlike TIER_BOUNDS this has no code-level
# SSoT: it was a CLI argument on the shipped run (`vectorize_region.py
# --threshold 0.30`), recorded in post-inference/south_products.md D1.
VECTORIZE_THRESHOLD = 0.30
# vectorize_region.vectorize_region: min_blob_px = max(2, int(min_area_m2 / max_geo_px));
# the shipped South run used no MMU, so the technical 2-px floor is what applied.
VECTORIZE_MIN_BLOB_PX = 2

# Published values the anchors must reproduce exactly, with their ledger citations.
ANCHORS: dict[str, dict] = {
    "anchor_A_thr065_mb80": {
        "source": "docs/experiment_ledger.md Finding J (frozen one-shot 2026-06-25)",
        "thr": 0.65, "min_blob": 80,
        "obj_precision": 0.5839, "obj_recall": 0.4372, "obj_f1": 0.5000,
    },
    "anchor_B_thr065_mb2000": {
        "source": "docs/experiment_ledger.md Finding K, Per-region Test-Realistic (2026-06-29)",
        "thr": 0.65, "min_blob": 2000,
        "obj_precision": 0.768, "obj_recall": 0.400, "obj_f1": 0.526,
        "obj_tp": 86, "obj_fp": 26, "obj_fn": 129,
    },
}

ROW_DEFINITIONS = {
    "anchor_A_thr065_mb80":
        "Ledger J anchor: float prob >= 0.65, min_blob 80 px. Parity gate.",
    "anchor_B_thr065_mb2000":
        "Ledger K anchor: float prob >= 0.65, min_blob 2000 px. Parity gate.",
    "product_candidates_all":
        "Replicates south_rts_candidates.gpkg: blobs of quantised prob >= 0.30 "
        "(u8 >= cut_u8) inside valid, 2-px technical floor, no tier filter. All three "
        "rts_class values ship in this file.",
    "product_high_confidence":
        "Replicates south_rts_high_confidence.gpkg: the same 0.30-contour blobs, kept "
        "when blob-max quantised prob >= TIER_BOUNDS high (u8 >= tier_u8) — conf_class "
        "'high' -> rts_class 'high_confidence', at ANY size.",
    "product_high_conf_geom065":
        "Geometry sensitivity, NOT a shipped file: the same kept blobs with the same "
        "object identity and count, but the outline re-cut to the tier contour instead "
        "of the 0.30 contour. Isolates 'which blobs are kept' from 'what outline they have'.",
}

FIDELITY_LIMITS = [
    "Cross-tile seam dissolve is not replicable on isolated 512x512 tiles. Production "
    "unions edge-touching polygons across window seams BEFORE the min_blob filter and "
    "BEFORE max_prob is computed. Truncated border blobs fragment (raising FP, lowering "
    "matched IoU) and the tier test becomes conservative, since max over a truncated blob "
    "<= max over the dissolved polygon. Size of the affected population is measured per "
    "row as n_pred_edge_touching. The same truncation applies to anchors J/K, so "
    "cross-row comparison stays fair.",
    "Geodesic area is not available from the cache, so the 'candidate' and 'marginal' "
    "sub-tiers are not scored. CANDIDATE_MAX_AREA_M2 = 500 m2 is a geodesic (pyproj.Geod) "
    "area; the 3857 ground pixel area is res^2*cos^2(lat) and therefore latitude-dependent, "
    "so no single pixel count reproduces it. The high tier has no area rule (scored exactly) "
    "and product_candidates_all bounds the rest from above.",
    "NoData vs ignore semantics differ. Production excludes u8 == 255 (missing imagery); "
    "this script excludes label == 255 (label ambiguity). Using the label mask is the only "
    "choice comparable with ledger J/K, but an ignore region can split a blob production "
    "would keep whole.",
    "The probability FIELD is the test-tile estimator, not the production fused one. "
    "Production probabilities fuse overlapping 344-px-stride windows (sigma 128); the cache "
    "holds one 512x512 window per tile. The RULE is replicated exactly; the field it is "
    "applied to is not the deployed mosaic's.",
    "Polygonisation is NOT a fidelity limit: features.shapes (connectivity 4) -> "
    "features.rasterize -> pvals round-trips to the same pixel set as the blob mask, so "
    "blob max, mean and pixel count are exact. Only sub-pixel boundary geometry differs, "
    "which raster IoU cannot see.",
    "Test GT is one region. All GT objects fall in Northwest Russian-Novaya Zemlya tundra; "
    "the other three test regions are specificity-only (0 GT positives). This is a "
    "one-region number — already true of ledger J and K.",
]

CAVEATS = [
    "Reporting re-cut of frozen cached predictions. Does NOT supersede the frozen ledger-J "
    "number (0.584 / 0.437 / 0.500 at thr 0.65 / min_blob 80) — it is a different rule on "
    "the same probabilities.",
    "Cached probs are already temperature-scaled (T=0.512321) and 3-seed mean-fused. T is "
    "NOT re-applied here.",
    "No row is designated primary. south_rts_candidates.gpkg and "
    "south_rts_high_confidence.gpkg are both delivered files; which one a downstream "
    "document calls 'the map' is that document's framing decision.",
]


# ---------------------------------------------------------------------------
# Product-rule constants and cut points
# ---------------------------------------------------------------------------


def load_product_constants(path: Path, names: tuple[str, ...]) -> dict[str, object]:
    """Read module-level literal constants from a source file without importing it.

    ``scripts/export_south_products.py`` imports geopandas at module scope, which is
    absent from the scoring image (requirements_frozen.txt has rasterio and scipy but
    no geopandas). Parsing the literals keeps that module the single source of truth
    for the product rule without paying for its import chain, and fails loudly rather
    than falling back to a local copy that could silently fork.

    Args:
        path: Source file to parse.
        names: Module-level constant names to extract.

    Returns:
        Mapping of name -> literal value.

    Raises:
        KeyError: A requested name is absent, or is not assigned a literal.
    """
    tree = ast.parse(path.read_text())
    found: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except ValueError as exc:
                    raise KeyError(
                        f"{target.id} in {path} is not a literal constant: {exc}"
                    ) from exc
    missing = [n for n in names if n not in found]
    if missing:
        raise KeyError(f"{path} does not define module-level constants {missing}")
    return found


def vector_cut_u8(threshold: float) -> int:
    """Quantised cut for the vectorisation contour.

    Mirrors ``vectorize_region._polygonize_block``'s ``int(round(thr * SCALE_U8))``
    verbatim, Python's round-half-to-even included — that rounding is exactly why a
    0.65 *raster* cut is really 162/0.648 (post-inference/south_products.md:96-104).
    At 0.30 it is unambiguously 75.
    """
    return int(round(threshold * SCALE_U8))


def tier_cut_u8(high_bound: float) -> int:
    """Smallest stored uint8 whose decoded value clears the high-tier bound.

    ``export_south_products.assign_conf_class`` tests the DECODED value
    (``max_prob = u8 / 250 >= high_bound``), so this is a ceil, not the round used by
    :func:`vector_cut_u8`. At 0.65 that is 163 (>= 0.652) — deliberately NOT the 162
    (>= 0.648) that binarising the raster at 0.65 would give, which is the documented
    reason 129 t65 cores sit in medium polygons.
    """
    cut = math.ceil(high_bound * SCALE_U8)
    assert cut / SCALE_U8 >= high_bound > (cut - 1) / SCALE_U8, (
        f"tier cut {cut} does not bracket {high_bound}"
    )
    return cut


# ---------------------------------------------------------------------------
# Per-tile scoring
# ---------------------------------------------------------------------------


@dataclass
class TileCounts:
    """Object + pixel counts and match diagnostics for one tile, one row."""

    obj_tp: int = 0
    obj_fp: int = 0
    obj_fn: int = 0
    pix_tp: int = 0
    pix_fp: int = 0
    pix_fn: int = 0
    n_splits: int = 0
    n_merges: int = 0
    n_pred: int = 0
    n_pred_edge: int = 0
    matched_ious: list[float] = field(default_factory=list)


def _pixel_counts(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[int, int, int]:
    """Pixel tp/fp/fn, using ``analyze_residual_errors.object_counts``'s formulas.

    NB the basis differs from ``object_counts``: that function measures pixels on the
    raw thresholded mask, *before* ``_filter_small_blobs``, so ledger J's published
    pixel IoU/F1 are unfiltered. Here every row measures its FINAL mask — the geometry
    that actually ships — so the pixel columns stay comparable across rows. Object
    metrics, which the parity gate covers, are unaffected.
    """
    pix_tp = int(np.logical_and(pred, gt).sum())
    pix_fp = int(np.logical_and(pred, np.logical_not(gt) & valid).sum())
    pix_fn = int(np.logical_and(np.logical_not(pred) & valid, gt).sum())
    return pix_tp, pix_fp, pix_fn


def _edge_touching(pred_labels: np.ndarray) -> int:
    """Number of distinct blobs touching the tile border.

    Measures the population the missing cross-tile seam dissolve would have altered.
    """
    border = np.concatenate([
        pred_labels[0, :], pred_labels[-1, :], pred_labels[:, 0], pred_labels[:, -1],
    ])
    return int(np.count_nonzero(np.unique(border)))


def _score_pred(
    pred_labels: np.ndarray,
    n_pred: int,
    conf: np.ndarray,
    gt_labels: np.ndarray,
    n_gt: int,
    gt: np.ndarray,
    valid: np.ndarray,
    *,
    iou_thr: float,
    overlap_frac: float,
) -> TileCounts:
    """Match one labelled prediction image against the tile's GT and count."""
    d = _object_match_detail(
        pred_labels, n_pred, gt_labels, n_gt, conf, iou_thr, overlap_frac,
    )
    pix_tp, pix_fp, pix_fn = _pixel_counts(pred_labels > 0, gt, valid)
    return TileCounts(
        obj_tp=d.tp, obj_fp=d.fp, obj_fn=d.fn,
        pix_tp=pix_tp, pix_fp=pix_fp, pix_fn=pix_fn,
        n_splits=d.n_splits, n_merges=d.n_merges,
        n_pred=n_pred, n_pred_edge=_edge_touching(pred_labels),
        matched_ious=d.matched_ious,
    )


def anchor_tile_counts(
    prob: np.ndarray,
    label: np.ndarray,
    *,
    ignore_index: int,
    thr: float,
    min_blob: int,
    iou_thr: float,
    overlap_frac: float,
) -> TileCounts:
    """Detail-path counts at a float threshold (the ledger J/K prediction path).

    Identical prediction construction to ``analyze_residual_errors.object_counts``
    (morph_r=0); routed through ``_object_match_detail`` so the anchors and the product
    rows share one aggregation path. tp/fp/fn are bit-identical to ``_match_objects``.
    """
    valid = label != ignore_index
    gt = (label == 1) & valid
    pred = (prob >= thr) & valid
    pred_filt = _filter_small_blobs(pred.astype(np.uint8), min_blob)
    pred_labels, n_pred = ndimage.label(pred_filt)
    gt_labels, n_gt = ndimage.label(gt.astype(np.uint8))
    conf = (np.array(ndimage.mean(prob, pred_labels, index=np.arange(1, n_pred + 1)),
                     dtype=np.float64) if n_pred > 0 else np.zeros(0))
    return _score_pred(pred_labels, n_pred, conf, gt_labels, n_gt, gt, valid,
                       iou_thr=iou_thr, overlap_frac=overlap_frac)


def product_tile_counts(
    prob: np.ndarray,
    label: np.ndarray,
    *,
    ignore_index: int,
    cut_u8: int,
    tier_u8: int,
    min_blob_px: int,
    iou_thr: float,
    overlap_frac: float,
) -> dict[str, TileCounts]:
    """One tile -> the three product rows. Labels the tile once, matches three times.

    Replicates the product path: quantise to the scaled_uint8 encoding the COGs carry,
    cut the 0.30 contour, apply the 2-px technical floor, then band by the per-blob MAX
    quantised value. Floor first, tier test after — the production order (the floor lives
    in ``vectorize_region._polygonize_block``, the tier test in ``export_south_products``).

    Note the two different confidences: the greedy matcher's ORDERING key is the mean
    FLOAT prob (the ledger J/K convention, held constant across every row so the mask is
    the only thing that varies), while the TIER test uses the max QUANTISED value (the
    product rule).

    Returns:
        Mapping of row key -> counts for ``product_candidates_all``,
        ``product_high_confidence`` and ``product_high_conf_geom065``.
    """
    valid = label != ignore_index
    gt = (label == 1) & valid
    gt_labels, n_gt = ndimage.label(gt.astype(np.uint8))

    u8 = _encode_scaled_uint8(prob)
    # `valid` stands in for production's `m != NODATA_SCALED_U8` (see fidelity limits);
    # building the blobs inside it also keeps every later blob max/mean NoData-free,
    # exactly like _record's `pvals = pvals[pvals != 255]`.
    cand = (u8 >= cut_u8) & valid
    cand = _filter_small_blobs(cand.astype(np.uint8), min_blob_px)
    # scipy's default 4-connectivity matches rasterio.features.shapes(connectivity=4):
    # this is why a raster blob here and a product polygon there are the same object.
    lab30, n30 = ndimage.label(cand)
    conf30 = (np.array(ndimage.mean(prob, lab30, index=np.arange(1, n30 + 1)),
                       dtype=np.float64) if n30 > 0 else np.zeros(0))

    rows = {
        "product_candidates_all": _score_pred(
            lab30, n30, conf30, gt_labels, n_gt, gt, valid,
            iou_thr=iou_thr, overlap_frac=overlap_frac),
    }

    # conf_class 'high' -> rts_class 'high_confidence', at ANY size.
    maxu8 = (np.array(ndimage.maximum(u8, lab30, index=np.arange(1, n30 + 1)))
             if n30 > 0 else np.zeros(0))
    keep = maxu8 >= tier_u8
    n_high = int(keep.sum())
    # Renumber contiguously — the matcher indexes blobs as `pred_labels == p + 1`.
    remap = np.zeros(n30 + 1, dtype=np.int32)
    remap[np.flatnonzero(keep) + 1] = np.arange(1, n_high + 1)
    lab_high = remap[lab30]
    conf_high = conf30[keep] if n30 > 0 else np.zeros(0)
    rows["product_high_confidence"] = _score_pred(
        lab_high, n_high, conf_high, gt_labels, n_gt, gt, valid,
        iou_thr=iou_thr, overlap_frac=overlap_frac)

    # Same objects, same ids, same count — only the outline changes. Every kept blob
    # has at least one pixel at u8 >= tier_u8 by construction, so no label goes empty.
    # Deliberately NOT relabelled: relabelling would change object cardinality too and
    # defeat the row's purpose of isolating outline from selection.
    lab_geom = np.where(u8 >= tier_u8, lab_high, 0)
    rows["product_high_conf_geom065"] = _score_pred(
        lab_geom, n_high, conf_high, gt_labels, n_gt, gt, valid,
        iou_thr=iou_thr, overlap_frac=overlap_frac)
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_row(
    counts: list[TileCounts],
    tids: list[str],
    tid_region: dict[str, str],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    """Roll per-tile counts up to {aggregate, per_region} in the object_scorecard shape.

    Args:
        counts: Per-tile counts, in the same order as ``tids``.
        tids: Tile IDs.
        tid_region: Tile ID -> RegionName.
        n_boot: Bootstrap resamples for the tile-cluster CIs.
        seed: Bootstrap seed.

    Returns:
        ``{"aggregate": {...}, "per_region": {region: {...}}}``.
    """
    by_region: dict[str, list[TileCounts]] = {}
    for c, tid in zip(counts, tids):
        by_region.setdefault(tid_region.get(tid, "UNKNOWN"), []).append(c)

    def _block(group: list[TileCounts], region_scoped: bool) -> dict:
        otp = sum(c.obj_tp for c in group)
        ofp = sum(c.obj_fp for c in group)
        ofn = sum(c.obj_fn for c in group)
        ptp = sum(c.pix_tp for c in group)
        pfp = sum(c.pix_fp for c in group)
        pfn = sum(c.pix_fn for c in group)
        n_gt = otp + ofn
        n_pred = sum(c.n_pred for c in group)
        n_edge = sum(c.n_pred_edge for c in group)
        op, orc, of1 = _prf_counts(otp, ofp, ofn)
        pp, prc, _ = _prf_counts(ptp, pfp, pfn)
        ious: list[float] = []
        for c in group:
            ious.extend(c.matched_ious)
        # Recall/F1 are undefined with no positives -> None, not a spurious 0
        # (matches score_by_region's convention).
        block = {
            "n_gt_objects": n_gt,
            "n_pred_objects": n_pred,
            "n_pred_edge_touching": n_edge,
            "edge_touching_frac": round(n_edge / n_pred, 4) if n_pred else None,
            "obj_tp": otp, "obj_fp": ofp, "obj_fn": ofn,
            "obj_precision": round(op, 4) if (otp + ofp) else None,
            "obj_recall": round(orc, 4) if n_gt else None,
            "obj_f1": round(of1, 4) if n_gt and (otp + ofp) else None,
            "obj_ci": bootstrap_region_object_ci(
                [(c.obj_tp, c.obj_fp, c.obj_fn) for c in group],
                n_boot=n_boot, seed=seed),
            "pixel_tp": ptp, "pixel_fp": pfp, "pixel_fn": pfn,
            "pixel_precision": round(pp, 4) if (ptp + pfp) else None,
            "pixel_recall": round(prc, 4) if (ptp + pfn) else None,
            "n_splits": sum(c.n_splits for c in group),
            "n_merges": sum(c.n_merges for c in group),
            "geometry": _geometry_summary(ious),
        }
        if region_scoped:
            block["low_sample"] = n_gt < LOW_SAMPLE_MIN_OBJECTS
        return block

    return {
        "aggregate": _block(counts, region_scoped=False),
        "per_region": {r: _block(g, region_scoped=True)
                       for r, g in sorted(by_region.items())},
    }


def _round_like(observed: float, published: float) -> float:
    """Round ``observed`` to the number of decimals ``published`` was quoted with."""
    text = f"{published!r}"
    decimals = len(text.split(".")[1]) if "." in text else 0
    return round(observed, decimals)


def check_anchors(
    probs: np.ndarray,
    labels: np.ndarray,
    tids: list[str],
    tid_region: dict[str, str],
    *,
    ignore_index: int,
    iou_thr: float,
    overlap_frac: float,
    n_boot: int,
    seed: int,
) -> tuple[bool, dict, dict, bool]:
    """Reproduce ledger J and K from the cache and compare against the published values.

    Runs each anchor through ``score_by_region`` — literally the function
    ``scripts/evaluate_test.py`` imports to produce J and K, so "same code path" is
    provable by import — and then through the detail path the product rows use, so the
    two aggregations are proven equal before the product rows rely on the detail one.

    Returns:
        ``(all_match, anchor_report, anchor_rows, detail_parity_ok)``.
    """
    report: dict = {}
    rows: dict = {}
    all_match = True
    detail_parity_ok = True

    for key, spec in ANCHORS.items():
        thr, min_blob = spec["thr"], spec["min_blob"]
        logger.info("Anchor %s: score_by_region at thr=%.2f min_blob=%d", key, thr, min_blob)
        pt = score_by_region(
            probs, labels, tids, tid_region,
            ignore_index=ignore_index, thr=thr, min_blob=min_blob, iou_thr=iou_thr,
        )["ALL"]

        expected = {k: v for k, v in spec.items() if k not in ("thr", "min_blob", "source")}
        observed = {k: pt[k] for k in expected}
        mismatches = {
            k: {"expected": expected[k], "observed": observed[k]}
            for k in expected
            if (observed[k] != expected[k] if isinstance(expected[k], int)
                else _round_like(observed[k], expected[k]) != expected[k])
        }
        match = not mismatches
        all_match &= match
        report[key] = {
            "source": spec["source"], "thr": thr, "min_blob": min_blob,
            "expected": expected, "observed": observed, "match": match,
        }
        if mismatches:
            report[key]["mismatches"] = mismatches
            logger.error("PARITY FAILURE %s: %s", key, json.dumps(mismatches, indent=2))
            continue

        logger.info("Anchor %s reproduced exactly: P %s / R %s / F1 %s",
                    key, pt["obj_precision"], pt["obj_recall"], pt["obj_f1"])
        counts = [
            anchor_tile_counts(prob, label, ignore_index=ignore_index, thr=thr,
                               min_blob=min_blob, iou_thr=iou_thr, overlap_frac=overlap_frac)
            for prob, label in zip(probs, labels)
        ]
        rows[key] = aggregate_row(counts, tids, tid_region, n_boot=n_boot, seed=seed)
        rows[key]["definition"] = ROW_DEFINITIONS[key]
        agg = rows[key]["aggregate"]
        if (agg["obj_tp"], agg["obj_fp"], agg["obj_fn"]) != (
                pt["obj_tp"], pt["obj_fp"], pt["obj_fn"]):
            detail_parity_ok = False
            logger.error("Detail path != score_by_region for %s: %s vs %s", key,
                         (agg["obj_tp"], agg["obj_fp"], agg["obj_fn"]),
                         (pt["obj_tp"], pt["obj_fp"], pt["obj_fn"]))

    report["all_match"] = all_match
    return all_match, report, rows, detail_parity_ok


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Current repo HEAD, read straight from .git, or 'unknown' outside a checkout.

    Deliberately does not shell out to git: the repo is bind-mounted into the scoring
    container under a foreign uid, which trips git's dubious-ownership guard, and git
    2.34 ignores ``safe.directory`` supplied via ``-c`` — so a subprocess would stamp
    every containerised artifact 'unknown'. Provenance on a cited artifact is worth
    the eight lines.
    """
    git_dir = REPO_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head  # detached HEAD holds the sha directly
        ref = head[5:]
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text().strip()
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(f" {ref}"):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"


def _sha256(path: Path) -> str:
    """Streaming SHA-256 of a file (the cache is >1 GB)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build_report(
    probs: np.ndarray,
    labels: np.ndarray,
    tids: list[str],
    tid_region: dict[str, str],
    *,
    ignore_index: int,
    cut_u8: int,
    tier_u8: int,
    min_blob_px: int,
    iou_thr: float,
    overlap_frac: float,
    n_boot: int,
    seed: int,
    constants: dict,
) -> dict | None:
    """Run the anchors then the product rows and assemble the artifact.

    Returns:
        The report dict, or ``None`` if either parity anchor failed (in which case
        nothing should be written).
    """
    all_match, anchor_report, anchor_rows, detail_parity_ok = check_anchors(
        probs, labels, tids, tid_region, ignore_index=ignore_index, iou_thr=iou_thr,
        overlap_frac=overlap_frac, n_boot=n_boot, seed=seed,
    )
    if not all_match:
        return None

    logger.info("Product pass: cut_u8=%d (thr %.2f) tier_u8=%d (>= %.4f) min_blob_px=%d",
                cut_u8, VECTORIZE_THRESHOLD, tier_u8, tier_u8 / SCALE_U8, min_blob_px)
    product_counts: dict[str, list[TileCounts]] = {}
    for prob, label in zip(probs, labels):
        for key, tc in product_tile_counts(
            prob, label, ignore_index=ignore_index, cut_u8=cut_u8, tier_u8=tier_u8,
            min_blob_px=min_blob_px, iou_thr=iou_thr, overlap_frac=overlap_frac,
        ).items():
            product_counts.setdefault(key, []).append(tc)

    rows = dict(anchor_rows)
    for key, counts in product_counts.items():
        rows[key] = aggregate_row(counts, tids, tid_region, n_boot=n_boot, seed=seed)
        rows[key]["definition"] = ROW_DEFINITIONS[key]
        agg = rows[key]["aggregate"]
        logger.info("%-26s P %-7s R %-7s F1 %-7s (tp %d fp %d fn %d, n_pred %d)",
                    key, agg["obj_precision"], agg["obj_recall"], agg["obj_f1"],
                    agg["obj_tp"], agg["obj_fp"], agg["obj_fn"], agg["n_pred_objects"])

    high = rows["product_high_confidence"]["aggregate"]
    geom = rows["product_high_conf_geom065"]["aggregate"]
    cand = rows["product_candidates_all"]["aggregate"]
    self_check = {
        "anchor_detail_vs_score_by_region_match": detail_parity_ok,
        "high_tier_subset_of_candidates":
            high["n_pred_objects"] <= cand["n_pred_objects"],
        "geom_row_npred_equals_high_row_npred":
            geom["n_pred_objects"] == high["n_pred_objects"],
    }
    for name, ok in self_check.items():
        if not ok:
            logger.warning("SELF-CHECK false: %s", name)

    return {
        "_caveats": CAVEATS,
        "_provenance": {
            "script": "scripts/score_product_rule.py",
            "git_sha": _git_sha(),
            "ran_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "n_tiles": len(tids),
            "n_gt_objects": rows["anchor_A_thr065_mb80"]["aggregate"]["n_gt_objects"],
            "probs_note": "already temperature-scaled (T=0.512321) and 3-seed mean-fused; "
                          "T is NOT re-applied",
            "product_rule_source": {
                "constants_file": "scripts/export_south_products.py",
                "TIER_BOUNDS": list(constants["TIER_BOUNDS"]),
                "CANDIDATE_MAX_AREA_M2": constants["CANDIDATE_MAX_AREA_M2"],
                "CANDIDATE_MAX_AREA_M2_unused_reason":
                    "geodesic area is not available from the cache; only the high tier "
                    "(which has no area rule) and the all-polygons bound are scored",
                "vectorize_threshold": VECTORIZE_THRESHOLD,
                "vectorize_threshold_source":
                    "CLI arg of the shipped run (vectorize_region.py --threshold 0.30); "
                    "post-inference/south_products.md D1 — no code-level SSoT",
                "min_blob_px": min_blob_px,
                "min_blob_px_source":
                    "vectorize_region.vectorize_region: max(2, int(min_area_m2 / "
                    "max_geo_px)); the shipped South run used no MMU",
            },
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        },
        "config": {
            "scale_u8": SCALE_U8,
            "nodata_scaled_u8": NODATA_SCALED_U8,
            "cut_u8": cut_u8,
            "tier_u8": tier_u8,
            "quantisation":
                "np.round(prob*250) clipped to [0,250] via "
                "inference.writer._encode_scaled_uint8 (round-half-to-even)",
            "tier_u8_note":
                f"{tier_u8} (>= {tier_u8 / SCALE_U8:.4f}) because conf_class tests the "
                f"decoded max_prob = u8/250 >= {constants['TIER_BOUNDS'][1]}. The 0.65 "
                "RASTER cut used by south_rts_t65.gpkg is 162 (>= 0.648) — "
                "post-inference/south_products.md:96-104",
            "object_iou_threshold": iou_thr,
            "overlap_frac": overlap_frac,
            "morph_radius": 0,
            "pixel_counts_basis":
                "final predicted mask (post size-filter, post tier filter) for every "
                "row, so the pixel columns are comparable across rows. This is NOT the "
                "basis of ledger J's published pixel IoU 0.432 / F1 0.604, which "
                "analyze_residual_errors.object_counts measures on the raw thresholded "
                "mask before _filter_small_blobs. Object metrics are unaffected.",
            "match_conf": "mean_float_prob",
            "match_conf_note":
                "match ORDER uses the mean FLOAT prob (ledger J/K convention, held "
                "constant across all rows); the TIER test uses the max QUANTISED u8 "
                "(the product rule)",
            "ignore_index": ignore_index,
            "n_boot": n_boot,
            "seed": seed,
            "low_sample_min_objects": LOW_SAMPLE_MIN_OBJECTS,
        },
        "anchors": anchor_report,
        "rows": rows,
        "self_check": self_check,
        "fidelity_limits": FIDELITY_LIMITS,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", required=True, help="*_probs.npz (tids, probs, labels)")
    p.add_argument("--metadata", required=True, help="metadata.csv for RegionName map")
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--tag", default="frozen_test", help="label for this split")
    p.add_argument("--vector-thr", type=float, default=VECTORIZE_THRESHOLD,
                   help="product vectorisation threshold (shipped South run: 0.30)")
    p.add_argument("--min-blob-px", type=int, default=VECTORIZE_MIN_BLOB_PX,
                   help="vectorize_region technical floor (shipped South run: 2)")
    p.add_argument("--iou-thr", type=float, default=0.3)
    p.add_argument("--overlap-frac", type=float, default=0.1,
                   help="min intersection/smaller-blob fraction to count a split/merge")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_dir / f"product_rule_scorecard_{args.tag}.log"))

    constants = load_product_constants(
        PRODUCT_CONSTANTS_FILE, ("TIER_BOUNDS", "CANDIDATE_MAX_AREA_M2"))
    cut_u8 = vector_cut_u8(args.vector_thr)
    tier_u8 = tier_cut_u8(float(constants["TIER_BOUNDS"][1]))
    logger.info("Product rule from %s: TIER_BOUNDS=%s CANDIDATE_MAX_AREA_M2=%s",
                PRODUCT_CONSTANTS_FILE, constants["TIER_BOUNDS"],
                constants["CANDIDATE_MAX_AREA_M2"])

    logger.info("Loading cached predictions: %s", args.cache)
    z = np.load(args.cache, allow_pickle=True)
    tids = [str(t) for t in z["tids"]]
    probs, labels = z["probs"], z["labels"]
    logger.info("%d tiles | tag=%s", len(tids), args.tag)

    meta = load_metadata(args.metadata)
    tid_region = dict(zip(meta["Tile_ID"], meta["RegionName"]))

    report = build_report(
        probs, labels, tids, tid_region,
        ignore_index=args.ignore_index, cut_u8=cut_u8, tier_u8=tier_u8,
        min_blob_px=args.min_blob_px, iou_thr=args.iou_thr,
        overlap_frac=args.overlap_frac, n_boot=args.n_boot, seed=args.seed,
        constants=constants,
    )
    if report is None:
        logger.error("Parity anchors did not reproduce — writing NO artifact. "
                     "The cache or the scoring machinery has drifted; investigate "
                     "before citing any number from this run.")
        return 2

    report["_tag"] = args.tag
    report["_source_cache"] = args.cache
    report["_provenance"]["metadata_csv"] = args.metadata
    report["_provenance"]["cache_bytes"] = Path(args.cache).stat().st_size
    report["_provenance"]["cache_sha256"] = _sha256(Path(args.cache))

    out_path = out_dir / f"product_rule_scorecard_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", out_path)

    print(json.dumps({
        "tag": args.tag,
        "anchors_all_match": report["anchors"]["all_match"],
        "n_gt_objects": report["_provenance"]["n_gt_objects"],
        "rows": {
            key: {
                "obj_precision": row["aggregate"]["obj_precision"],
                "obj_recall": row["aggregate"]["obj_recall"],
                "obj_f1": row["aggregate"]["obj_f1"],
                "obj_tp": row["aggregate"]["obj_tp"],
                "obj_fp": row["aggregate"]["obj_fp"],
                "obj_fn": row["aggregate"]["obj_fn"],
                "n_pred_objects": row["aggregate"]["n_pred_objects"],
                "iou_median": (row["aggregate"]["geometry"] or {}).get("iou_median"),
            }
            for key, row in report["rows"].items()
        },
        "self_check": report["self_check"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
