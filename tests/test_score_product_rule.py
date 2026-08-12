"""Unit tests for the shipped-product-rule scorer (scripts/score_product_rule.py).

Covers the parts of the product replication that are easy to get subtly wrong:
  - the 1/250 quantisation boundary that decides conf_class (the reason a 0.65
    tier cut is u8 >= 163 and NOT the u8 >= 162 that binarising the raster gives)
  - the 2-px technical floor, and its ordering *before* the tier test
  - the geometry-sensitivity row preserving object identity and count
  - anchor_tile_counts staying bit-identical to the ledger-J/K object machinery
  - load_product_constants reading export_south_products.py without importing it

All synthetic, no GPU/GCS. (Modules import training.metrics -> torch; torch is a
test dependency per tests/tests.md.)
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.writer import SCALE_U8
from scripts.analyze_residual_errors import object_counts
from scripts.score_product_rule import (
    PRODUCT_CONSTANTS_FILE,
    TileCounts,
    aggregate_row,
    anchor_tile_counts,
    load_product_constants,
    product_tile_counts,
    tier_cut_u8,
    vector_cut_u8,
)

CUT_U8 = vector_cut_u8(0.30)     # 75
TIER_U8 = tier_cut_u8(0.65)      # 163


def _blank(size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Empty (prob, label) tile: all-zero probability, all-background label."""
    return np.zeros((size, size), np.float32), np.zeros((size, size), np.int16)


def _product(prob: np.ndarray, label: np.ndarray, min_blob_px: int = 2) -> dict[str, TileCounts]:
    return product_tile_counts(
        prob, label, ignore_index=255, cut_u8=CUT_U8, tier_u8=TIER_U8,
        min_blob_px=min_blob_px, iou_thr=0.3, overlap_frac=0.1,
    )


# ---------------------------------------------------------------------------
# Cut points
# ---------------------------------------------------------------------------


def test_cut_points_match_the_product_arithmetic():
    """0.30 -> round -> 75; 0.65 -> ceil -> 163 (not the 162 of a raster cut)."""
    assert vector_cut_u8(0.30) == 75
    assert tier_cut_u8(0.65) == 163
    # The tier cut brackets the bound: 162/250 = 0.648 is below it, 163/250 = 0.652 clears it.
    assert 162 / SCALE_U8 < 0.65 <= 163 / SCALE_U8


# ---------------------------------------------------------------------------
# The 1/250 tier boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "peak, expect_high",
    [
        (0.6499, False),  # *250 = 162.475 -> u8 162 -> max_prob 0.648 -> medium
        (0.652, True),    # *250 = 163.0   -> u8 163 -> max_prob 0.652 -> high
        (0.99, True),
        (0.50, False),
    ],
)
def test_tier_boundary_is_on_the_quantised_grid(peak: float, expect_high: bool):
    """conf_class is decided by the quantised max, not the raw float max."""
    prob, label = _blank()
    prob[10:14, 10:14] = 0.40   # a blob that clears the 0.30 contour
    prob[11, 11] = peak         # ...with a single peak pixel deciding its tier
    rows = _product(prob, label)
    assert rows["product_candidates_all"].n_pred == 1
    assert rows["product_high_confidence"].n_pred == (1 if expect_high else 0)


# ---------------------------------------------------------------------------
# The 2-px technical floor, and its ordering
# ---------------------------------------------------------------------------


def test_floor_drops_one_pixel_blobs_and_keeps_two():
    prob, label = _blank()
    prob[5, 5] = 0.90            # 1 px -> dropped
    prob[20, 20:22] = 0.90       # 2 px -> kept
    rows = _product(prob, label)
    assert rows["product_candidates_all"].n_pred == 1
    assert rows["product_high_confidence"].n_pred == 1


def test_floor_is_applied_before_the_tier_test():
    """A lone high-probability pixel must not become a high_confidence object.

    Production applies the pixel floor in vectorize_region._polygonize_block and only
    then bands by max_prob in export_south_products; testing the tier first would
    resurrect single-pixel noise as a 'fact map' detection.
    """
    prob, label = _blank()
    prob[7, 7] = 0.99
    rows = _product(prob, label)
    assert rows["product_candidates_all"].n_pred == 0
    assert rows["product_high_confidence"].n_pred == 0


# ---------------------------------------------------------------------------
# Row relationships
# ---------------------------------------------------------------------------


def test_high_tier_is_a_subset_and_geometry_row_preserves_identity():
    """geom065 keeps the same objects, only shrinks their outlines."""
    prob, label = _blank()
    prob[4:10, 4:10] = 0.40      # medium-only blob -> candidates row only
    prob[20:26, 20:26] = 0.40    # high blob: broad 0.30 skirt...
    prob[22:24, 22:24] = 0.90    # ...with a compact >=0.65 core
    label[21:25, 21:25] = 1
    rows = _product(prob, label)

    assert rows["product_candidates_all"].n_pred == 2
    assert rows["product_high_confidence"].n_pred == 1
    assert rows["product_high_conf_geom065"].n_pred == 1

    # Same object, different outline: the 0.30 skirt is 36 px, the core is 4 px.
    high = rows["product_high_confidence"]
    geom = rows["product_high_conf_geom065"]
    assert high.pix_tp + high.pix_fp == 36
    assert geom.pix_tp + geom.pix_fp == 4


def test_edge_touching_counts_border_blobs():
    prob, label = _blank()
    prob[0, 0:2] = 0.90          # on the border
    prob[15:17, 15:17] = 0.90    # interior
    rows = _product(prob, label)
    assert rows["product_candidates_all"].n_pred == 2
    assert rows["product_candidates_all"].n_pred_edge == 1


def test_ignore_regions_are_excluded_from_predictions():
    prob, label = _blank()
    prob[10:14, 10:14] = 0.90
    label[10:14, 10:14] = 255
    rows = _product(prob, label)
    assert rows["product_candidates_all"].n_pred == 0


# ---------------------------------------------------------------------------
# Anchor path parity with the ledger J/K machinery
# ---------------------------------------------------------------------------


def test_anchor_tile_counts_match_object_counts():
    """The detail path the product rows use must equal the frozen anchor path.

    Object counts must be bit-identical. Pixel counts deliberately are not: this
    script measures the FINAL mask so the pixel columns are comparable across rows,
    while object_counts measures the raw mask before _filter_small_blobs (which is
    the basis of ledger J's published pixel IoU/F1).
    """
    rng = np.random.default_rng(0)
    prob = rng.random((32, 32)).astype(np.float32)
    label = np.zeros((32, 32), np.int16)
    label[8:16, 8:16] = 1
    label[0:4, 0:4] = 255

    tc = anchor_tile_counts(prob, label, ignore_index=255, thr=0.65, min_blob=4,
                            iou_thr=0.3, overlap_frac=0.1)
    otp, ofp, ofn, ptp, pfp, pfn = object_counts(prob, label, 255, 0.65, 4, 0, 0.3)
    assert (tc.obj_tp, tc.obj_fp, tc.obj_fn) == (otp, ofp, ofn)
    # Post-filter <= pre-filter on both positive-prediction counts.
    assert tc.pix_tp <= ptp and tc.pix_fp <= pfp
    # With min_blob=1 the two bases coincide.
    tc1 = anchor_tile_counts(prob, label, ignore_index=255, thr=0.65, min_blob=1,
                             iou_thr=0.3, overlap_frac=0.1)
    _, _, _, ptp1, pfp1, pfn1 = object_counts(prob, label, 255, 0.65, 1, 0, 0.3)
    assert (tc1.pix_tp, tc1.pix_fp, tc1.pix_fn) == (ptp1, pfp1, pfn1)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_row_sums_counts_and_flags_low_sample():
    counts = [
        TileCounts(obj_tp=3, obj_fp=1, obj_fn=2, n_pred=4, matched_ious=[0.5, 0.6, 0.7]),
        TileCounts(obj_tp=0, obj_fp=2, obj_fn=0, n_pred=2),
    ]
    out = aggregate_row(counts, ["a", "b"], {"a": "R1", "b": "R2"}, n_boot=20, seed=42)
    agg = out["aggregate"]
    assert (agg["obj_tp"], agg["obj_fp"], agg["obj_fn"]) == (3, 3, 2)
    assert agg["obj_precision"] == 0.5
    assert agg["obj_recall"] == 0.6
    assert agg["n_pred_objects"] == 6
    assert agg["geometry"]["n_matched"] == 3
    # Regions with no GT report None recall rather than a spurious 0, per score_by_region.
    assert out["per_region"]["R2"]["obj_recall"] is None
    assert out["per_region"]["R2"]["low_sample"] is True
    # R1 has 5 GT objects, exactly the LOW_SAMPLE_MIN_OBJECTS bound (flag is n_gt < 5).
    assert out["per_region"]["R1"]["low_sample"] is False


# ---------------------------------------------------------------------------
# SSoT constant loading
# ---------------------------------------------------------------------------


def test_load_product_constants_reads_the_shipped_rule():
    got = load_product_constants(
        PRODUCT_CONSTANTS_FILE, ("TIER_BOUNDS", "CANDIDATE_MAX_AREA_M2"))
    assert got["TIER_BOUNDS"] == (0.45, 0.65)
    assert got["CANDIDATE_MAX_AREA_M2"] == 500.0


def test_load_product_constants_raises_on_missing_name():
    with pytest.raises(KeyError):
        load_product_constants(PRODUCT_CONSTANTS_FILE, ("NOT_A_CONSTANT",))
