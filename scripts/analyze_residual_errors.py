"""Residual-error diagnostics on cached Val-Realistic predictions — report-only.

Two facts-only analyses on the already-cached 3-seed ensemble val predictions
(``object_operating_point/effb5_ensemble/val_probs.npz`` — per-tile 2-D probs +
labels on the deployed temperature scale). No forward pass, no GPU.

A. **FN typology by model signal** (the v3 go/no-go diagnostic). For every GT
   object, take the *max predicted probability inside its footprint* and bin it:
     - ``>= deploy_thr``           → ``detected_at_deploy``  (model fires above the
                                      deployed threshold over the object)
     - ``[low_thr, deploy_thr)``   → ``recoverable_below_deploy`` (sub-threshold
                                      signal — a free post-hoc threshold fix)
     - ``< low_thr``               → ``perception_invisible``  (model produces ~no
                                      signal — the population that adjudicates v3)
   max-prob is min_blob/IoU-independent: it answers "can thresholding recover
   this object", which is exactly the recall-ceiling question. We also tag each
   object's actual object-match status at the deployed point for cross-check.

B. **Per-region stratified scoring.** Group val tiles by RegionName and report
   pixel + object P/R/F1 per region at the deployed operating point, reusing the
   exact ``tune_object_operating_point.object_counts`` machinery. Computed at
   min_blob=10 (the report's deployed-block anchor, for the parity check) and at
   the product min_blob from deployment.yaml.

FACTS ONLY: counts and metrics; no recommendations. REPORT-ONLY: writes JSON
only, never touches configs/deployment.yaml. Val-Realistic ONLY.

Run:
    python scripts/analyze_residual_errors.py \
        --cache /mnt/outputs/v1.0/object_operating_point/effb5_ensemble/val_probs.npz \
        --metadata /mnt/outputs/v1.0/data_local/metadata.csv \
        --out /mnt/outputs/v1.0/diagnostics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.label_cleaning import apply_min_mapping_unit  # noqa: E402
from data.splits import load_metadata  # noqa: E402
from training.metrics import (  # noqa: E402
    _filter_small_blobs,
    _match_objects,
    _object_match_detail,
    _safe_div,
)
from utils.config import vectorize_min_blob_px  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

_EPS = 1e-6


# Object machinery — mirrors scripts/tune_object_operating_point.py:127-166
# (imported here directly from training.metrics to avoid that module's
# calibrate→dataset→albumentations import chain, which is broken in the image).


def _pred_mask(prob: np.ndarray, valid: np.ndarray, thr: float, morph_r: int) -> np.ndarray:
    """Binary prediction at (thr, morph close radius), restricted to valid."""
    pred = (prob >= thr) & valid
    if morph_r > 0:
        struct = ndimage.iterate_structure(ndimage.generate_binary_structure(2, 1), morph_r)
        pred = ndimage.binary_closing(pred, structure=struct) & valid
    return pred


def object_counts(
    prob: np.ndarray, label: np.ndarray, ignore_index: int,
    thr: float, min_blob: int, morph_r: int, iou_thr: float,
) -> tuple[int, int, int, int, int, int]:
    """One tile → (obj_tp, obj_fp, obj_fn, pix_tp, pix_fp, pix_fn)."""
    valid = label != ignore_index
    gt = (label == 1) & valid
    pred = _pred_mask(prob, valid, thr, morph_r)
    pix_tp = int(np.logical_and(pred, gt).sum())
    pix_fp = int(np.logical_and(pred, np.logical_not(gt) & valid).sum())
    pix_fn = int(np.logical_and(np.logical_not(pred) & valid, gt).sum())
    pred_filt = _filter_small_blobs(pred.astype(np.uint8), min_blob)
    pred_labels, n_pred = ndimage.label(pred_filt)
    gt_labels, n_gt = ndimage.label(gt.astype(np.uint8))
    conf = (np.array(ndimage.mean(prob, pred_labels, index=np.arange(1, n_pred + 1)),
                     dtype=np.float64) if n_pred > 0 else np.zeros(0))
    tp, fp, fn = _match_objects(pred_labels, n_pred, gt_labels, n_gt, conf, iou_thr)
    return tp, fp, fn, pix_tp, pix_fp, pix_fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return p, r, _safe_div(2 * p * r, p + r)


# ---------------------------------------------------------------------------
# Analysis A — per-GT-object signal typology
# ---------------------------------------------------------------------------


def object_typology(
    probs: np.ndarray,
    labels: np.ndarray,
    tids: list[str],
    tid_region: dict[str, str],
    *,
    ignore_index: int,
    low_thr: float,
    deploy_thr: float,
    min_blob: int,
    iou_thr: float,
) -> tuple[list[dict], dict]:
    """Classify every GT object by the model's max signal inside its footprint.

    Returns:
        (per_object rows, summary). Each row: tile_id, region, area_px, max_prob,
        mean_prob, signal_class, matched_at_deploy (object-match TP at the deployed
        point — cross-check that max-prob signal actually yields a detection).
    """
    rows: list[dict] = []
    for prob, label, tid in zip(probs, labels, tids):
        valid = label != ignore_index
        gt = (label == 1) & valid
        gt_labels, n_gt = ndimage.label(gt.astype(np.uint8))
        if n_gt == 0:
            continue

        # Which GT components are matched (object-level TP) at the deployed point.
        pred = _pred_mask(prob, valid, deploy_thr, 0)
        pred_filt = _filter_small_blobs(pred.astype(np.uint8), min_blob)
        pred_labels, n_pred = ndimage.label(pred_filt)
        matched_gt: set[int] = set()
        if n_pred > 0:
            conf = np.array(
                ndimage.mean(prob, pred_labels, index=np.arange(1, n_pred + 1)),
                dtype=np.float64,
            )
            # _match_objects greedily fills matched GT; recover which GT got matched
            # by replaying its logic on the IoU matrix it uses.
            iou = np.zeros((n_pred, n_gt))
            for p in range(1, n_pred + 1):
                pm = pred_labels == p
                for g in range(1, n_gt + 1):
                    inter = int(np.logical_and(pm, gt_labels == g).sum())
                    if inter:
                        iou[p - 1, g - 1] = inter / int(
                            np.logical_or(pm, gt_labels == g).sum()
                        )
            for p in np.argsort(conf)[::-1]:
                row = iou[p].copy()
                for g in matched_gt:
                    row[g] = 0.0
                g = int(np.argmax(row))
                if row[g] >= iou_thr:
                    matched_gt.add(g)

        for g in range(1, n_gt + 1):
            footprint = gt_labels == g
            obj_probs = prob[footprint]
            max_p = float(obj_probs.max())
            if max_p >= deploy_thr:
                cls = "detected_at_deploy"
            elif max_p >= low_thr:
                cls = "recoverable_below_deploy"
            else:
                cls = "perception_invisible"
            rows.append({
                "tile_id": tid,
                "region": tid_region.get(tid, "UNKNOWN"),
                "area_px": int(footprint.sum()),
                "max_prob": round(max_p, 4),
                "mean_prob": round(float(obj_probs.mean()), 4),
                "signal_class": cls,
                "matched_at_deploy": bool((g - 1) in matched_gt),
            })

    classes = ["detected_at_deploy", "recoverable_below_deploy", "perception_invisible"]
    n_total = len(rows)
    summary = {
        "n_gt_objects": n_total,
        "low_thr": low_thr,
        "deploy_thr": deploy_thr,
        "counts": {c: sum(r["signal_class"] == c for r in rows) for c in classes},
        "perception_invisible_floor": round(
            sum(r["signal_class"] == "perception_invisible" for r in rows) / n_total, 4
        ) if n_total else 0.0,
        "recoverable_fraction": round(
            sum(r["signal_class"] == "recoverable_below_deploy" for r in rows) / n_total, 4
        ) if n_total else 0.0,
    }
    # Size + region distribution of the perception-invisible set (routes v3).
    inv = [r for r in rows if r["signal_class"] == "perception_invisible"]
    if inv:
        areas = np.array([r["area_px"] for r in inv])
        summary["perception_invisible_area_px"] = {
            "p50": int(np.percentile(areas, 50)),
            "p90": int(np.percentile(areas, 90)),
            "max": int(areas.max()),
        }
        by_region: dict[str, int] = {}
        for r in inv:
            by_region[r["region"]] = by_region.get(r["region"], 0) + 1
        summary["perception_invisible_by_region"] = by_region
    return rows, summary


# ---------------------------------------------------------------------------
# Analysis B — per-region stratified scoring
# ---------------------------------------------------------------------------


def score_by_region(
    probs: np.ndarray,
    labels: np.ndarray,
    tids: list[str],
    tid_region: dict[str, str],
    *,
    ignore_index: int,
    thr: float,
    min_blob: int,
    iou_thr: float,
) -> dict:
    """Aggregate pixel + object metrics per region (and an ALL roll-up).

    Reuses ``object_counts`` (the exact training.metrics object machinery) per
    tile, summed within each region. The ALL roll-up summed over regions must
    equal the global aggregate — that is the parity check.
    """
    agg: dict[str, dict[str, int]] = {}
    for prob, label, tid in zip(probs, labels, tids):
        region = tid_region.get(tid, "UNKNOWN")
        otp, ofp, ofn, ptp, pfp, pfn = object_counts(
            prob, label, ignore_index, thr, min_blob, 0, iou_thr
        )
        a = agg.setdefault(region, dict(otp=0, ofp=0, ofn=0, ptp=0, pfp=0, pfn=0))
        a["otp"] += otp; a["ofp"] += ofp; a["ofn"] += ofn
        a["ptp"] += ptp; a["pfp"] += pfp; a["pfn"] += pfn

    def _metrics(a: dict[str, int]) -> dict:
        op, orc, of1 = _prf(a["otp"], a["ofp"], a["ofn"])
        pp, prc, pf1 = _prf(a["ptp"], a["pfp"], a["pfn"])
        n_pos_obj = a["otp"] + a["ofn"]      # GT objects present
        n_pos_pix = a["ptp"] + a["pfn"]      # GT positive pixels present
        # Recall/F1 are undefined with no positives → report None, not a spurious 0.
        m = {
            "n_gt_objects": n_pos_obj,
            "obj_tp": a["otp"], "obj_fp": a["ofp"], "obj_fn": a["ofn"],
            "obj_precision": round(op, 4) if (a["otp"] + a["ofp"]) else None,
            "obj_recall": round(orc, 4) if n_pos_obj else None,
            "obj_f1": round(of1, 4) if n_pos_obj and (a["otp"] + a["ofp"]) else None,
            "pixel_precision": round(pp, 4) if (a["ptp"] + a["pfp"]) else None,
            "pixel_recall": round(prc, 4) if n_pos_pix else None,
        }
        return m

    per_region = {r: _metrics(a) for r, a in sorted(agg.items())}
    total = {k: sum(a[k] for a in agg.values())
             for k in ("otp", "ofp", "ofn", "ptp", "pfp", "pfn")}
    return {"threshold": thr, "min_blob_size": min_blob,
            "per_region": per_region, "ALL": _metrics(total)}


# ---------------------------------------------------------------------------
# Object scorecard helpers — split/merge, geometry, per-region bootstrap CIs
# (report-only; point P/R/F1 + parity stay owned by score_by_region above)
# ---------------------------------------------------------------------------


def object_detail_counts(
    prob: np.ndarray, label: np.ndarray, ignore_index: int,
    thr: float, min_blob: int, iou_thr: float, overlap_frac: float = 0.1,
) -> tuple[int, int, int, int, int, list[float]]:
    """One tile → (obj_tp, obj_fp, obj_fn, n_splits, n_merges, matched_ious).

    Same prediction/labeling path as ``object_counts`` (morph_r=0), but routed
    through ``_object_match_detail`` for the split/merge + matched-geometry readout.
    obj tp/fp/fn are bit-identical to ``object_counts``.
    """
    valid = label != ignore_index
    gt = (label == 1) & valid
    pred = _pred_mask(prob, valid, thr, 0)
    pred_filt = _filter_small_blobs(pred.astype(np.uint8), min_blob)
    pred_labels, n_pred = ndimage.label(pred_filt)
    gt_labels, n_gt = ndimage.label(gt.astype(np.uint8))
    conf = (np.array(ndimage.mean(prob, pred_labels, index=np.arange(1, n_pred + 1)),
                     dtype=np.float64) if n_pred > 0 else np.zeros(0))
    d = _object_match_detail(
        pred_labels, n_pred, gt_labels, n_gt, conf, iou_thr, overlap_frac,
    )
    return d.tp, d.fp, d.fn, d.n_splits, d.n_merges, d.matched_ious


def _geometry_summary(ious: list[float]) -> dict | None:
    """Matched-pair IoU distribution (object-geometry quality of detections)."""
    if not ious:
        return None
    a = np.asarray(ious, dtype=np.float64)
    return {
        "n_matched": int(a.size),
        "iou_median": round(float(np.median(a)), 4),
        "iou_p10": round(float(np.percentile(a, 10)), 4),
        "iou_p90": round(float(np.percentile(a, 90)), 4),
        "iou_mean": round(float(a.mean()), 4),
    }


def _prf_counts(otp: int, ofp: int, ofn: int) -> tuple[float, float, float]:
    p = _safe_div(otp, otp + ofp)
    r = _safe_div(otp, otp + ofn)
    return p, r, _safe_div(2 * p * r, p + r)


def bootstrap_region_object_ci(
    tile_counts: list[tuple[int, int, int]],
    *, n_boot: int = 1000, seed: int = 42, ci: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Cluster-bootstrap obj precision/recall/F1 CIs over TILES within a region.

    ``tile_counts``: list of per-tile (obj_tp, obj_fp, obj_fn). Tiles are the
    resampling unit (objects within a tile are spatially correlated), so this is a
    tile-cluster bootstrap. Recall/F1 are only meaningful when the region has GT
    objects; callers gate display on n_gt_objects.
    """
    T = len(tile_counts)
    arr = np.asarray(tile_counts, dtype=np.float64).reshape(T, 3) if T else np.zeros((0, 3))
    s = arr.sum(0) if T else np.zeros(3)
    pt_p, pt_r, pt_f = _prf_counts(int(s[0]), int(s[1]), int(s[2]))
    if T == 0:
        none3 = {"point": None, "lo": None, "hi": None}
        return {"n_tiles": 0, "precision": none3, "recall": none3, "f1": none3}
    rng = np.random.default_rng(seed)
    ps, rs, fs = [], [], []
    for _ in range(n_boot):
        ss = arr[rng.integers(0, T, size=T)].sum(0)
        p, r, f = _prf_counts(ss[0], ss[1], ss[2])
        ps.append(p); rs.append(r); fs.append(f)

    def _ci(vals: list[float], pt: float) -> dict:
        v = np.asarray(vals)
        return {"point": round(pt, 4),
                "lo": round(float(np.percentile(v, ci[0])), 4),
                "hi": round(float(np.percentile(v, ci[1])), 4)}

    return {"n_tiles": T,
            "precision": _ci(ps, pt_p), "recall": _ci(rs, pt_r), "f1": _ci(fs, pt_f)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", required=True, help="val_probs.npz (tids, probs, labels)")
    p.add_argument("--metadata", required=True, help="metadata.csv for RegionName map")
    p.add_argument("--out", required=True, help="output dir for residual_errors_report.json")
    p.add_argument("--deployment-yaml", default="configs/deployment.yaml")
    p.add_argument("--low-thr", type=float, default=0.3,
                   help="recoverable/invisible boundary (default 0.3)")
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument("--iou-thr", type=float, default=0.3, help="object match IoU")
    p.add_argument("--parity-min-blob", type=int, default=10,
                   help="min_blob matching the report deployed-block (parity check)")
    p.add_argument("--min-mapping-unit", type=int, default=0,
                   help="Minimum Mapping Unit (px): GT positive components smaller than this are "
                        "relabeled to ignore before scoring (0 = off). Frozen-model re-score.")
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_dir / "analyze_residual_errors.log"))

    dep = yaml.safe_load(Path(args.deployment_yaml).read_text())
    deploy_thr = float(dep["threshold"])
    product_min_blob = vectorize_min_blob_px(dep, 10)

    logger.info("Loading cached val predictions: %s", args.cache)
    z = np.load(args.cache, allow_pickle=True)
    tids = [str(t) for t in z["tids"]]
    probs, labels = z["probs"], z["labels"]
    if args.min_mapping_unit > 1:
        labels = np.stack([
            apply_min_mapping_unit(lab, args.min_mapping_unit, ignore_index=args.ignore_index)
            for lab in labels
        ])
    logger.info("%d val tiles | deploy_thr=%.3f product_min_blob=%d min_mapping_unit=%d",
                len(tids), deploy_thr, product_min_blob, args.min_mapping_unit)

    meta = load_metadata(args.metadata)
    tid_region = dict(zip(meta["Tile_ID"], meta["RegionName"]))

    # Analysis A — object typology (min_blob = product point for the match cross-check).
    obj_rows, typology_summary = object_typology(
        probs, labels, tids, tid_region,
        ignore_index=args.ignore_index, low_thr=args.low_thr, deploy_thr=deploy_thr,
        min_blob=product_min_blob, iou_thr=args.iou_thr,
    )
    logger.info("Typology: %s | invisible_floor=%.3f",
                typology_summary["counts"], typology_summary["perception_invisible_floor"])

    # Analysis B — per-region scoring at the parity point and the product point.
    region_parity = score_by_region(
        probs, labels, tids, tid_region, ignore_index=args.ignore_index,
        thr=deploy_thr, min_blob=args.parity_min_blob, iou_thr=args.iou_thr,
    )
    region_product = score_by_region(
        probs, labels, tids, tid_region, ignore_index=args.ignore_index,
        thr=deploy_thr, min_blob=product_min_blob, iou_thr=args.iou_thr,
    )

    report = {
        "_source_cache": args.cache,
        "_min_mapping_unit": args.min_mapping_unit,
        "_caveats": [
            "n GT objects is small (~val positives) — typology is qualitative, "
            "not proportions with CIs.",
            "max_prob typology is min_blob/IoU-independent; matched_at_deploy is "
            "the object-match cross-check at the deployed point.",
            "per-region: few regions = wide region-level variance.",
        ],
        "analysis_A_typology": {"summary": typology_summary, "objects": obj_rows},
        "analysis_B_per_region": {
            "parity_point": region_parity,
            "product_point": region_product,
        },
    }
    out_path = out_dir / "residual_errors_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", out_path)

    print(json.dumps({
        "typology": typology_summary["counts"],
        "perception_invisible_floor": typology_summary["perception_invisible_floor"],
        "recoverable_fraction": typology_summary["recoverable_fraction"],
        "per_region_ALL_parity(min_blob=%d)" % args.parity_min_blob: region_parity["ALL"],
        "per_region_obj_f1_product(min_blob=%d)" % product_min_blob: {
            r: m["obj_f1"] for r, m in region_product["per_region"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
