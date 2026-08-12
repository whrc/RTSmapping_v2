"""Object-level scorecard on cached predictions — report-only, no GPU.

The trustworthy object-level instrument for the v3 lever decision (plan: Phase 0A).
Runs on ANY cached ``*_probs.npz`` (held-out val, frozen test, or the in-sample
train sample), so the same scorecard scores every split identically.

Per region AND aggregate it reports, at the deployed operating point:
  - **signal typology** (detected / recoverable / perception-invisible) + per-region
    invisible floor — reuses ``analyze_residual_errors.object_typology``.
  - **obj precision / recall / F1** (point) — reuses ``score_by_region`` (the parity
    SSoT) — plus **per-region tile-cluster bootstrap CIs**.
  - **split / merge counts** (over- / under-segmentation) and **matched-pair IoU
    geometry** — via ``object_detail_counts`` / ``_object_match_detail``.
  - **low-sample flag** (n_gt_objects < 5 → region is "unmeasurable").
  - a **self-check**: detail-path obj P/R/F1 must equal the frozen ``score_by_region``
    path (guards the new detail aggregation against drift).

FACTS ONLY: counts, metrics, CIs — no recommendations. Writes JSON only; never
touches configs/deployment.yaml.

Run:
    python scripts/object_scorecard.py \
        --cache /mnt/outputs/v1.0/.../val_probs.npz \
        --metadata /mnt/outputs/v1.0/data_local/metadata.csv \
        --out /mnt/outputs/v1.0/diagnostics \
        --tag heldout_val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.label_cleaning import apply_min_mapping_unit  # noqa: E402
from data.splits import load_metadata  # noqa: E402
from scripts.analyze_residual_errors import (  # noqa: E402
    bootstrap_region_object_ci,
    object_detail_counts,
    object_typology,
    score_by_region,
    _geometry_summary,
    _prf_counts,
)
from utils.config import vectorize_min_blob_px  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

LOW_SAMPLE_MIN_OBJECTS = 5


def _region_typology_floors(obj_rows: list[dict]) -> dict[str, dict]:
    """Per-region invisible/recoverable fractions from the typology object rows."""
    by_region: dict[str, dict[str, int]] = {}
    for r in obj_rows:
        d = by_region.setdefault(r["region"], {"n": 0, "invisible": 0, "recoverable": 0})
        d["n"] += 1
        if r["signal_class"] == "perception_invisible":
            d["invisible"] += 1
        elif r["signal_class"] == "recoverable_below_deploy":
            d["recoverable"] += 1
    out: dict[str, dict] = {}
    for region, d in by_region.items():
        n = d["n"]
        out[region] = {
            "n_gt_objects": n,
            "invisible_floor": round(d["invisible"] / n, 4) if n else None,
            "recoverable_fraction": round(d["recoverable"] / n, 4) if n else None,
        }
    return out


def build_scorecard(
    probs: np.ndarray,
    labels: np.ndarray,
    tids: list[str],
    tid_region: dict[str, str],
    *,
    ignore_index: int,
    thr: float,
    min_blob: int,
    iou_thr: float,
    low_thr: float,
    overlap_frac: float,
    n_boot: int,
    seed: int,
) -> dict:
    """Assemble the per-region + aggregate object scorecard (see module docstring)."""
    # --- point P/R/F1 + parity SSoT (the exact Finding-K object machinery) ---
    region_pt = score_by_region(
        probs, labels, tids, tid_region,
        ignore_index=ignore_index, thr=thr, min_blob=min_blob, iou_thr=iou_thr,
    )

    # --- typology (per-object signal classes) + per-region floors ---
    obj_rows, typ_summary = object_typology(
        probs, labels, tids, tid_region,
        ignore_index=ignore_index, low_thr=low_thr, deploy_thr=thr,
        min_blob=min_blob, iou_thr=iou_thr,
    )
    region_floors = _region_typology_floors(obj_rows)

    # --- detail pass: split/merge, geometry, per-tile counts for bootstrap ---
    per_region_tiles: dict[str, list[tuple[int, int, int]]] = {}
    per_region_sm: dict[str, dict] = {}          # splits/merges accumulators
    per_region_iou: dict[str, list[float]] = {}
    all_tiles: list[tuple[int, int, int]] = []
    all_sm = {"n_splits": 0, "n_merges": 0}
    all_iou: list[float] = []
    for prob, label, tid in zip(probs, labels, tids):
        region = tid_region.get(tid, "UNKNOWN")
        otp, ofp, ofn, ns, nm, ious = object_detail_counts(
            prob, label, ignore_index, thr, min_blob, iou_thr, overlap_frac,
        )
        per_region_tiles.setdefault(region, []).append((otp, ofp, ofn))
        sm = per_region_sm.setdefault(region, {"n_splits": 0, "n_merges": 0})
        sm["n_splits"] += ns; sm["n_merges"] += nm
        per_region_iou.setdefault(region, []).extend(ious)
        all_tiles.append((otp, ofp, ofn))
        all_sm["n_splits"] += ns; all_sm["n_merges"] += nm
        all_iou.extend(ious)

    def _region_block(region: str) -> dict:
        pt = region_pt["per_region"][region]
        tiles = per_region_tiles.get(region, [])
        n_gt = pt["n_gt_objects"]
        block = {
            "n_gt_objects": n_gt,
            "low_sample": n_gt < LOW_SAMPLE_MIN_OBJECTS,
            "obj_tp": pt["obj_tp"], "obj_fp": pt["obj_fp"], "obj_fn": pt["obj_fn"],
            "obj_precision": pt["obj_precision"],
            "obj_recall": pt["obj_recall"],
            "obj_f1": pt["obj_f1"],
            "obj_ci": bootstrap_region_object_ci(tiles, n_boot=n_boot, seed=seed),
            "n_splits": per_region_sm.get(region, {}).get("n_splits", 0),
            "n_merges": per_region_sm.get(region, {}).get("n_merges", 0),
            "geometry": _geometry_summary(per_region_iou.get(region, [])),
            "typology": region_floors.get(region),
        }
        return block

    per_region = {r: _region_block(r) for r in sorted(region_pt["per_region"])}

    # --- aggregate ---
    all_pt = region_pt["ALL"]
    aggregate = {
        "n_gt_objects": all_pt["n_gt_objects"],
        "obj_tp": all_pt["obj_tp"], "obj_fp": all_pt["obj_fp"], "obj_fn": all_pt["obj_fn"],
        "obj_precision": all_pt["obj_precision"],
        "obj_recall": all_pt["obj_recall"],
        "obj_f1": all_pt["obj_f1"],
        "obj_ci": bootstrap_region_object_ci(all_tiles, n_boot=n_boot, seed=seed),
        "n_splits": all_sm["n_splits"], "n_merges": all_sm["n_merges"],
        "geometry": _geometry_summary(all_iou),
        "typology": {
            "counts": typ_summary["counts"],
            "perception_invisible_floor": typ_summary["perception_invisible_floor"],
            "recoverable_fraction": typ_summary["recoverable_fraction"],
        },
    }

    # --- self-check: detail aggregation must equal the score_by_region SSoT ---
    s = np.asarray(all_tiles, dtype=np.int64).sum(0) if all_tiles else np.zeros(3, dtype=np.int64)
    dp, dr, df = _prf_counts(int(s[0]), int(s[1]), int(s[2]))
    parity_ok = (
        int(s[0]) == all_pt["obj_tp"]
        and int(s[1]) == all_pt["obj_fp"]
        and int(s[2]) == all_pt["obj_fn"]
    )
    self_check = {
        "detail_vs_score_by_region_counts_match": bool(parity_ok),
        "detail_obj_tp_fp_fn": [int(s[0]), int(s[1]), int(s[2])],
        "score_by_region_obj_tp_fp_fn": [all_pt["obj_tp"], all_pt["obj_fp"], all_pt["obj_fn"]],
        "detail_obj_recall": round(dr, 4),
    }

    return {
        "config": {
            "threshold": thr, "min_blob_size": min_blob, "object_iou_threshold": iou_thr,
            "low_thr": low_thr, "overlap_frac": overlap_frac,
            "n_boot": n_boot, "seed": seed,
            "low_sample_min_objects": LOW_SAMPLE_MIN_OBJECTS,
        },
        "aggregate": aggregate,
        "per_region": per_region,
        "self_check": self_check,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", required=True, help="*_probs.npz (tids, probs, labels)")
    p.add_argument("--metadata", required=True, help="metadata.csv for RegionName map")
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--tag", default="scorecard",
                   help="label for this split (e.g. heldout_val, frozen_test, insample_train)")
    p.add_argument("--deployment-yaml", default="configs/deployment.yaml")
    p.add_argument("--min-blob", type=int, default=None,
                   help="PREDICTION-side size floor (px): predicted blobs smaller than this are "
                        "dropped before matching. Default: deployment.yaml vectorize_min_blob_px. "
                        "NOT the same as --min-mapping-unit, which filters GT.")
    p.add_argument("--min-mapping-unit", type=int, default=0,
                   help="GROUND-TRUTH-side size floor (px): GT positive components smaller than "
                        "this are relabeled to ignore before scoring (0 = off). Frozen-model "
                        "re-score; no retrain. NOT the same as --min-blob, which filters predictions.")
    p.add_argument("--low-thr", type=float, default=0.3)
    p.add_argument("--iou-thr", type=float, default=0.3)
    p.add_argument("--overlap-frac", type=float, default=0.1,
                   help="min intersection/smaller-blob fraction to count a split/merge association")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(level="INFO", log_file=str(out_dir / f"object_scorecard_{args.tag}.log"))

    dep = yaml.safe_load(Path(args.deployment_yaml).read_text())
    thr = float(dep["threshold"])
    min_blob = int(args.min_blob) if args.min_blob is not None else vectorize_min_blob_px(dep, 10)

    logger.info("Loading cached predictions: %s", args.cache)
    z = np.load(args.cache, allow_pickle=True)
    tids = [str(t) for t in z["tids"]]
    probs, labels = z["probs"], z["labels"]
    if args.min_mapping_unit > 1:
        # Sub-Minimum-Mapping-Unit GT positives → ignore, once, so all three scoring
        # paths (score_by_region / typology / detail) see identical labels (parity holds).
        labels = np.stack([
            apply_min_mapping_unit(lab, args.min_mapping_unit, ignore_index=args.ignore_index)
            for lab in labels
        ])
    logger.info("%d tiles | tag=%s thr=%.3f min_blob=%d min_mapping_unit=%d",
                len(tids), args.tag, thr, min_blob, args.min_mapping_unit)

    meta = load_metadata(args.metadata)
    tid_region = dict(zip(meta["Tile_ID"], meta["RegionName"]))

    scorecard = build_scorecard(
        probs, labels, tids, tid_region,
        ignore_index=args.ignore_index, thr=thr, min_blob=min_blob, iou_thr=args.iou_thr,
        low_thr=args.low_thr, overlap_frac=args.overlap_frac, n_boot=args.n_boot, seed=args.seed,
    )
    scorecard["_tag"] = args.tag
    scorecard["_source_cache"] = args.cache
    scorecard["config"]["min_mapping_unit"] = args.min_mapping_unit

    if not scorecard["self_check"]["detail_vs_score_by_region_counts_match"]:
        logger.error("SELF-CHECK FAILED: detail counts != score_by_region: %s",
                     scorecard["self_check"])

    mmu_sfx = f"_mmu{args.min_mapping_unit}" if args.min_mapping_unit > 1 else ""
    out_path = out_dir / f"object_scorecard_{args.tag}{mmu_sfx}.json"
    out_path.write_text(json.dumps(scorecard, indent=2))
    logger.info("Wrote %s", out_path)

    agg = scorecard["aggregate"]
    print(json.dumps({
        "tag": args.tag,
        "min_mapping_unit": args.min_mapping_unit,
        "n_gt_objects": agg["n_gt_objects"],
        "obj_precision": agg["obj_precision"],
        "obj_recall": agg["obj_recall"],
        "obj_recall_CI": [agg["obj_ci"]["recall"]["lo"], agg["obj_ci"]["recall"]["hi"]],
        "obj_f1": agg["obj_f1"],
        "invisible_floor": agg["typology"]["perception_invisible_floor"],
        "n_splits": agg["n_splits"], "n_merges": agg["n_merges"],
        "geometry_iou_median": (agg["geometry"] or {}).get("iou_median"),
        "self_check_ok": scorecard["self_check"]["detail_vs_score_by_region_counts_match"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
