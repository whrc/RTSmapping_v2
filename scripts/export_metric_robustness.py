#!/usr/bin/env python3
"""Export the hypothesis-test battery to outputs/metric_robustness.csv (the manuscript deliverable).

One row per (run, seed) for every in-scope run (existing comparators + the C1–C4 new runs), plus
per-condition aggregate rows (seed=mean / seed=std). Metrics are read at the **pr-auc best_epoch**
step, matching how the ledger reports PR-AUC (see docs/future_work/experiments_hypothesis_test.md §6).

Sources per run dir /mnt/outputs/v1.0/runs/<name>/:
  - run_summary.json  -> seed, best_epoch, best_smoothed (== pr_auc_geomean column)
  - config.yaml       -> backbone, architecture, channels, train_positive_subset_pct, _config_path
  - MLflow file store  /mnt/outputs/v1.0/mlflow/<name>/*/*/metrics/<metric>  (value at best_epoch)
       -> pixel_iou, pixel_f1, object_precision, object_recall, object_f1
Secondary metrics are NOT in run_summary.json (often None) — they must come from MLflow.
git_sha = repo HEAD at export time (torch isn't on host to read per-run checkpoint SHA).

Usage: python scripts/export_metric_robustness.py [--out outputs/metric_robustness.csv]
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, subprocess, statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = Path("/mnt/outputs/v1.0/runs")
MLFLOW_ROOT = Path("/mnt/outputs/v1.0/mlflow")
THRESHOLD = 0.5            # reporting_threshold (base_v2_fast.yaml metrics.reporting_threshold)
SPLIT = "val_realistic"    # corrected leakage-free split; metrics are the realistic-val readout

MLFLOW_METRICS = ["pixel_iou", "pixel_f1", "object_precision", "object_recall", "object_f1"]

# ---- in-scope battery (existing comparators + 23 new). Explicit = reviewable, avoids the other ~100 runs.
SCOPE = [
    # EffB5+NDVI, deploy +TrivialAugment recipe (C1/C3 headline 100%)
    "aug_trivialaugment_deploy", "aug_trivialaugment_deploy_seed43", "aug_trivialaugment_deploy_seed44",
    # EffB5+NDVI, phase0c recipe (C1 EffB5 NDVI-effect comparator)
    "phase4_extra_ndvi", "phase4_extra_ndvi_seed43", "phase4_extra_ndvi_seed44",
    # EffB5-RGB, phase0c recipe (C1) — seed42 existing + new 43/44
    "phase4_extra_rgb_baseline", "phase4_extra_rgb_baseline_seed43", "phase4_extra_rgb_baseline_seed44",
    # EffB5+NDVI, no-TrivialAugment deploy recipe (C2 B5 point)
    "deploy_v1_ndvi_seed42", "deploy_v1_ndvi_seed43", "deploy_v1_ndvi_seed44",
    # EffB3 (C2 B3 point) — seed42 existing + new 43/44
    "effb3_deploy", "effb3_deploy_seed43", "effb3_deploy_seed44",
    # C2 new capacity endpoints
    "c2_effb0_ndvi_seed42", "c2_effb0_ndvi_seed43", "c2_effb0_ndvi_seed44",
    "c2_effb7_ndvi_seed42", "c2_effb7_ndvi_seed43", "c2_effb7_ndvi_seed44",
    # ViT-L+NDVI locked (C1 / C3 100%)
    "fm_dinov3sat_l_ndvi_locked", "fm_dinov3sat_l_ndvi_locked_seed43", "fm_dinov3sat_l_ndvi_locked_seed44",
    # ViT-L-RGB off-recipe (existing; directional reference) + clean locked (C1 new)
    "fm_dinov3sat_l_rgb", "fm_dinov3sat_l_rgb_seed43", "fm_dinov3sat_l_rgb_seed44",
    "fm_dinov3sat_l_rgb_locked_seed42", "fm_dinov3sat_l_rgb_locked_seed43", "fm_dinov3sat_l_rgb_locked_seed44",
    # C3 ViT-L+NDVI data budgets
    "c3_vitl_ndvi_scale25_seed42", "c3_vitl_ndvi_scale25_seed43", "c3_vitl_ndvi_scale25_seed44",
    "c3_vitl_ndvi_scale50_seed42", "c3_vitl_ndvi_scale50_seed43", "c3_vitl_ndvi_scale50_seed44",
    # small-model data-scaling curve (C3 overlay / C4)
    "scale_ndvi_25", "scale_ndvi_25_seed43", "scale_ndvi_25_seed44",
    "scale_ndvi_50", "scale_ndvi_50_seed43", "scale_ndvi_50_seed44",
    "scale_ndvi_75", "scale_ndvi_75_seed43", "scale_ndvi_75_seed44",
]

FIELDS = ["run_name", "family", "backbone", "representation", "train_positive_subset_pct", "seed",
          "split", "best_epoch", "threshold", "pr_auc_geomean", "pixel_iou", "pixel_f1",
          "object_precision", "object_recall", "object_f1", "git_sha", "config_path"]


def repo_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def mlflow_value_at(run: str, metric: str, best_epoch: int):
    """Value of <metric> at the best_epoch step from the per-run MLflow file store."""
    hist = {}
    for p in glob.glob(str(MLFLOW_ROOT / run / "*" / "*" / "metrics" / metric)):
        for line in open(p):
            a = line.split()
            if len(a) >= 3:
                hist[int(a[2])] = float(a[1])
    if not hist:
        return None
    if best_epoch in hist:
        return hist[best_epoch]
    le = [s for s in hist if s <= best_epoch]
    return hist[max(le)] if le else hist[max(hist)]


def family_of(backbone: str, arch: str) -> str:
    if arch == "foundation" or backbone.startswith("vit_"):
        return "dinov3sat_vitl"
    if backbone.startswith("efficientnet"):
        return "efficientnet"
    return arch or backbone


def condition_key(run_name: str) -> str:
    """Collapse seed siblings: strip a trailing _seed\\d+; unsuffixed name is the seed-42 member."""
    return re.sub(r"_seed\d+$", "", run_name)


def row_for(run: str, git_sha: str):
    d = RUNS_DIR / run
    summ_p, cfg_p = d / "run_summary.json", d / "config.yaml"
    if not summ_p.exists() or not cfg_p.exists():
        return None
    import yaml
    summ = json.load(open(summ_p))
    cfg = yaml.safe_load(open(cfg_p))
    best_epoch = summ.get("best_epoch")
    backbone = cfg.get("model", {}).get("backbone", "")
    arch = cfg.get("model", {}).get("architecture", "")
    extra = cfg.get("channels", {}).get("extra") or []
    pct = cfg.get("splits", {}).get("train_positive_subset_pct")
    row = {
        "run_name": run,
        "family": family_of(backbone, arch),
        "backbone": backbone,
        "representation": "RGB+NDVI" if extra else "RGB",
        "train_positive_subset_pct": 100 if pct in (None, "") else int(pct),
        "seed": summ.get("seed"),
        "split": SPLIT,
        "best_epoch": best_epoch,
        "threshold": THRESHOLD,
        "pr_auc_geomean": round(summ["best_smoothed"], 6) if summ.get("best_smoothed") is not None else None,
        "git_sha": git_sha,
        "config_path": cfg.get("_config_path", ""),
    }
    for m in MLFLOW_METRICS:
        v = mlflow_value_at(run, m, best_epoch) if best_epoch is not None else None
        row[m] = round(v, 6) if v is not None else None
    return row


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Per-condition seed=mean and seed=std rows over the numeric metric columns."""
    metrics = ["pr_auc_geomean"] + MLFLOW_METRICS
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(condition_key(r["run_name"]), []).append(r)
    out = []
    for cond, rs in groups.items():
        base = rs[0]
        for tag, fn in (("mean", statistics.mean),
                        ("std", (lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0))):
            agg = {k: "" for k in FIELDS}
            agg.update({"run_name": cond, "family": base["family"], "backbone": base["backbone"],
                        "representation": base["representation"],
                        "train_positive_subset_pct": base["train_positive_subset_pct"],
                        "seed": tag, "split": SPLIT, "threshold": THRESHOLD,
                        "git_sha": base["git_sha"], "config_path": f"n={len(rs)}"})
            for m in metrics:
                vals = [r[m] for r in rs if r.get(m) is not None]
                agg[m] = round(fn(vals), 6) if vals else None
            out.append(agg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "outputs" / "metric_robustness.csv"))
    args = ap.parse_args()
    git_sha = repo_head()

    per_seed, missing = [], []
    for run in SCOPE:
        r = row_for(run, git_sha)
        (per_seed if r else missing).append(r if r else run)

    all_rows = per_seed + aggregate_rows(per_seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    conds = sorted({condition_key(r["run_name"]) for r in per_seed})
    print(f"wrote {out}")
    print(f"  per-seed rows: {len(per_seed)}  | aggregate rows: {len(all_rows)-len(per_seed)}  "
          f"| conditions: {len(conds)}")
    if missing:
        print(f"  MISSING (no run_summary.json yet — will fill in once complete): {len(missing)}")
        for m in missing:
            print(f"    - {m}")


if __name__ == "__main__":
    main()
