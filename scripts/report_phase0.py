"""Generate a self-contained HTML report for Phase 0 experiments.

Queries MLflow for all runs whose run_name starts with 'phase0', then renders:
  - Phase 0a: normalization arm comparison (A / B / C)
  - Phase 0b: LR range test curves (frozen + unfrozen)
  - Phase 0c: multi-seed baseline table + σ₀ / μ₀ summary
  - Artifact locations table (every Phase 0 run)

Usage:
    python scripts/report_phase0.py \\
        --config configs/baseline.yaml \\
        --output docs/phase0_report.html

Requirements: mlflow, pandas, matplotlib (all in requirements.txt).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _connect_mlflow(tracking_uri: str):
    import mlflow
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                          os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow


def _search_runs(mlflow, experiment_name: str, prefix: str) -> pd.DataFrame:
    """Return all runs whose run_name starts with `prefix`."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=f"tags.mlflow.runName LIKE '{prefix}%'",
            order_by=["tags.mlflow.runName ASC"],
        )
    except Exception as exc:
        logger.warning("MLflow search failed (%s); returning empty frame.", exc)
        runs = pd.DataFrame()
    return runs


def _get_metric_history(mlflow, run_id: str, metric: str) -> list[tuple[int, float]]:
    """Return [(step, value), ...] for a metric in a run."""
    try:
        client = mlflow.MlflowClient()
        history = client.get_metric_history(run_id, metric)
        return [(h.step, h.value) for h in history]
    except Exception:
        return []


def _download_artifact_text(mlflow, run_id: str, artifact_name: str) -> str | None:
    """Download a small text artifact from MLflow and return its content."""
    try:
        client = mlflow.MlflowClient()
        local = client.download_artifacts(run_id, artifact_name)
        return Path(local).read_text()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _plot_metric_curves(
    histories: dict[str, list[tuple[int, float]]],
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "",
    colors: list[str] | None = None,
) -> str:
    """Plot multiple (step, value) histories; return base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = colors or ["#2563EB", "#DC2626", "#16A34A", "#9333EA"]
    for i, (label, pts) in enumerate(histories.items()):
        if not pts:
            continue
        steps, vals = zip(*pts)
        ax.plot(steps, vals, label=label, color=palette[i % len(palette)], linewidth=1.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_lr_range_curve(csv_text: str, run_name: str) -> str:
    """Plot step vs loss from lr_range_curve.csv; return base64 PNG."""
    rows = []
    for line in csv_text.strip().splitlines():
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not rows:
        return ""
    steps, lrs, losses = zip(*rows)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twiny()
    ax1.semilogy(lrs, losses, color="#2563EB", linewidth=1.8)
    ax1.set_xlabel("Learning Rate (log scale)")
    ax1.set_ylabel("Loss")
    ax2.set_xlabel("Step")
    ax2.set_xlim(0, max(steps))
    ax1.set_title(f"LR Range Test — {run_name}", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML sections
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #1e293b; }
h1   { font-size: 1.8rem; border-bottom: 3px solid #2563EB; padding-bottom: 8px; }
h2   { font-size: 1.3rem; margin-top: 2.5rem; color: #1d4ed8; }
h3   { font-size: 1.05rem; color: #374151; margin-top: 1.5rem; }
table{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
th   { background: #1d4ed8; color: #fff; padding: 8px 12px; text-align: left; }
td   { padding: 7px 12px; border-bottom: 1px solid #e2e8f0; }
tr:nth-child(even) td { background: #f8fafc; }
.winner { background: #dcfce7 !important; font-weight: 600; }
.card   { border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px;
          margin: 1rem 0; background: #f8fafc; }
.metric { font-size: 1.4rem; font-weight: 700; color: #1d4ed8; }
.label  { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
img     { max-width: 100%; border-radius: 6px; margin: 0.5rem 0; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
.todo   { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px;
          padding: 10px 14px; margin: 0.8rem 0; font-size: 0.9rem; }
.mono   { font-family: monospace; font-size: 0.85rem; }
"""


def _html_table(df: pd.DataFrame, winner_col: str | None = None,
                winner_val=None) -> str:
    if df.empty:
        return "<p><em>No data.</em></p>"
    rows_html = ""
    for _, row in df.iterrows():
        cls = ""
        if winner_col and winner_val is not None and row.get(winner_col) == winner_val:
            cls = ' class="winner"'
        cells = "".join(f"<td{cls}>{v}</td>" for v in row.values)
        rows_html += f"<tr>{cells}</tr>\n"
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>"


def _section_phase0a(mlflow, experiment_name: str) -> str:
    runs = _search_runs(mlflow, experiment_name, "phase0a_arm")
    if runs.empty:
        return "<div class='todo'>⏳ Phase 0a not yet run — no data in MLflow.</div>"

    arm_rows = []
    histories = {}
    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", "?")
        rid = run.get("run_id", "")
        best_metric = run.get("metrics.val_realistic_pr_auc_geomean", float("nan"))
        final_loss = run.get("metrics.train_loss", float("nan"))
        val_iou = run.get("metrics.val_balanced_iou", float("nan"))
        arm_rows.append({
            "Run": name,
            "Best val_realistic_pr_auc_geomean": f"{best_metric:.4f}" if not np.isnan(best_metric) else "—",
            "Final train_loss": f"{final_loss:.4f}" if not np.isnan(final_loss) else "—",
            "Val-Balanced IoU": f"{val_iou:.4f}" if not np.isnan(val_iou) else "—",
            "Run ID": f"<span class='mono'>{rid[:8]}</span>",
        })
        hist = _get_metric_history(mlflow, rid, "val_realistic_pr_auc_geomean")
        if hist:
            histories[name] = hist

    df = pd.DataFrame(arm_rows)
    # Find winner
    numerics = [r for r in arm_rows if r["Best val_realistic_pr_auc_geomean"] != "—"]
    winner_run = None
    if numerics:
        best = max(numerics, key=lambda r: float(r["Best val_realistic_pr_auc_geomean"]))
        winner_run = best["Run"]

    table_html = _html_table(df, winner_col="Run", winner_val=winner_run)

    chart = ""
    if histories:
        b64 = _plot_metric_curves(
            histories,
            title="Phase 0a — val_realistic_pr_auc_geomean per epoch",
            ylabel="PR-AUC geomean",
        )
        chart = f"<img src='data:image/png;base64,{b64}' alt='Phase 0a curves'>"

    winner_note = ""
    if winner_run:
        winner_note = (
            f"<div class='card'><span class='label'>Winner</span><br>"
            f"<span class='metric'>{winner_run}</span><br>"
            f"Lock this arm's normalization_stats_path into phase0b and phase0c configs.</div>"
        )

    return f"""
<h2>Phase 0a — RGB Normalization Arm Comparison</h2>
<p>Arm A: per-dataset z-score (baseline) &nbsp;|&nbsp;
   Arm B: ImageNet mean/std (/255 → imagenet) &nbsp;|&nbsp;
   Arm C: scale only (/255). Winner = Δ ≥ 0.01 over Arm A; tie-break C &gt; B.</p>
{table_html}
{winner_note}
{chart}
"""


def _section_phase0b(mlflow, experiment_name: str) -> str:
    runs_frozen = _search_runs(mlflow, experiment_name, "phase0b_lr_frozen")
    runs_unfrozen = _search_runs(mlflow, experiment_name, "phase0b_lr_unfrozen")

    sections = []
    for label, runs in [("Frozen backbone", runs_frozen), ("Unfrozen (full fine-tune)", runs_unfrozen)]:
        if runs.empty:
            sections.append(f"<div class='todo'>⏳ {label} LR range test not yet run.</div>")
            continue
        run = runs.iloc[0]
        rid = run.get("run_id", "")
        csv_text = _download_artifact_text(mlflow, rid, "lr_range_curve.csv")
        if csv_text:
            b64 = _plot_lr_range_curve(csv_text, run.get("tags.mlflow.runName", label))
            sections.append(f"<h3>{label}</h3><img src='data:image/png;base64,{b64}' alt='LR range {label}'>")
        else:
            sections.append(f"<div class='todo'>⏳ {label}: lr_range_curve.csv not found in MLflow artifacts.</div>")

    return f"""
<h2>Phase 0b — LR Range Test</h2>
<p>Pick <code>frozen_lr</code> (frozen run) and <code>base_lr</code> (unfrozen run) at the steepest
stable loss descent before divergence. Update phase0b and phase0c configs before running 0c.</p>
{"".join(sections)}
"""


def _section_phase0c(mlflow, experiment_name: str) -> str:
    runs = _search_runs(mlflow, experiment_name, "phase0c_seed")
    if runs.empty:
        return "<div class='todo'>⏳ Phase 0c multi-seed baseline not yet run.</div>"

    seed_rows = []
    best_vals = []
    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", "?")
        best = run.get("metrics.val_realistic_pr_auc_geomean", float("nan"))
        pr200 = run.get("metrics.val_200_pr_auc", float("nan"))
        pr500 = run.get("metrics.val_500_pr_auc", float("nan"))
        pr1000 = run.get("metrics.val_1000_pr_auc", float("nan"))
        iou = run.get("metrics.val_200_iou_rts", float("nan"))
        obj_p = run.get("metrics.val_200_obj_precision", float("nan"))
        obj_r = run.get("metrics.val_200_obj_recall", float("nan"))
        seed_rows.append({
            "Run": name,
            "PR-AUC geomean (best)": f"{best:.4f}" if not np.isnan(best) else "—",
            "PR-AUC 1:200": f"{pr200:.4f}" if not np.isnan(pr200) else "—",
            "PR-AUC 1:500": f"{pr500:.4f}" if not np.isnan(pr500) else "—",
            "PR-AUC 1:1000": f"{pr1000:.4f}" if not np.isnan(pr1000) else "—",
            "IoU_RTS 1:200": f"{iou:.4f}" if not np.isnan(iou) else "—",
            "Obj Prec": f"{obj_p:.3f}" if not np.isnan(obj_p) else "—",
            "Obj Rec": f"{obj_r:.3f}" if not np.isnan(obj_r) else "—",
        })
        if not np.isnan(best):
            best_vals.append(best)

    df = pd.DataFrame(seed_rows)
    table_html = _html_table(df)

    stats_html = ""
    if len(best_vals) >= 2:
        mu0 = float(np.mean(best_vals))
        sigma0 = float(np.std(best_vals, ddof=1))
        gate_g = max(0.01, 2 * sigma0)
        if sigma0 < 0.005:
            band = "Low-noise (σ₀ < 0.005) — single seed per candidate reliable"
        elif sigma0 < 0.015:
            band = "Medium-noise (0.005 ≤ σ₀ < 0.015) — single seed OK; re-run top ties at seed 43"
        else:
            band = "High-noise (σ₀ ≥ 0.015) — investigate noise before continuing"

        stats_html = f"""
<div class='card'>
  <div style='display:flex; gap:2rem; flex-wrap:wrap;'>
    <div><span class='label'>μ₀ (mean PR-AUC geomean)</span><br><span class='metric'>{mu0:.4f}</span></div>
    <div><span class='label'>σ₀ (std across seeds)</span><br><span class='metric'>{sigma0:.4f}</span></div>
    <div><span class='label'>Gate G = max(2σ₀, 0.01)</span><br><span class='metric'>{gate_g:.4f}</span></div>
  </div>
  <p style='margin-top:0.8rem; font-size:0.9rem;'><strong>Noise band:</strong> {band}</p>
</div>"""

    return f"""
<h2>Phase 0c — Multi-Seed Baseline</h2>
<p>Seeds 42, 43, 44 with locked normalization and LRs from Phase 0a/0b.
   σ₀ calibrates Gate G for all subsequent phase comparisons.</p>
{table_html}
{stats_html}
"""


def _section_artifacts(mlflow, experiment_name: str) -> str:
    runs = _search_runs(mlflow, experiment_name, "phase0")
    if runs.empty:
        return "<div class='todo'>⏳ No Phase 0 runs found in MLflow.</div>"

    rows = []
    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", "?")
        rid = run.get("run_id", "")
        tracking_uri = run.get("artifact_uri", "")
        status = run.get("status", "")
        rows.append({
            "Run name": name,
            "Run ID": f"<span class='mono'>{rid}</span>",
            "Status": status,
            "Artifact URI": f"<span class='mono' style='font-size:0.8rem'>{tracking_uri}</span>",
            "Checkpoints": f"<span class='mono' style='font-size:0.8rem'>runs/{name}/checkpoints/</span>",
        })
    df = pd.DataFrame(rows)
    return f"""
<h2>Artifact Locations</h2>
<p>Every training run emits an artifact summary to the log. Checkpoints are saved to
<code>runs/&lt;run_name&gt;/checkpoints/</code> on the host VM and to MLflow as
<code>best_deployment.pth</code> at run end. MLflow artifacts are at the Tracking URI below.</p>
{_html_table(df)}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Phase 0 HTML report from MLflow")
    p.add_argument("--config", default="configs/baseline.yaml",
                   help="Config YAML for MLflow tracking_uri and experiment_name")
    p.add_argument("--output", default="docs/phase0_report.html",
                   help="Output HTML file path")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    cfg = load_config(args.config)
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    experiment_name = cfg["mlflow"]["experiment_name"]

    logger.info("Connecting to MLflow: %s / %s", tracking_uri, experiment_name)
    mlflow = _connect_mlflow(tracking_uri)

    s0a = _section_phase0a(mlflow, experiment_name)
    s0b = _section_phase0b(mlflow, experiment_name)
    s0c = _section_phase0c(mlflow, experiment_name)
    s_art = _section_artifacts(mlflow, experiment_name)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phase 0 Experiments — RTS Segmentation v2</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Phase 0 — Baseline Calibration Report</h1>
<p>Auto-generated from MLflow at <code>{tracking_uri}</code> &mdash;
experiment <code>{experiment_name}</code>.</p>
<p>Phase 0 objective: lock RGB normalization (0a), pick LRs (0b),
run 3-seed baseline to measure μ₀ and σ₀ (0c). σ₀ calibrates Gate G
for all subsequent experiment phases.</p>

{s0a}
{s0b}
{s0c}
{s_art}

<hr style="margin-top:3rem">
<p style="color:#94a3b8; font-size:0.8rem">
Generated by <code>scripts/report_phase0.py</code>.
Re-run after each Phase 0 sub-step to refresh results.
</p>
</body>
</html>"""

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
