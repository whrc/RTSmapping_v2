"""Generate the project's single living HTML report from MLflow.

The report (`docs/report.html`) is the project dashboard — all findings, past/current/future,
accumulate here. Rules (frequency / contents / format) are in `docs/report.md`; the experiment
program SSoT is `training/experiments.md`. Sections (each auto-populates from MLflow by run_name
prefix, or shows a pending/blocked/gated badge):
  1. Overview & status      5. Phase 3 — loss family → boundary
  2. Phase 0 — baseline      6. Phase 4 — EXTRA channels
  3. Phase 1 — temporal      7. Phase 5 — architecture (gated)
  4. Phase 2 — data scaling  8. Findings & insights   9. Open questions / future

Usage:
    python scripts/build_report.py \\
        --config configs/baseline.yaml \\
        --output docs/report.html

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
    ax1.loglog(lrs, losses, color="#2563EB", linewidth=1.8)
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
.badge  { display:inline-block; padding:2px 9px; border-radius:11px; font-size:0.72rem;
          font-weight:700; text-transform:uppercase; letter-spacing:0.03em; vertical-align:middle; }
.b-done    { background:#dcfce7; color:#166534; }
.b-running { background:#dbeafe; color:#1e40af; }
.b-pending { background:#f1f5f9; color:#64748b; }
.b-blocked { background:#fee2e2; color:#991b1b; }
.b-gated   { background:#ede9fe; color:#6d28d9; }
.insight{ background:#eff6ff; border-left:4px solid #2563EB; border-radius:0 6px 6px 0;
          padding:12px 16px; margin:1rem 0; font-size:0.92rem; }
.insight strong { color:#1d4ed8; }
.pass { color:#166534; font-weight:700; } .fail { color:#94a3b8; }
.toc a { margin-right: 1rem; font-size: 0.9rem; text-decoration: none; color:#2563EB; }
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


def _fmt(v: float, places: int = 4) -> str:
    """Format a metric value, or em-dash for NaN."""
    return f"{v:.{places}f}" if not np.isnan(v) else "—"


def _best_smoothed_from_history(mlflow, run_id: str, metric: str, window: int = 3) -> float:
    """Max of the trailing `window`-validation moving average.

    Replicates training/early_stopping.py so the reported "best" matches the
    smoothed value the early-stopper used for best-checkpoint selection
    (window = training.early_stopping.smoothing_window, 3 in all phase0 configs).
    """
    from collections import deque
    vals = [v for _, v in sorted(_get_metric_history(mlflow, run_id, metric))]
    if not vals:
        return float("nan")
    win: deque = deque(maxlen=window)
    best = float("-inf")
    for v in vals:
        win.append(v)
        best = max(best, sum(win) / len(win))
    return best


def _final_from_history(mlflow, run_id: str, metric: str) -> float:
    """Last logged value of a metric (NaN if none)."""
    hist = _get_metric_history(mlflow, run_id, metric)
    return hist[-1][1] if hist else float("nan")


def _section_phase0c(mlflow, experiment_name: str) -> str:
    runs = _search_runs(mlflow, experiment_name, "phase0c_seed")
    if runs.empty:
        return "<div class='todo'>⏳ Phase 0c multi-seed baseline not yet run.</div>"

    # Defensive: a relaunched seed creates a second run with the same name.
    # Keep only the most-recent run per seed so μ₀/σ₀ aren't double-counted.
    if {"tags.mlflow.runName", "start_time"}.issubset(runs.columns):
        runs = (runs.sort_values("start_time")
                    .drop_duplicates("tags.mlflow.runName", keep="last"))

    # Gate metric is the geomean over the honestly-supported ratios [5,10,20]
    # (metrics.pr_auc_ratios; see docs/baseline_unetpp_effb5.md). pixel_iou and
    # obj_f1 are logged as monotonic stability anchors.
    seed_rows = []
    best_vals = []
    curve_blocks = []
    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", "?")
        rid = run.get("run_id", "")
        # Best-per-seed = MAX over the epoch history (project defines μ₀/σ₀ on
        # the best-per-seed gate value, not the last-logged one).
        best = _best_smoothed_from_history(mlflow, rid, "val_realistic_pr_auc_geomean")
        seed_rows.append({
            "Run": name,
            "PR-AUC geomean (best)": f"{best:.4f}" if not np.isnan(best) else "—",
            "PR-AUC 1:5 (final)": _fmt(_final_from_history(mlflow, rid, "pr_auc_ratio_5")),
            "PR-AUC 1:10 (final)": _fmt(_final_from_history(mlflow, rid, "pr_auc_ratio_10")),
            "PR-AUC 1:20 (final)": _fmt(_final_from_history(mlflow, rid, "pr_auc_ratio_20")),
            "pixel_IoU (final)": _fmt(_final_from_history(mlflow, rid, "pixel_iou")),
            "obj_F1 (final)": _fmt(_final_from_history(mlflow, rid, "object_f1")),
        })
        # μ₀/σ₀ are calibrated on COMPLETED seeds only — a still-running seed's
        # best-so-far would otherwise contaminate the gate.
        if not np.isnan(best) and run.get("status", "") == "FINISHED":
            best_vals.append(best)

        # Per-seed curve panels: (1) train vs val loss overlay — overfitting
        # detector; (2) gate metric + IoU/F1 quality anchors.
        loss_hist = {
            "train_loss": _get_metric_history(mlflow, rid, "train_loss"),
            "val_loss": _get_metric_history(mlflow, rid, "val_loss"),
        }
        qual_hist = {
            "PR-AUC geomean (gate)": _get_metric_history(mlflow, rid, "val_realistic_pr_auc_geomean"),
            "pixel_IoU": _get_metric_history(mlflow, rid, "pixel_iou"),
            "obj_F1": _get_metric_history(mlflow, rid, "object_f1"),
        }
        imgs = []
        if any(loss_hist.values()):
            b64 = _plot_metric_curves(loss_hist, title=f"{name} — train vs val loss",
                                      ylabel="loss", colors=["#2563EB", "#DC2626"])
            imgs.append(f"<img src='data:image/png;base64,{b64}' alt='{name} loss'>")
        if any(qual_hist.values()):
            b64 = _plot_metric_curves(qual_hist, title=f"{name} — gate metric + quality anchors",
                                      ylabel="score", colors=["#9333EA", "#16A34A", "#EA580C"])
            imgs.append(f"<img src='data:image/png;base64,{b64}' alt='{name} quality'>")
        if imgs:
            curve_blocks.append(
                f"<div style='display:flex; gap:1rem; flex-wrap:wrap; margin:0.5rem 0;'>"
                f"{''.join(imgs)}</div>")

    df = pd.DataFrame(seed_rows)
    table_html = _html_table(df)

    stats_html = ""
    if len(best_vals) >= 2:
        n_done = len(best_vals)
        prelim = ("<p style='color:#B45309; font-weight:600;'>⚠ Preliminary — "
                  f"only {n_done}/3 seeds finished; σ₀ from &lt;3 seeds is unreliable. "
                  "Final gate requires all 3.</p>") if n_done < 3 else ""
        mu0 = float(np.mean(best_vals))
        sigma0 = float(np.std(best_vals, ddof=1))
        # experiments.md §1.4: a candidate wins iff Δ(PR-AUC geomean) vs baseline μ₀ ≥ G
        # AND precision@recall=0.5 does not regress. G is a Δ-threshold, NOT a perf floor.
        gate_g = max(0.01, 2 * sigma0)
        if sigma0 < 0.005:
            band = "Low-noise (σ₀ < 0.005) — single seed per candidate reliable"
        elif sigma0 < 0.015:
            band = "Medium-noise (0.005 ≤ σ₀ < 0.015) — single-seed first-pass; re-run top ties at seed 43"
        else:
            band = "High-noise (σ₀ ≥ 0.015) — investigate noise before continuing"

        stats_html = f"""
<div class='card'>
  {prelim}
  <div style='display:flex; gap:2rem; flex-wrap:wrap;'>
    <div><span class='label'>μ₀ (mean best PR-AUC geomean — baseline ref)</span><br><span class='metric'>{mu0:.4f}</span></div>
    <div><span class='label'>σ₀ (std across seeds)</span><br><span class='metric'>{sigma0:.4f}</span></div>
    <div><span class='label'>Gate G = max(2σ₀, 0.01)</span><br><span class='metric'>{gate_g:.4f}</span></div>
  </div>
  <p style='margin-top:0.8rem; font-size:0.9rem;'><strong>Noise band:</strong> {band}</p>
  <p style='margin-top:0.4rem; font-size:0.85rem; color:#555;'>Gate metric = geomean(PR-AUC @ 1:5/1:10/1:20).
  Per <code>experiments.md §1.4</code>, a candidate <strong>wins</strong> only if Δ(PR-AUC geomean) vs
  baseline μ₀={mu0:.4f} is ≥ <strong>G = {gate_g:.4f}</strong> <em>and</em> precision @ recall=0.5 does not regress.</p>
</div>"""

    return f"""
<h2>Phase 0c — Multi-Seed Baseline</h2>
<p>Seeds 42, 43, 44 on the frozen dataset snapshot with locked normalization and LRs from Phase 0a/0b.
   μ₀ is the baseline; σ₀ sets the winner gate <strong>G = max(2σ₀, 0.01)</strong> (experiments.md §1.4).</p>
{table_html}
{stats_html}
<h3>Per-seed training curves</h3>
<p>Left: train vs val loss (overfitting detector). Right: gate metric with pixel_IoU / obj_F1 anchors.</p>
{"".join(curve_blocks) if curve_blocks else "<p>(no curve history available)</p>"}
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

# ---------------------------------------------------------------------------
# Project-wide sections (experiments.md program: Phase 0 → 5 → Final)
# ---------------------------------------------------------------------------

# ⚠ v2.0 constants (frozen 2026-06-07) — recompute after the v2.1 re-baseline
# (docs/v21_staleness_audit.md #3). _section_phase0c computes the live values
# from MLflow; these mirror them for the cross-section gate comparisons.
MU0, SIGMA0 = 0.5683, 0.0125
GATE_G = max(0.01, 2 * SIGMA0)        # 0.025 — experiments.md §1.4
GATE_RATIOS = "[5, 10, 20]"
PCT_TO_NPOS = {25: 475, 50: 950, 75: 1425, 100: 1900}


def _dedup_latest(runs):
    """Keep the most-recent run per run_name (relaunches create duplicates)."""
    if not runs.empty and {"tags.mlflow.runName", "start_time"}.issubset(runs.columns):
        return runs.sort_values("start_time").drop_duplicates("tags.mlflow.runName", keep="last")
    return runs


def _badge(status: str) -> str:
    cls = {"done": "b-done", "running": "b-running", "pending": "b-pending",
           "blocked": "b-blocked", "gated": "b-gated"}.get(status, "b-pending")
    return f"<span class='badge {cls}'>{status}</span>"


def _plot_data_scaling(points) -> str:
    """points = [(n_pos, gate)]; gate vs log10(n_pos)."""
    pts = sorted(points)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="#2563EB", linewidth=2, markersize=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("train positives (log scale)")
    ax.set_ylabel("best gate (PR-AUC geomean)")
    ax.set_title("Phase 2 — data-scaling curve")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_gap_bars(labels, train_ious, val_ious) -> str:
    """Grouped bars: train vs val pixel-IoU per subset (generalization gap)."""
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, train_ious, w, label="train IoU", color="#2563EB")
    ax.bar(x + w / 2, val_ious, w, label="val IoU", color="#DC2626")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("pixel IoU (final epoch)")
    ax.set_title("Phase 2 §5.4 — train vs val IoU (gap = overfitting)")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _section_overview(tracking_uri: str) -> str:
    return f"""
<h2 id='overview'>1. Overview &amp; status</h2>
<p>Semantic segmentation of <strong>Retrogressive Thaw Slumps (RTS)</strong> in pan-arctic PlanetScope
imagery (60–74°N). UNet++ / EfficientNet-B5; balanced sampling + curriculum + focal loss. This is the
project's single living dashboard — auto-generated from MLflow (<code>{tracking_uri}</code>).</p>
<div class='card'>
  <div style='display:flex; gap:2rem; flex-wrap:wrap;'>
    <div><span class='label'>Dataset</span><br><span class='metric'>v0.2</span> <span style='font-size:0.8rem'>15,528 tiles · 1,819 pos</span></div>
    <div><span class='label'>Baseline μ₀</span><br><span class='metric'>{MU0:.4f}</span></div>
    <div><span class='label'>Gate G = max(2σ₀,0.01)</span><br><span class='metric'>{GATE_G:.3f}</span></div>
    <div><span class='label'>Current phase</span><br><span class='metric'>2 → 3</span></div>
  </div>
  <p style='margin-top:0.6rem; font-size:0.85rem; color:#555'>A candidate <strong>wins</strong> iff
  Δ(PR-AUC geomean) vs μ₀ ≥ G <em>and</em> precision@recall0.5 does not regress (experiments.md §1.4).
  Gate metric = geomean(PR-AUC @ {GATE_RATIOS}); 1:200/1000 deferred to Test-Realistic.</p>
</div>
<p class='toc'><strong>Jump:</strong>
<a href='#p0a'>Phase 0</a><a href='#p1'>Phase 1</a><a href='#p2'>Phase 2</a><a href='#p3'>Phase 3</a>
<a href='#p4'>Phase 4</a><a href='#p5'>Phase 5</a><a href='#findings'>Findings</a><a href='#future'>Future</a></p>
"""


def _section_phase1() -> str:
    return f"""
<h2 id='p1'>3. Phase 1 — Temporal sanity (2025) {_badge('blocked')}</h2>
<p>Detect 2024→2025 domain drift on a 2025 micro-set (experiments.md §4) — <strong>blocked</strong> on the
micro-set definition (user + Heidi Rodenhizer). ~1 GPU-hour once it exists.</p>
"""


def _section_phase2(mlflow, experiment_name: str) -> str:
    runs = _dedup_latest(_search_runs(mlflow, experiment_name, "phase2_scale_"))
    if runs.empty:
        return f"<h2 id='p2'>4. Phase 2 — Data scaling {_badge('pending')}</h2><p>experiments.md §5.</p>"
    rows, curve, glab, gtr, gva = [], [], [], [], []
    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", ""); rid = run.get("run_id", "")
        try:
            pct = int(name.split("_")[-1])
        except ValueError:
            continue
        gate = _best_smoothed_from_history(mlflow, rid, "val_realistic_pr_auc_geomean")
        tiou = _final_from_history(mlflow, rid, "train_iou")
        viou = _final_from_history(mlflow, rid, "pixel_iou")
        n = PCT_TO_NPOS.get(pct, pct)
        curve.append((n, gate))
        gap = (tiou - viou) if not (np.isnan(tiou) or np.isnan(viou)) else float("nan")
        rows.append({"Subset": f"{pct}%", "≈ pos": n, "Best gate": _fmt(gate),
                     "train IoU": _fmt(tiou, 3), "val IoU": _fmt(viou, 3), "gap": _fmt(gap, 3)})
        if not (np.isnan(tiou) or np.isnan(viou)):  # skip subsets without both IoUs (a 0.0 bar fakes a negative gap)
            glab.append(f"{pct}%"); gtr.append(tiou); gva.append(viou)
    base = _dedup_latest(_search_runs(mlflow, experiment_name, "phase0c_seed42"))
    if not base.empty:
        g100 = _best_smoothed_from_history(mlflow, base.iloc[-1]["run_id"], "val_realistic_pr_auc_geomean")
        curve.append((1900, g100))
        rows.append({"Subset": "100%", "≈ pos": 1900, "Best gate": _fmt(g100),
                     "train IoU": "—", "val IoU": "—", "gap": "—"})
    rows = sorted(rows, key=lambda r: r["≈ pos"])
    cmap = {n: g for n, g in curve}
    ratio_txt, regime = "—", "—"
    try:
        s_lo = (cmap[950] - cmap[475]) / (np.log(950) - np.log(475))
        s_hi = (cmap[1900] - cmap[1425]) / (np.log(1900) - np.log(1425))
        ratio = s_hi / s_lo if s_lo != 0 else float("inf")
        if np.isnan(ratio):  # a run mid-training yields NaN gates; NaN compares
            raise KeyError   # False everywhere and would misreport "Plateau"
        ratio_txt = f"{ratio:.1f}"
        regime = ("Severely under-scaled" if ratio > 1.0 else
                  "Diminishing but still scaling" if ratio >= 0.5 else "Plateau before 100%")
    except KeyError:
        pass
    imgs = ""
    if len(curve) >= 2:
        imgs += f"<img src='data:image/png;base64,{_plot_data_scaling(curve)}'>"
    if glab:
        imgs += f"<img src='data:image/png;base64,{_plot_gap_bars(glab, gtr, gva)}'>"
    badge = _badge("done") if len(rows) >= 4 else _badge("running")
    return f"""
<h2 id='p2'>4. Phase 2 — Data scaling {badge}</h2>
<p>Does more labeled data help, and is the model using its capacity? (experiments.md §5)</p>
<div style='display:flex; gap:1rem; flex-wrap:wrap;'>{imgs}</div>
{_html_table(pd.DataFrame(rows))}
<div class='insight'><strong>§5.3 slope</strong> (75→100)/(25→50) ≈ <strong>{ratio_txt}</strong> →
<strong>{regime}</strong>.<br>
<strong>§5.4 gap</strong>: see table (gap &gt; 0.4 ⇒ severe over-parameterization ⇒ §6.3 weight-decay sweep
triggered; bigger architectures would overfit more → Phase 5 lean SKIP). Real levers when data-limited +
over-parameterized: <strong>more data + regularization</strong>, not capacity.<br>
<em>Prose interpretation snapshot 2026-06-07 (v2.0 runs): ratio ≈ 4.4, gap ≈ 0.43 — see
docs/phase2_data_scaling.md.</em></div>
"""


def _section_phase3(mlflow, experiment_name: str) -> str:
    runs = _dedup_latest(_search_runs(mlflow, experiment_name, "phase3_loss_"))
    abl = _dedup_latest(_search_runs(mlflow, experiment_name, "abl_loss_"))
    allruns = pd.concat([runs, abl]) if not abl.empty else runs
    # §1.4 defines the win as Δ vs baseline μ₀ (multi-seed mean), not the
    # single seed-42 number — that reference was off by +0.0076 (30% of G).
    rows = [{"Candidate": "focal (baseline μ₀)", "Best gate": _fmt(MU0),
             "Δ vs baseline": "ref", "Win (≥G)?": "—"}]
    for _, run in allruns.iterrows():
        name = run.get("tags.mlflow.runName", ""); rid = run.get("run_id", "")
        gate = _best_smoothed_from_history(mlflow, rid, "val_realistic_pr_auc_geomean")
        if np.isnan(gate):
            continue
        d = gate - MU0
        passed = d >= GATE_G
        rows.append({"Candidate": name.replace("phase3_loss_", "").replace("abl_loss_", ""),
                     "Best gate": _fmt(gate), "Δ vs baseline": f"{d:+.4f}",
                     "Win (≥G)?": f"<span class='{'pass' if passed else 'fail'}'>{'PASS' if passed else 'no'}</span>"})
    return f"""
<h2 id='p3'>5. Phase 3 — Loss family → boundary {_badge('running')}</h2>
<p>Sequential elimination (experiments.md §6): pick loss, lock, then boundary. Win = Δ ≥ G={GATE_G:.3f} + no precision drop.</p>
{_html_table(pd.DataFrame(rows))}
<div class='insight'><em>Prose interpretation snapshot 2026-06-07 (v2.0 runs):</em> <strong>focal-only wins
so far.</strong> Tversky (precision-focused) collapses the imbalanced gate; compound (Focal+Dice) ties.
Per §1.4 tie-break the simpler <strong>focal</strong> holds unless a candidate clears G. The table above is
live from MLflow — trust it over this prose if they disagree. Next: boundary handling (§6.2) + the §6.3
weight-decay sweep (triggered by the §5.4 gap above).</div>
"""


def _section_phase45() -> str:
    return f"""
<h2 id='p4'>6. Phase 4 — EXTRA channels {_badge('pending')}</h2>
<p>EXTRA group ablation — NDVI / NBR / SE-PCA / SE-Proto / TC (experiments.md §7). Blocked: EXTRA tiles not yet generated.</p>
<h2 id='p5'>7. Phase 5 — Architecture {_badge('gated')}</h2>
<p>Gated (experiments.md §8.1): runs only with data headroom AND an unsaturated model. Evidence — data still
scaling (good) but the <strong>gap ≈ 0.43 is large</strong> → over-parameterized; bigger backbones
(B7 / SegFormer) would overfit more → <strong>lean SKIP</strong>. SegFormer support is implemented and
<code>configs/phase5_*</code> are ready if the gate flips after Phase 3/4.</p>
<div class='todo'>⚠ Spec note: §8.1 cond-2 ("gap not closed &lt; 0.3 → run Phase 5") reads backwards vs §5.4
("large gap → over-parameterized → regularize"). Interpreted here as <strong>large gap ⇒ skip Phase 5</strong>.</div>
"""


def _section_findings() -> str:
    return """
<h2 id='findings'>8. Findings &amp; insights</h2>
<ul>
<li><strong>Data is the bottleneck.</strong> Data-scaling still rising at 100% + a large train/val gap →
more labeled positives + regularization are the highest-leverage levers, not bigger models.</li>
<li><strong>Severe overfitting</strong> — train IoU ~0.68 vs val ~0.24. The §6.3 weight-decay sweep is now warranted.</li>
<li><strong>Focal-only loss wins so far</strong>; precision-focused Tversky hurts the imbalanced gate.</li>
<li><strong>Gate is data-limited</strong> — measured at 1:5–1:20 (honest), not deployment 1:1000, so these
numbers are optimistic vs deployment precision (still unmeasured).</li>
<li><strong>Feasibility</strong> — pan-arctic mapping is realistic as a <strong>QC-assisted candidate map</strong>
(high recall + filtering), not yet a fully-automated high-precision product.</li>
</ul>
"""


def _section_future() -> str:
    return """
<h2 id='future'>9. Open questions, future work &amp; decisions</h2>
<ul>
<li><strong>Spatial generalization</strong> — leave-one-ecoregion-out CV (make-or-break for pan-arctic).</li>
<li><strong>Honest deployment precision</strong> — false-positives per true-positive on a held-out region at realistic prevalence.</li>
<li><strong>More labels</strong> — Phase 2 says they help; path toward ~3500 positives.</li>
<li><strong>v0.3 dataset</strong> (selection upgrade, no relabel) → re-baseline + re-gate.</li>
<li><strong>Multi-GPU VM</strong> → parallel orchestrator + concurrency-safe MLflow; run the full program fast.</li>
<li><strong>Multi-year consistency</strong> (2024∧2025) as a precision lever at deployment.</li>
</ul>
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the project living HTML report from MLflow")
    p.add_argument("--config", default="configs/baseline.yaml",
                   help="Config YAML for MLflow experiment_name")
    p.add_argument("--tracking-uri", default=None,
                   help="Override mlflow.tracking_uri from config (e.g. /mnt/outputs/mlflow)")
    p.add_argument("--output", default="docs/report.html",
                   help="Output HTML file path")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    cfg = load_config(args.config)
    tracking_uri = args.tracking_uri or cfg["mlflow"]["tracking_uri"]
    experiment_name = cfg["mlflow"]["experiment_name"]

    logger.info("Connecting to MLflow: %s / %s", tracking_uri, experiment_name)
    mlflow = _connect_mlflow(tracking_uri)

    import datetime
    overview = _section_overview(tracking_uri)
    s0a = _section_phase0a(mlflow, experiment_name)
    s0b = _section_phase0b(mlflow, experiment_name)
    s0c = _section_phase0c(mlflow, experiment_name)
    s1 = _section_phase1()
    s2 = _section_phase2(mlflow, experiment_name)
    s3 = _section_phase3(mlflow, experiment_name)
    s45 = _section_phase45()
    findings = _section_findings()
    future = _section_future()
    s_art = _section_artifacts(mlflow, experiment_name)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RTS Segmentation v2 — Project Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>RTS Segmentation v2 — Project Report</h1>
<p style="color:#64748b">Living dashboard for the pan-arctic RTS mapping project. Auto-generated
{now} from MLflow <code>{tracking_uri}</code> (experiment <code>{experiment_name}</code>). Rules:
<code>docs/report.md</code> · program SSoT: <code>training/experiments.md</code>.</p>

{overview}
<h2 id='p0a'>2. Phase 0 — Baseline calibration <span class='badge b-done'>done</span></h2>
{s0a}
{s0b}
{s0c}
{s1}
{s2}
{s3}
{s45}
{findings}
{future}

<hr style="margin-top:3rem">
{s_art}
<p style="color:#94a3b8; font-size:0.8rem">
Generated by <code>scripts/build_report.py</code>. Regenerate after each run / phase per
<code>docs/report.md</code>.
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
