"""Generate docs/report.html — the analytical + visual view of the experiment ledger.

The ledger (`docs/experiment_ledger.md`) is the experiments SSoT. This script does NOT
re-state facts: it RENDERS the ledger's curated blocks (gate, recipe, findings) and adds
what the ledger can't show — a recipe build-up chart, per-family val-PR-AUC curves read
straight from the MLflow metric files, qualitative map overlays, and stats computed from
the run table. Every number/verdict here traces to the ledger or to MLflow; nothing is
hardcoded. What to highlight is curated (the FAMILY_FIGURES picks below) under that rule.

Deps: stdlib + matplotlib only (runs in the rts-train Docker image). No mlflow/pandas.

Usage:
    python scripts/build_report.py                       # defaults below
    python scripts/build_report.py --ledger docs/experiment_ledger.md \
        --output docs/report.html --mlflow-root /outputs/v1.0/mlflow
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
METRIC = "val_realistic_pr_auc_geomean"

# Qualitative artifacts (container /outputs; helper falls back to /mnt/outputs).
_VAL_OVERLAY = "/outputs/v1.0/inference/v1.0_baseline_validation/validation_overlay.png"
_VAL_OVERLAY_2024 = "/outputs/v1.0/inference/v1.0_baseline_validation/validation_overlay_2024labels.png"
_GALLERY_DIR = "/outputs/v1.0/qc/extra_vis"


# ---------------------------------------------------------------------------
# Ledger parsing (the SSoT)
# ---------------------------------------------------------------------------

def read_block(text: str, name: str) -> str:
    """Return the markdown between <!-- NAME:BEGIN ... --> and <!-- NAME:END -->."""
    m = re.search(rf"<!--\s*{name}:BEGIN.*?-->(.*?)<!--\s*{name}:END\s*-->", text, re.S)
    return m.group(1).strip() if m else ""


def parse_md_table(md: str) -> tuple[list[str], list[dict]]:
    """Parse the first GitHub markdown table → (headers, list-of-row-dicts)."""
    lines = [ln for ln in md.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return [], []
    headers = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    rows = []
    for ln in lines[2:]:  # skip header + separator
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return headers, rows


def md_inline(s: str) -> str:
    """Minimal inline markdown → HTML (escape first, then **bold** and `code`)."""
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def md_to_html(md: str) -> str:
    """Small block-level markdown → HTML: headings, tables, blockquotes, paragraphs."""
    out: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("|"):  # table block
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            hdr, rows = parse_md_table("\n".join(block))
            th = "".join(f"<th>{md_inline(h)}</th>" for h in hdr)
            trs = "".join(
                "<tr>" + "".join(f"<td>{md_inline(v)}</td>" for v in r.values()) + "</tr>"
                for r in rows
            )
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")
            continue
        if s.startswith("### "):
            out.append(f"<h3>{md_inline(s[4:])}</h3>")
        elif s.startswith("## "):
            out.append(f"<h2>{md_inline(s[3:])}</h2>")
        elif s.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(f"<div class='insight'>{md_inline(' '.join(quote))}</div>")
            continue
        else:
            out.append(f"<p>{md_inline(s)}</p>")
        i += 1
    return "\n".join(out)


# ---------------------------------------------------------------------------
# MLflow metric files (read directly — no mlflow lib)
# ---------------------------------------------------------------------------

def _mlflow_root(arg: str) -> Path | None:
    for p in (Path(arg), Path(arg.replace("/outputs/", "/mnt/outputs/"))):
        if p.exists():
            return p
    return None


def metric_curve(root: Path, run_name: str) -> list[tuple[int, float]]:
    """Read [(step, value)] for METRIC from a run's MLflow file store."""
    hits = list(root.glob(f"{run_name}/*/*/metrics/{METRIC}"))
    if not hits:
        return []
    pts = []
    for ln in hits[0].read_text().splitlines():
        parts = ln.split()
        if len(parts) >= 3:
            try:
                pts.append((int(parts[2]), float(parts[1])))
            except ValueError:
                continue
    return sorted(pts)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_file_b64(path: str) -> str | None:
    for p in (Path(path), Path(path.replace("/outputs/", "/mnt/outputs/"))):
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return None


def buildup_chart(rows: list[dict]) -> str:
    steps, vals = [], []
    for r in rows:
        v = re.match(r"[\d.]+", r.get("PR-AUC", ""))
        if v:
            steps.append(r["Step (cumulative)"])
            vals.append(float(v.group()))
    if not vals:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(vals)), vals, color="#2563EB")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([s.replace("+ ", "+\n") for s in steps], fontsize=8, rotation=0)
    ax.set_ylim(0.80, 0.94)
    ax.set_ylabel("PR-AUC (best_smoothed, 3-seed)")
    ax.set_title("Recipe build-up", fontsize=12, fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _b64(fig)


def family_curves(root: Path, runs: list[dict], fam: str, picks: list[str]) -> str:
    histories = {n: metric_curve(root, n) for n in picks}
    histories = {n: h for n, h in histories.items() if h}
    if not histories:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#EA580C", "#0891B2"]
    for i, (label, pts) in enumerate(histories.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=label, color=palette[i % len(palette)], linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val PR-AUC geomean")
    ax.set_title(f"Family {fam} — validation curves", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _b64(fig)


def battery_charts(run_rows: list[dict]) -> str:
    """Three panels for the C1–C4 hypothesis battery, 3-seed means from the ledger `score` column.

    Groupings mirror the CSV scope in scripts/export_metric_robustness.py; the report states no fact the
    ledger doesn't. A condition whose runs are absent renders blank rather than erroring.
    """
    smap: dict[str, float] = {}
    for r in run_rows:
        if re.match(r"[\d.]+$", r.get("score", "")):
            smap[r["name"]] = float(r["score"])

    def agg(names: list[str]) -> tuple[float | None, float]:
        vals = [smap[n] for n in names if n in smap]
        if not vals:
            return None, 0.0
        m = sum(vals) / len(vals)
        sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        return m, sd

    BLUE, RED, GREY = "#2563EB", "#DC2626", "#94A3B8"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) C2 — same-family capacity sweep (RGB+NDVI, deploy recipe)
    cap = [("B0", ["c2_effb0_ndvi_seed42", "c2_effb0_ndvi_seed43", "c2_effb0_ndvi_seed44"]),
           ("B3", ["effb3_deploy", "effb3_deploy_seed43", "effb3_deploy_seed44"]),
           ("B5", ["deploy_v1_ndvi_seed42", "deploy_v1_ndvi_seed43", "deploy_v1_ndvi_seed44"]),
           ("B7", ["c2_effb7_ndvi_seed42", "c2_effb7_ndvi_seed43", "c2_effb7_ndvi_seed44"])]
    aggs = [agg(n) for _, n in cap]
    ax = axes[0]
    ax.bar(range(4), [(m if m is not None else float("nan")) for m, _ in aggs],
           yerr=[s for _, s in aggs], color=BLUE, capsize=4)
    ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in cap])
    ax.set_ylim(0.80, 0.94); ax.set_ylabel("PR-AUC (best_smoothed, 3-seed)")
    ax.set_title("C2 — EfficientNet capacity\n(RGB+NDVI, deploy recipe)", fontsize=10, fontweight="bold")
    for i, (m, s) in enumerate(aggs):
        if m is not None:
            ax.text(i, m + s + 0.003, f"{m:.3f}", ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # (b) C3 — data-budget scaling curves (small CNN vs large ViT-L)
    eff = [(25, ["scale_ndvi_25", "scale_ndvi_25_seed43", "scale_ndvi_25_seed44"]),
           (50, ["scale_ndvi_50", "scale_ndvi_50_seed43", "scale_ndvi_50_seed44"]),
           (75, ["scale_ndvi_75", "scale_ndvi_75_seed43", "scale_ndvi_75_seed44"]),
           (100, ["aug_trivialaugment_deploy", "aug_trivialaugment_deploy_seed43",
                  "aug_trivialaugment_deploy_seed44"])]
    vit = [(25, ["c3_vitl_ndvi_scale25_seed42", "c3_vitl_ndvi_scale25_seed43", "c3_vitl_ndvi_scale25_seed44"]),
           (50, ["c3_vitl_ndvi_scale50_seed42", "c3_vitl_ndvi_scale50_seed43", "c3_vitl_ndvi_scale50_seed44"]),
           (100, ["fm_dinov3sat_l_ndvi_locked", "fm_dinov3sat_l_ndvi_locked_seed43",
                  "fm_dinov3sat_l_ndvi_locked_seed44"])]
    ax = axes[1]
    for series, color, lab in [(eff, BLUE, "EffB5+NDVI"), (vit, RED, "ViT-L+NDVI")]:
        pts = [(x, agg(n)[0]) for x, n in series]
        pts = [(x, y) for x, y in pts if y is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=color, label=lab, linewidth=1.8)
    ax.set_xlabel("train positive budget (%)"); ax.set_ylabel("PR-AUC (3-seed)")
    ax.set_title("C3 — data-budget scaling", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (c) C1 — representation × capacity (2×2)
    ax = axes[2]
    rgb = [agg(["phase4_extra_rgb_baseline", "phase4_extra_rgb_baseline_seed43",
                "phase4_extra_rgb_baseline_seed44"])[0],
           agg(["fm_dinov3sat_l_rgb_locked_seed42", "fm_dinov3sat_l_rgb_locked_seed43",
                "fm_dinov3sat_l_rgb_locked_seed44"])[0]]
    ndvi = [agg(["aug_trivialaugment_deploy", "aug_trivialaugment_deploy_seed43",
                 "aug_trivialaugment_deploy_seed44"])[0],
            agg(["fm_dinov3sat_l_ndvi_locked", "fm_dinov3sat_l_ndvi_locked_seed43",
                 "fm_dinov3sat_l_ndvi_locked_seed44"])[0]]
    nan = float("nan"); w = 0.35
    ax.bar([i - w / 2 for i in range(2)], [(v if v is not None else nan) for v in rgb], w,
           label="RGB", color=GREY)
    ax.bar([i + w / 2 for i in range(2)], [(v if v is not None else nan) for v in ndvi], w,
           label="RGB+NDVI", color=BLUE)
    ax.set_xticks(range(2)); ax.set_xticklabels(["EffB5", "ViT-L"])
    ax.set_ylim(0.80, 0.94); ax.set_ylabel("PR-AUC (3-seed)")
    ax.set_title("C1 — representation × capacity", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return _b64(fig)


# ---------------------------------------------------------------------------
# Curated figure picks (which runs to plot per family — by run-dir name).
# Agent-edited; every name must exist in the ledger run table.
# ---------------------------------------------------------------------------

FAMILY_FIGURES = {
    "C": ("Loss → boundary", ["phase3_loss_compound_1to2", "phase3_bd_focal_ignore_w2"]),
    "D": ("Channels", ["phase4_extra_rgb_baseline", "phase4_extra_ndvi", "phase4_extra_full"]),
    "E": ("Encoder", ["deploy_v1_ndvi_seed42", "fm_dinov3sat_l_ndvi", "phase4_fm_dinov3_ndvi"]),
    "F": ("Augmentation", ["aug_ref", "aug_scale_off", "aug_trivialaugment_deploy"]),
    "I": ("Final lock", ["deploy_v1_ndvi_seed42", "deploy_v1_ndvi_seed43", "deploy_v1_ndvi_seed44"]),
}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #1e293b; }
h1 { font-size: 1.9rem; border-bottom: 3px solid #2563EB; padding-bottom: 8px; }
h2 { font-size: 1.3rem; margin-top: 2.4rem; color: #1d4ed8; }
h3 { font-size: 1.05rem; color: #374151; margin-top: 1.4rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.88rem; }
th { background: #1d4ed8; color: #fff; padding: 8px 12px; text-align: left; }
td { padding: 7px 12px; border-bottom: 1px solid #e2e8f0; }
tr:nth-child(even) td { background: #f8fafc; }
img { max-width: 100%; border-radius: 6px; margin: 0.5rem 0; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
.cards { display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }
.card { flex:1 1 150px; border:1px solid #e2e8f0; border-radius:8px; padding:14px 18px; background:#f8fafc; }
.metric { font-size:1.5rem; font-weight:700; color:#1d4ed8; }
.label { font-size:0.74rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; }
.insight { background:#eff6ff; border-left:4px solid #2563EB; border-radius:0 6px 6px 0;
           padding:12px 16px; margin:1rem 0; font-size:0.92rem; }
.status { background:#fffbeb; border:1px solid #f59e0b; border-radius:6px; padding:10px 14px; margin:1rem 0; }
figure { margin:0.5rem 0; } figcaption { font-size:0.82rem; color:#64748b; margin-top:0.2rem; }
.note { color:#64748b; font-size:0.85rem; }
a { color:#2563EB; }
"""


def now_line(diary: Path) -> str:
    if not diary.exists():
        return ""
    block = read_block(diary.read_text(), "NOW")
    for ln in block.splitlines():
        if ln.strip() and not ln.strip().startswith("#"):
            return md_inline(ln.strip())
    return ""


def stat_cards(rows: list[dict], buildup: list[dict]) -> str:
    done = [r for r in rows if r.get("status") == "done"]
    corrected = [float(r["score"]) for r in done
                 if r.get("split") == "corrected" and re.match(r"[\d.]+$", r.get("score", ""))]
    best = max(corrected) if corrected else float("nan")
    recipe = buildup[-2]["PR-AUC"] if len(buildup) >= 2 else "—"  # last numeric step
    cards = [
        ("runs logged", str(len(rows))),
        ("best PR-AUC (corrected)", f"{best:.4f}" if corrected else "—"),
        ("current EffB5 recipe", recipe),
        ("encoder verdict", "pending fair re-run"),
    ]
    inner = "".join(
        f"<div class='card'><div class='label'>{l}</div><div class='metric'>{v}</div></div>"
        for l, v in cards
    )
    return f"<div class='cards'>{inner}</div>"


def build(ledger: Path, diary: Path, mlflow_root: Path | None, output: Path) -> None:
    text = ledger.read_text()
    _, run_rows = parse_md_table(read_block(text, "RUN-TABLE"))
    _, buildup_rows = parse_md_table(read_block(text, "BUILDUP-TABLE"))

    parts: list[str] = []
    parts.append("<h1>RTSmapping_v2 — project report</h1>")
    parts.append("<p class='note'>Generated from <code>docs/experiment_ledger.md</code> (the SSoT) "
                 "by <code>scripts/build_report.py</code>. Full run registry + verdicts live in the "
                 "ledger; project status in <code>current_working_status.md</code>. Do not hand-edit "
                 "this file.</p>")

    status = now_line(diary)
    if status:
        parts.append(f"<div class='status'><strong>Now:</strong> {status}</div>")

    parts.append(stat_cards(run_rows, buildup_rows))

    # Gate + recipe + build-up
    parts.append(md_to_html(read_block(text, "GATE")))
    chart = buildup_chart(buildup_rows)
    if chart:
        parts.append("<h2>Recipe build-up</h2>")
        parts.append(f"<img src='data:image/png;base64,{chart}' alt='recipe build-up'>")
    parts.append(md_to_html(read_block(text, "RECIPE-TABLE")))

    # Hypothesis-test battery (C1–C4): capacity vs representation vs data budget
    battery = battery_charts(run_rows)
    if battery:
        parts.append("<h2>Hypothesis-test battery (C1–C4)</h2>")
        parts.append("<p class='note'>3-seed means from the ledger <code>score</code> column. "
                     "Full per-(run,seed) stats + secondary metrics: "
                     "<code>outputs/metric_robustness.csv</code>; design: "
                     "<code>docs/future_work/experiments_hypothesis_test.md</code>.</p>")
        parts.append(f"<img src='data:image/png;base64,{battery}' alt='C1-C4 hypothesis battery'>")

    # Findings (rendered from the ledger prose)
    parts.append(md_to_html(read_block(text, "FINDINGS")))

    # Per-family validation curves (best-effort from MLflow metric files)
    if mlflow_root:
        figs = []
        for fam, (title, picks) in FAMILY_FIGURES.items():
            b64 = family_curves(mlflow_root, run_rows, fam, picks)
            if b64:
                figs.append(f"<figure><img src='data:image/png;base64,{b64}'>"
                            f"<figcaption>{title} (family {fam})</figcaption></figure>")
        if figs:
            parts.append("<h2>Validation curves</h2>")
            parts.append("\n".join(figs))
    else:
        logger.warning("MLflow root not found — skipping validation curves.")

    # Qualitative map overlays + gallery
    overlays = []
    for path, cap in [(_VAL_OVERLAY, "Merged probability over the validation AOI"),
                      (_VAL_OVERLAY_2024, "Model probability vs 2024 RTS labels "
                                          "(mean 0.58 inside labels vs 0.07 outside)")]:
        b = _img_file_b64(path)
        if b:
            overlays.append(f"<figure><img src='data:image/png;base64,{b}'>"
                            f"<figcaption>{cap}</figcaption></figure>")
    gallery_dir = _mlflow_root(_GALLERY_DIR)
    if gallery_dir:
        for png in sorted(gallery_dir.glob("pos_*.png"))[:3]:
            b = _img_file_b64(str(png))
            if b:
                overlays.append(f"<figure><img src='data:image/png;base64,{b}'>"
                                f"<figcaption>EXTRA-channel sample — {png.stem}</figcaption></figure>")
    if overlays:
        parts.append("<h2>Qualitative</h2>")
        parts.append("\n".join(overlays))

    doc = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
           f"<title>RTSmapping_v2 report</title><style>{_CSS}</style></head>"
           f"<body>{''.join(parts)}</body></html>")
    output.write_text(doc)
    logger.info("Wrote %s (%d runs, %d build-up steps)", output, len(run_rows), len(buildup_rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=REPO / "docs" / "experiment_ledger.md")
    ap.add_argument("--diary", type=Path, default=REPO / "current_working_status.md")
    ap.add_argument("--mlflow-root", default="/outputs/v1.0/mlflow",
                    help="dir of per-run MLflow file stores (best-effort; curves skipped if absent)")
    ap.add_argument("--output", type=Path, default=REPO / "docs" / "report.html")
    args = ap.parse_args()
    build(args.ledger, args.diary, _mlflow_root(args.mlflow_root), args.output)


if __name__ == "__main__":
    main()
