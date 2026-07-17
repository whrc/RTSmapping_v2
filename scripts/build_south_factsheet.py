"""South RTS factsheet (D3): south_rts_summary.md + .html.

Totals (thresholded vs threshold-free expected area), size & latitude
distributions per confidence tier, and the 0.5° hotspot map — computed from the
shipped products (candidates GPKG + density grids), so re-running after any
product refresh regenerates the sheet. Figures follow the dataviz method:
categorical slots in fixed order (blue/aqua/yellow = high/medium/low),
sequential single-hue map ramp, one axis per chart, thin marks.

Usage:
    python scripts/build_south_factsheet.py \
        --products-dir /outputs/inference/south/products_local \
        --out-dir /outputs/inference/south/products_local
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

TIERS = ("high", "medium", "low")
TIER_COLOR = {"high": "#2a78d6", "medium": "#1baf7a", "low": "#eda100"}
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color="#eceae5", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def fig_size_dist(gdf: gpd.GeoDataFrame,
                  arts_areas: "np.ndarray | None" = None) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.logspace(1, 6, 46)  # MMU≈0 inventory: 10 m² … 100 ha
    if arts_areas is not None:
        ax.hist(arts_areas, bins=bins, color="#dedcd6", zorder=2,
                label=f"ARTS v6 (n={len(arts_areas):,})")
        ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
    for t in TIERS:
        a = gdf.loc[gdf["conf_class"] == t, "area_m2"]
        ax.hist(a, bins=bins, histtype="step", linewidth=2,
                color=TIER_COLOR[t], zorder=3)
        ax.annotate(t, xy=(a.median(), np.histogram(a, bins=bins)[0].max()),
                    color=TIER_COLOR[t], fontsize=9, fontweight="bold",
                    xytext=(0, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("polygon area (m², geodesic — 0.30 outline)", color=INK2)
    ax.set_ylabel("count", color=INK2)
    _style(ax)
    return _png(fig)


def fig_lat_dist(gdf: gpd.GeoDataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.arange(50, 77.5, 1.0)
    for t in TIERS:
        lat = gdf.loc[gdf["conf_class"] == t, "centroid_lat"]
        ax.hist(lat, bins=bins, histtype="step", linewidth=2,
                color=TIER_COLOR[t], label=t, zorder=3)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    ax.set_xlabel("latitude (°N)", color=INK2)
    ax.set_ylabel("count", color=INK2)
    _style(ax)
    return _png(fig)


def fig_hotspot(density_tif: Path) -> str:
    with rasterio.open(density_tif) as src:
        d = src.read(1)
        b = src.bounds
    d = np.where(d <= 0, np.nan, d) / 1e6  # → km² expected per cell
    cmap = LinearSegmentedColormap.from_list(
        "seqblue", ["#eef4fc", "#9dc1ec", "#2a78d6", "#123a6b"])
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    im = ax.imshow(d, extent=(b.left, b.right, b.bottom, b.top), cmap=cmap,
                   norm=LogNorm(vmin=np.nanmax(d) / 1e4, vmax=np.nanmax(d)),
                   aspect="auto", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cb.set_label("expected RTS area (km²/cell)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel("longitude", color=INK2)
    ax.set_ylabel("latitude", color=INK2)
    _style(ax)
    return _png(fig)


def build(products_dir: Path, out_dir: Path,
          arts_geojson: Path | None = None) -> None:
    gdf = gpd.read_file(products_dir / "south_rts_candidates.gpkg")
    arts_areas = None
    if arts_geojson is not None:
        # EPSG:6933 (equal-area) areas ≈ geodesic to <0.5% — fine for a histogram
        arts_areas = gpd.read_file(arts_geojson).to_crs(6933).area.to_numpy()
    grid = gpd.read_file(products_dir / "density_0.5deg.gpkg")
    expected_km2 = grid["expected_rts_m2"].sum() / 1e6
    valid_km2 = grid["valid_km2"].sum()
    n = {t: int((gdf["conf_class"] == t).sum()) for t in TIERS}
    a30 = {t: gdf.loc[gdf["conf_class"] == t, "area_m2"].sum() / 1e6
           for t in TIERS}
    a65 = gdf["area_m2_t65"].sum() / 1e6
    ncls = ({c: int((gdf["rts_class"] == c).sum())
             for c in ("confirmed", "candidate", "marginal")}
            if "rts_class" in gdf.columns else None)
    qc_path = products_dir / "qc_precision_grid.csv"
    qc = pd.read_csv(qc_path) if qc_path.exists() else None

    stats = {
        "candidates": len(gdf), "n": n, "area30_km2": sum(a30.values()),
        "area30": a30, "area65_km2": a65, "expected_km2": expected_km2,
        "valid_km2": valid_km2, "ncls": ncls, "qc": qc,
    }
    imgs = {"size": fig_size_dist(gdf, arts_areas), "lat": fig_lat_dist(gdf),
            "hotspot": fig_hotspot(products_dir / "density_0.5deg_expected_m2.tif")}
    _write_md(stats, out_dir / "south_rts_summary.md")
    _write_html(stats, imgs, out_dir / "south_rts_summary.html")
    logger.info("factsheet written to %s (candidates=%d, expected=%.1f km²)",
                out_dir, len(gdf), expected_km2)


def _write_md(s: dict, path: Path) -> None:
    md = f"""# South 2025Q3 RTS — summary factsheet

Generated from the shipped products (catalog: `south_products.md`).

## Headline numbers

| Statistic | Value |
|---|---|
| Candidate polygons (thr 0.30) | **{s['candidates']:,}** |
| — high tier (max_prob ≥ 0.65) | {s['n']['high']:,} |
| — medium tier (0.45–0.65) | {s['n']['medium']:,} |
| — low tier (0.30–0.45) | {s['n']['low']:,} |
| Total candidate area (0.30 outlines) | {s['area30_km2']:.1f} km² |
| Same objects re-cut at 0.65 (`area_m2_t65`) | {s['area65_km2']:.1f} km² |
| **Expected RTS area (threshold-free, Σ calibrated P)** | **{s['expected_km2']:.1f} km²** |
| Valid imaged area | {s['valid_km2']:,.0f} km² |

How to read the three area numbers: the 0.65 re-cut matches the deployed
operating point; the 0.30 outlines are the permissive envelope of *detected
objects*; the calibrated expectation integrates **all** probability mass —
including diffuse sub-min_blob / sub-0.30 signal no polygon product carries —
so it sits above both and is the abundance estimate, not an inventory area.

Per-tier area (0.30 outlines): high {s['area30']['high']:.1f} /
medium {s['area30']['medium']:.1f} / low {s['area30']['low']:.1f} km².
"""
    if s["ncls"]:
        md += (f"\nrts_class (QC-calibrated rule): confirmed "
               f"{s['ncls']['confirmed']:,} / candidate "
               f"{s['ncls']['candidate']:,} / marginal "
               f"{s['ncls']['marginal']:,}.\n")
    if s["qc"] is not None:
        md += "\n" + _qc_md(s["qc"])
    md += ("\nFigures (size & latitude distributions with ARTS v6 overlay, "
           "hotspot map): see `south_rts_summary.html`. Caveats and tier "
           "definitions: `south_products.md`.\n")
    path.write_text(md)


def _qc_md(qc: "pd.DataFrame") -> str:
    """Measured precision grid (2026-07 QC, 279 ratings) as a markdown table."""
    lines = ["## Measured precision (tier × size band, 2026-07 QC)", "",
             "| tier | band | n rated | precision | 95% CI | unsure |",
             "|---|---|---|---|---|---|"]
    for _, r in qc.iterrows():
        p = ("—" if r["n_rated"] == 0 else f"{r['precision']:.2f}")
        ci = ("—" if r["n_rated"] == 0
              else f"{r['wilson_lo']:.2f}–{r['wilson_hi']:.2f}")
        lines.append(f"| {r['conf_class']} | {r['band']} | {int(r['n_rated'])}"
                     f" | {p} | {ci} | {int(r['n_unsure'])} |")
    lines.append("")
    lines.append("`unsure` verdicts (63/279) are excluded from precision — "
                 "true values lie between the all-false and all-rts extremes. "
                 "Sample is stratified per cell, so pooled rates are not "
                 "map-level precision.")
    return "\n".join(lines) + "\n"


def _write_html(s: dict, imgs: dict, path: Path) -> None:
    tile = ("<div style='background:#f4f3f0;border-radius:8px;padding:14px 18px;"
            "min-width:150px'><div style='font-size:12px;color:#52514e'>{}</div>"
            "<div style='font-size:26px;font-weight:700;color:#0b0b0b'>{}</div></div>")
    tiles = "".join([
        tile.format("candidates (thr 0.30)", f"{s['candidates']:,}"),
        tile.format("high tier", f"{s['n']['high']:,}"),
        tile.format("candidate area", f"{s['area30_km2']:.0f} km²"),
        tile.format("re-cut at 0.65", f"{s['area65_km2']:.0f} km²"),
        tile.format("expected (threshold-free)", f"{s['expected_km2']:.0f} km²"),
    ])
    if s["ncls"]:
        tiles += tile.format("confirmed (rts_class)",
                             f"{s['ncls']['confirmed']:,}")
    qc_html = ""
    if s["qc"] is not None:
        rows = "".join(
            f"<tr><td>{r['conf_class']}</td><td>{r['band']}</td>"
            f"<td>{int(r['n_rated'])}</td>"
            f"<td>{'—' if r['n_rated'] == 0 else f'{r.precision:.2f}'}</td>"
            f"<td>{'—' if r['n_rated'] == 0 else f'{r.wilson_lo:.2f}–{r.wilson_hi:.2f}'}</td>"
            f"<td>{int(r['n_unsure'])}</td></tr>"
            for _, r in s["qc"].iterrows())
        qc_html = (
            "<h2 style='font-size:17px'>Measured precision (2026-07 QC, 279 "
            "ratings)</h2><table style='border-collapse:collapse;font-size:13px'>"
            "<tr style='color:#52514e'><th style='text-align:left;padding:2px "
            "8px'>tier</th><th>band</th><th>n</th><th>precision</th>"
            "<th>95% CI</th><th>unsure</th></tr>" + rows + "</table>"
            "<p style='color:#52514e;font-size:12px'>unsure excluded from "
            "precision; stratified sample — pooled rates ≠ map-level "
            "precision. rts_class rule: confirmed = high tier; candidate = "
            "medium &lt;500 m²; marginal = rest.</p>")
    img = ("<figure style='margin:24px 0'><img style='max-width:100%' "
           "src='data:image/png;base64,{}'><figcaption style='color:#52514e;"
           "font-size:12px'>{}</figcaption></figure>")
    html = f"""<!doctype html><meta charset="utf-8">
<title>South 2025Q3 RTS — factsheet</title>
<body style="font-family:system-ui,sans-serif;max-width:900px;margin:32px auto;
background:#fcfcfb;color:#0b0b0b;padding:0 16px">
<h1 style="font-size:22px">South 2025Q3 RTS — summary factsheet</h1>
<p style="color:#52514e">Pan-Arctic South (≈50–76°N), model v2 3-seed ensemble,
calibrated probabilities. Catalog &amp; caveats: <code>south_products.md</code>.</p>
<div style="display:flex;gap:12px;flex-wrap:wrap">{tiles}</div>
{qc_html}
{img.format(imgs['size'], "Size distribution per confidence tier (log-scale areas, geodesic 0.30 outlines), grey: ARTS v6 pan-Arctic inventory for comparison.")}
{img.format(imgs['lat'], "Latitude distribution per confidence tier.")}
{img.format(imgs['hotspot'], "Hotspot map: expected RTS area per 0.5° cell (threshold-free Σ calibrated P × pixel area), log color scale.")}
</body>"""
    path.write_text(html)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--arts", type=Path, default=None,
                   help="ARTS positive geojson — overlays its size "
                        "distribution on the size panel")
    args = p.parse_args()
    setup_logging()
    build(args.products_dir, args.out_dir, args.arts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
