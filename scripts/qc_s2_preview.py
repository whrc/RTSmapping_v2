"""Visual-QC preview for the Sentinel-2 download (doc s2_extra_data_prep.md §3/§4).

Pulls a few RGB + NDVI windows straight from Earth Engine (the SSoT
`data/extra_channels.s2_sr_composite` recipe, via computePixels — no batch export
or GCS bucket needed) at real RTS sites, and renders side-by-side PNGs so a human
can confirm imagery quality + that RTS features are visible before any bulk run.

Run inside the rts-train Docker image with `pip install earthengine-api` + ADC:
  python scripts/qc_s2_preview.py --year 2024 --n 8 \
     --points domain/train_points.geojson --positives-only \
     --out-dir /outputs/s2_qc/2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.extra_channels import _bbox, _fetch, s2_bands, s2_sr_composite, tile_grid  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qc_s2")


def _stretch(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Percentile contrast stretch to [0,1] for display (ignores NaN)."""
    a, b = np.nanpercentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-6), 0, 1)


def fetch_rgb_ndvi(bounds, year: int):
    """(RGB HxWx3 reflectance, NDVI HxW) on the tile's co-registered EPSG:3857 grid."""
    grid = tile_grid(bounds)
    comp = s2_sr_composite(_bbox(bounds), year).select(["B4", "B3", "B2"])
    px = _fetch(comp, grid, ["B4", "B3", "B2"])
    rgb = np.stack([px["B4"], px["B3"], px["B2"]], axis=-1)
    ndvi = s2_bands(bounds, grid, year)[0]
    return rgb, ndvi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--points", type=Path, default=Path("domain/train_points.geojson"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8, help="number of sites to preview")
    ap.add_argument("--positives-only", action="store_true", help="only RTS==1 sites")
    ap.add_argument("--size-m", type=float, default=2560.0, help="window side in metres")
    ap.add_argument("--project", default="pdg-project-406720")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from data.extra_channels import init_ee
    init_ee(args.project)

    pts = gpd.read_file(args.points).to_crs("EPSG:3857")
    if args.positives_only and "RTS" in pts.columns:
        pts = pts[pts["RTS"] == 1]
    pts = pts.sample(min(args.n, len(pts)), random_state=args.seed).reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("previewing %d sites (year=%d) -> %s", len(pts), args.year, args.out_dir)

    half = args.size_m / 2.0
    for i, row in pts.iterrows():
        cx, cy = row.geometry.x, row.geometry.y
        bounds = (cx - half, cy - half, cx + half, cy + half)
        region = str(row.get("RegionName", "?"))
        rts = int(row.get("RTS", -1))
        try:
            rgb, ndvi = fetch_rgb_ndvi(bounds, args.year)
        except Exception as e:  # noqa: BLE001
            logger.error("site %d (%s): fetch failed: %r", i, region, repr(e)[:160])
            continue

        fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
        ax[0].imshow(_stretch(rgb)); ax[0].set_title(f"S2 RGB {args.year}")
        m = ax[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        ax[1].set_title("NDVI"); fig.colorbar(m, ax=ax[1], fraction=0.046)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        finite = float(np.isfinite(ndvi).mean())
        fig.suptitle(f"{region}  | RTS={rts} | finite={finite:.0%}", fontsize=10)
        fig.tight_layout()
        out = args.out_dir / f"qc_{i:02d}_rts{rts}_{args.year}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
        logger.info("  [%d/%d] %s (NDVI finite %.0f%%)", i + 1, len(pts), out.name, finite * 100)

    logger.info("DONE -> %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
