"""EXTRA RTS-signal sanity check — one PNG per sample, all EXTRA channels as panels.

For N random positives + N random negatives (deterministic --seed), render a panel
figure per tile: RGB + each of the 8 EXTRA bands (NDVI, NBR, SE_PCA1-3, SE_PROTO,
TCB, TCW) + an SE_PCA false-colour composite. On positives the RTS label contour is
overlaid on every panel, so you can see whether channel values track the slump
footprint — the visual accept/reject gate for the EXTRA stack (plan §5).

Run in rts-train Docker (matplotlib/rasterio). Usage:
  python scripts/plot_extra_channels.py \
     --metadata /outputs/v1.0/data_local/metadata.csv \
     --rgb-dir  /outputs/v1.0/data_local/PLANET-RGB \
     --extra-dir /outputs/v1.0/data_local/EXTRA \
     --labels-dir /outputs/v1.0/data_local/labels \
     --out-dir  /outputs/v1.0/qc/extra_vis [--n 10 --seed 42]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.extra_channels import GROUP_BANDS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_extra")

# EXTRA band index -> (title, display spec). Diverging where a zero/contrast is
# meaningful (NDVI/NBR/SE_PROTO); grayscale percentile-stretch otherwise.
BAND_DISPLAY = {
    0: ("NDVI", "RdYlGn"),
    1: ("NBR", "RdYlBu"),
    2: ("SE_PCA1", "gray"),
    3: ("SE_PCA2", "gray"),
    4: ("SE_PCA3", "gray"),
    5: ("SE_PROTO", "proto"),   # diverging, fixed [-1, 1]
    6: ("TCB", "gray"),
    7: ("TCW", "gray"),
}


def _stretch(a: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Percentile stretch finite values to [0, 1] for display."""
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a)
    vmin, vmax = np.percentile(v, lo), np.percentile(v, hi)
    if vmax - vmin < 1e-12:
        return np.zeros_like(a)
    return np.clip((a - vmin) / (vmax - vmin), 0, 1)


def _panel(ax, title, mask=None):
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    if mask is not None and mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors="red", linewidths=0.7)


def plot_tile(tid, rgb_dir, extra_dir, labels_dir, out_path, positive):
    with rasterio.open(rgb_dir / f"{tid}.tif") as ds:
        rgb = ds.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
    with rasterio.open(extra_dir / f"{tid}.tif") as ds:
        extra = ds.read().astype(np.float32)   # (8, H, W)
    mask = None
    if positive:
        lp = labels_dir / f"{tid}.tif"
        if lp.exists():
            with rasterio.open(lp) as ds:
                mask = ds.read(1) == 1

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.3))
    axes = axes.ravel()
    # RGB
    axes[0].imshow(np.clip(rgb / 255.0, 0, 1)); _panel(axes[0], f"RGB ({tid})", mask)
    # 8 EXTRA bands
    for i, (b, (title, spec)) in enumerate(BAND_DISPLAY.items(), start=1):
        a = extra[b]
        if spec == "proto":
            # Cosine-to-RTS-prototype is positive-dominated (~0.5-0.99), so a fixed
            # [-1,1] diverging scale washes out. Center on the tile median and stretch
            # to ±max|p2,p98 - median| so above/below-typical (slump vs intact) reads.
            v = a[np.isfinite(a)]
            med = float(np.median(v)) if v.size else 0.0
            lim = max(abs(np.percentile(v, 2) - med), abs(np.percentile(v, 98) - med)) if v.size else 1.0
            lim = lim or 1.0
            axes[i].imshow(a, cmap="RdBu_r", vmin=med - lim, vmax=med + lim)
        elif spec in ("RdYlGn", "RdYlBu"):
            v = a[np.isfinite(a)]
            lim = np.percentile(np.abs(v), 98) if v.size else 1.0
            axes[i].imshow(a, cmap=spec, vmin=-lim, vmax=lim)
        else:
            axes[i].imshow(_stretch(a), cmap="gray")
        _panel(axes[i], title, mask)
    # SE_PCA false-colour composite (PC1-3 -> RGB)
    pca = np.stack([_stretch(extra[c]) for c in GROUP_BANDS["SE_PCA"]], axis=-1)
    axes[9].imshow(pca); _panel(axes[9], "SE_PCA (PC1-3=RGB)", mask)

    cls = "POSITIVE" if positive else "negative"
    fig.suptitle(f"{tid} — {cls}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=90)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--rgb-dir", required=True, type=Path)
    ap.add_argument("--extra-dir", required=True, type=Path)
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata, dtype={"Tile_ID": str})
    rng = np.random.default_rng(args.seed)

    def _has_rts(tid):
        lp = args.labels_dir / f"{tid}.tif"
        if not lp.exists():
            return False
        with rasterio.open(lp) as ds:
            return bool((ds.read(1) == 1).any())

    pos_all = meta.loc[meta["TrainClass"] == "positive", "Tile_ID"].tolist()
    neg_all = meta.loc[meta["TrainClass"] == "negative", "Tile_ID"].tolist()
    rng.shuffle(pos_all); rng.shuffle(neg_all)
    pos = [t for t in pos_all if _has_rts(t)][: args.n]
    neg = neg_all[: args.n]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tid in pos:
        plot_tile(tid, args.rgb_dir, args.extra_dir, args.labels_dir,
                  args.out_dir / f"pos_{tid}.png", positive=True)
        logger.info("wrote pos_%s.png", tid)
    for tid in neg:
        plot_tile(tid, args.rgb_dir, args.extra_dir, args.labels_dir,
                  args.out_dir / f"neg_{tid}.png", positive=False)
        logger.info("wrote neg_%s.png", tid)
    logger.info("DONE: %d positives + %d negatives -> %s", len(pos), len(neg), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
