"""Annotate the validation overlay with 2024 training RTS labels.

The tiny-area validation (`docs/inference_validation.md`) runs the model on **2025**
imagery. This overlays the **2024** RTS labels (training ground truth) that fall in
the same footprint, so the deployed-model probability can be eyeballed against truth.
RTS persist/grow year-on-year, so 2024 labels are an approximate (not exact) reference.

Reads the merged probability raster + v1.0 metadata, finds positive tiles whose
centroid lands in (a margin around) the canvas, reprojects their label rasters onto
the canvas grid, and renders: 2025 probability heatmap + 2024 RTS label contours +
positive-tile footprints.

Run inside rts-train:v2 with the user ADC mounted (labels read from gs:// via /vsigs/).
Usage: python scripts/annotate_overlay_labels.py [--val-dir /outputs/inference/v1.0_baseline_validation]
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from google.cloud import storage
from pyproj import Transformer
from rasterio.warp import Resampling, reproject

PROJECT = "abruptthawmapping"
BUCKET = "rts-arctic-us"
PREFIX = "training/v1.0/"
TILE_HALF_M = 1225  # ~512px * 4.78 m / 2, to catch footprint overlaps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-dir", default="/outputs/inference/v1.0_baseline_validation", type=Path)
    args = ap.parse_args()

    merged_path = args.val_dir / "merged_prob.tif"
    with rasterio.open(merged_path) as src:
        prob = src.read(1)
        canvas_t, canvas_crs = src.transform, src.crs
        b = src.bounds
        H, W = prob.shape

    client = storage.Client(project=PROJECT)
    import pandas as pd
    meta = pd.read_csv(io.StringIO(
        client.bucket(BUCKET).blob(PREFIX + "metadata.csv").download_as_text()))
    pos = meta[meta.TrainClass == "positive"].copy()
    tf = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    pos["x"], pos["y"] = tf.transform(pos.centroid_lon.values, pos.centroid_lat.values)
    m = ((pos.x > b.left - TILE_HALF_M) & (pos.x < b.right + TILE_HALF_M)
         & (pos.y > b.bottom - TILE_HALF_M) & (pos.y < b.top + TILE_HALF_M))
    hit = pos[m]
    print(f"2024 positive tiles in canvas: {len(hit)}")

    # Build the 2024 label mask on the canvas grid.
    label_canvas = np.zeros((H, W), dtype=np.uint8)
    footprints = []
    for tid in hit.Tile_ID:
        src_path = f"/vsigs/{BUCKET}/{PREFIX}labels/{tid}.tif"
        try:
            with rasterio.open(src_path) as ls:
                lab = ls.read(1)
                lb = ls.bounds
                dst = np.zeros((H, W), dtype=np.uint8)
                reproject(lab, dst, src_transform=ls.transform, src_crs=ls.crs,
                          dst_transform=canvas_t, dst_crs=canvas_crs,
                          resampling=Resampling.nearest)
                label_canvas |= (dst == 1).astype(np.uint8)
                footprints.append((lb.left, lb.bottom, lb.right - lb.left, lb.top - lb.bottom))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {tid}: {e!r}")

    n_label_px = int(label_canvas.sum())
    print(f"2024 RTS label pixels on canvas: {n_label_px}")

    # Render: probability heatmap + 2024 label contours + footprints.
    extent = (b.left, b.right, b.bottom, b.top)
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    for a in ax:
        im = a.imshow(np.ma.masked_less(prob, 0), extent=extent, origin="upper",
                      cmap="magma", vmin=0, vmax=1)
        if n_label_px:
            a.contour(label_canvas, levels=[0.5], colors="cyan", linewidths=1.2,
                      extent=extent, origin="upper")
        for (fx, fy, fw, fh) in footprints:
            a.add_patch(plt.Rectangle((fx, fy), fw, fh, fill=False,
                                      edgecolor="lime", linewidth=0.8, linestyle=":"))
    ax[0].set_title(f"2025 prob + 2024 RTS labels (cyan, {len(hit)} tiles) + footprints (green)")
    # right panel: zoom to the label cluster
    if n_label_px:
        ys, xs = np.where(label_canvas == 1)
        cx = b.left + (xs.mean() + 0.5) * canvas_t.a
        cy = b.top + (ys.mean() + 0.5) * canvas_t.e
        z = 2500
        ax[1].set_xlim(cx - z, cx + z); ax[1].set_ylim(cy - z, cy + z)
        ax[1].set_title("zoom: model prob vs 2024 RTS labels")
    fig.colorbar(im, ax=ax, fraction=0.025, label="RTS probability (2025)")
    out = args.val_dir / "validation_overlay_2024labels.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")
    # quick agreement stat: mean prob inside vs outside 2024 labels
    if n_label_px:
        valid = prob >= 0
        inside = prob[(label_canvas == 1) & valid]
        outside = prob[(label_canvas == 0) & valid]
        print(f"mean prob inside 2024 labels = {inside.mean():.3f} | outside = {outside.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
