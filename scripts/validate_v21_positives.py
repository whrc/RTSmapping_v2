"""QC of the v2.1 quality-filtered positive training set (pre-negatives drop).

Read-only over gs://abrupt_thaw/RTS_MODEL_V2/DATA/TRAINING_DATA/. Checks every
positive tile while the data team is still producing, so problems are cheap to
fix before the v2.1 re-baseline:

  1. metadata <-> objects, both directions (PLANET-RGB/ and labels/)
  2. raster integrity: 512x512, EPSG:3857, RGB uint8 3-band, label uint8 1-band
  3. label values in {0, 1, 255}; flag zero-RTS-pixel positives (v2.0 had one)
  4. per-band degradation: any RGB band >50% zero (v2.0 had 209 missing-BLUE)
  5. centroid (metadata lat/lon) inside tile bounds
  6. per-batch / per-region counts

Figures (written to --out-dir):
  qc_gallery.png      - 24 sample tiles, RGB with mask overlay, across batches
  qc_batch_map.png    - batch1/2/3 centroid map
  qc_histograms.png   - RTS-pixel fraction + mean brightness per batch

Usage (inside rts-train:v2; needs ADC + GOOGLE_CLOUD_PROJECT):
    python scripts/validate_v21_positives.py \
        --data-root gs://abrupt_thaw/RTS_MODEL_V2/DATA/TRAINING_DATA \
        --out-dir /outputs/inference/v21_qc [--sample 0]
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    _adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if _adc.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_adc)

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _check_tile(args: tuple[str, str]) -> dict:
    """Open one tile's RGB + label and return per-tile QC facts."""
    data_root, tile_id = args
    out: dict = {"tile_id": tile_id}
    try:
        with rasterio.open(f"{data_root}/PLANET-RGB/{tile_id}.tif") as src:
            out["rgb_shape_ok"] = (src.width, src.height) == (512, 512)
            out["rgb_crs_ok"] = src.crs is not None and src.crs.to_string() == "EPSG:3857"
            out["rgb_bands"] = src.count
            out["rgb_dtype"] = src.dtypes[0]
            rgb = src.read()
            out["bounds"] = tuple(src.bounds)
        out["band_zero_frac"] = [float((rgb[b] == 0).mean()) for b in range(min(3, rgb.shape[0]))]
        out["mean_brightness"] = float(rgb[:3].mean())
        out["rgb_all_zero"] = bool((rgb[:3] == 0).all())
    except Exception as e:  # noqa: BLE001
        out["rgb_error"] = str(e)[:200]
        return out
    try:
        with rasterio.open(f"{data_root}/labels/{tile_id}.tif") as src:
            out["label_shape_ok"] = (src.width, src.height) == (512, 512)
            out["label_bands"] = src.count
            label = src.read(1)
        vals = set(np.unique(label).tolist())
        out["label_values_ok"] = vals <= {0, 1, 255}
        out["label_values"] = sorted(vals)
        out["rts_frac"] = float((label == 1).mean())
    except Exception as e:  # noqa: BLE001
        out["label_error"] = str(e)[:200]
    return out


def make_figures(meta: pd.DataFrame, results: pd.DataFrame, data_root: str,
                 out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Gallery: 24 tiles spread across batches, weighted toward larger RTS.
    rng = np.random.default_rng(42)
    merged = results.merge(meta, left_on="tile_id", right_on="Tile_ID")
    picks = []
    for batch, grp in merged.groupby("Version"):
        grp = grp[grp["rts_frac"].fillna(0) > 0].sort_values("rts_frac", ascending=False)
        n = max(1, int(round(24 * len(grp) / len(merged))))
        idx = rng.choice(len(grp.head(200)), size=min(n, len(grp)), replace=False)
        picks.append(grp.head(200).iloc[idx])
    picks = pd.concat(picks).head(24)

    fig, axes = plt.subplots(4, 6, figsize=(21, 14))
    for ax, (_, row) in zip(axes.ravel(), picks.iterrows()):
        tid = row["tile_id"]
        with rasterio.open(f"{data_root}/PLANET-RGB/{tid}.tif") as src:
            rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        with rasterio.open(f"{data_root}/labels/{tid}.tif") as src:
            label = src.read(1)
        ax.imshow(rgb)
        overlay = np.zeros((*label.shape, 4), dtype=np.float32)
        overlay[label == 1] = (1.0, 0.1, 0.1, 0.45)   # RTS: translucent red
        overlay[label == 255] = (0.6, 0.6, 0.6, 0.45)  # ignore: grey
        ax.imshow(overlay)
        ax.set_title(f"{tid} [{row['Version']}] rts={row['rts_frac']:.1%}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle("v2.1 positives — RGB with RTS mask overlay (red; grey = ignore)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "qc_gallery.png", dpi=110); plt.close(fig)

    # Batch map.
    fig, ax = plt.subplots(figsize=(13, 7))
    for batch, grp in meta.groupby("Version"):
        ax.scatter(grp["centroid_lon"], grp["centroid_lat"], s=6, alpha=0.6, label=batch)
    ax.legend(); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title("v2.1 positive tile centroids by batch")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "qc_batch_map.png", dpi=120); plt.close(fig)

    # Histograms.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for batch, grp in merged.groupby("Version"):
        axes[0].hist(grp["rts_frac"].dropna(), bins=40, alpha=0.5, label=batch)
        axes[1].hist(grp["mean_brightness"].dropna(), bins=40, alpha=0.5, label=batch)
    axes[0].set_title("RTS pixel fraction per tile"); axes[0].set_yscale("log")
    axes[1].set_title("mean RGB brightness per tile")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "qc_histograms.png", dpi=120); plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="gs://abrupt_thaw/RTS_MODEL_V2/DATA/TRAINING_DATA")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--sample", type=int, default=0,
                   help="check only N random tiles (0 = all)")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    setup_logging()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.data_root.rstrip("/")

    # Listing + metadata via the storage client.
    from google.cloud import storage
    bucket_name, prefix = root[5:].split("/", 1)
    client = storage.Client()
    meta = pd.read_csv(io.BytesIO(
        client.bucket(bucket_name).blob(f"{prefix}/metadata.csv").download_as_bytes()))
    rgb_ids, label_ids = set(), set()
    for blob in client.list_blobs(bucket_name, prefix=f"{prefix}/"):
        rest = blob.name[len(prefix) + 1:]
        if rest.startswith("PLANET-RGB/") and rest.endswith(".tif"):
            rgb_ids.add(rest.split("/")[1][:-4])
        elif rest.startswith("labels/") and rest.endswith(".tif"):
            label_ids.add(rest.split("/")[1][:-4])

    meta_ids = set(meta["Tile_ID"].astype(str))
    print(f"\n== v2.1 positive QC — {len(meta)} metadata rows, "
          f"{len(rgb_ids)} RGB, {len(label_ids)} labels ==")
    issues: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            issues.append(f"{name}: {detail}")

    check("metadata->objects", meta_ids <= rgb_ids and meta_ids <= label_ids,
          f"missing RGB: {sorted(meta_ids - rgb_ids)[:5]}; "
          f"missing labels: {sorted(meta_ids - label_ids)[:5]}")
    check("objects->metadata", rgb_ids <= meta_ids and label_ids <= meta_ids,
          f"unregistered RGB: {sorted(rgb_ids - meta_ids)[:5]}; "
          f"unregistered labels: {sorted(label_ids - meta_ids)[:5]}")
    check("metadata classes", set(meta["TrainClass"].unique()) == {"positive"},
          f"classes: {dict(meta['TrainClass'].value_counts())}")

    tile_ids = sorted(meta_ids & rgb_ids)
    if args.sample:
        tile_ids = list(np.random.default_rng(42).choice(tile_ids, args.sample, replace=False))
    logger.info("Checking %d tiles with %d workers ...", len(tile_ids), args.workers)
    with ThreadPoolExecutor(args.workers) as ex:
        results = pd.DataFrame(list(ex.map(_check_tile, [(root, t) for t in tile_ids])))
    results.to_csv(args.out_dir / "qc_per_tile.csv", index=False)

    errs = results[results.get("rgb_error").notna() | results.get("label_error").notna()] \
        if "rgb_error" in results or "label_error" in results else pd.DataFrame()
    check("readable", len(errs) == 0, f"{len(errs)} tiles with read errors "
          f"{errs['tile_id'].tolist()[:5] if len(errs) else ''}")
    good = results[results["rgb_shape_ok"].fillna(False)]
    check("raster geometry", bool(results["rgb_shape_ok"].fillna(False).all()
                                  and results["rgb_crs_ok"].fillna(False).all()
                                  and results["label_shape_ok"].fillna(False).all()),
          f"shape/crs failures: "
          f"{results.loc[~(results['rgb_shape_ok'].fillna(False)), 'tile_id'].tolist()[:5]}")
    check("rgb bands/dtype", bool((results["rgb_bands"].dropna() >= 3).all()
                                  and (results["rgb_dtype"].dropna() == "uint8").all()),
          f"band counts: {dict(results['rgb_bands'].value_counts())}")
    check("label values", bool(results["label_values_ok"].fillna(False).all()),
          f"bad: {results.loc[~results['label_values_ok'].fillna(False), 'tile_id'].tolist()[:5]}")
    zero_rts = results[results["rts_frac"].fillna(1) == 0]
    check("no zero-RTS positives", len(zero_rts) == 0,
          f"{len(zero_rts)} positives with empty masks: {zero_rts['tile_id'].tolist()[:10]}")
    degraded = results[results["band_zero_frac"].apply(
        lambda v: isinstance(v, list) and max(v) > 0.5)]
    check("no degraded bands (>50% zero)", len(degraded) == 0,
          f"{len(degraded)} tiles: {degraded['tile_id'].tolist()[:10]}")

    # Centroid inside bounds (lat/lon -> 3857).
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    m = results.merge(meta, left_on="tile_id", right_on="Tile_ID")
    mx, my = t.transform(m["centroid_lon"].values, m["centroid_lat"].values)
    b = np.array([list(v) for v in m["bounds"]])
    inside = (b[:, 0] <= mx) & (mx <= b[:, 2]) & (b[:, 1] <= my) & (my <= b[:, 3])
    check("centroid in tile bounds", bool(inside.all()),
          f"{int((~inside).sum())} outside: {m.loc[~inside, 'tile_id'].tolist()[:5]}")

    print("\n  per-batch:", dict(meta["Version"].value_counts()))
    print("  regions:", meta["RegionName"].nunique(), "| top:",
          dict(meta["RegionName"].value_counts().head(5)))

    make_figures(meta, results, root, args.out_dir)
    logger.info("Figures + qc_per_tile.csv written to %s", args.out_dir)
    print(f"\n== {len(issues)} issue(s) ==")
    for i in issues:
        print("  -", i)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
