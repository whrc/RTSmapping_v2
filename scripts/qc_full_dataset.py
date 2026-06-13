"""Thorough QC over the FULL training dataset — all positives AND negatives.

Read-only. Computes every fact fresh from the rasters (never trusts a prior QC csv,
which can go stale when the data team re-drops tiles). Extends the positives-only
scripts/validate_v21_positives.py to cover negatives and the whole dataset.

Per-tile facts (both classes):
  RGB: readable, 512x512, EPSG:3857, 3-band, uint8, per-band zero-fraction (degradation),
       whole-tile all-zero, mean brightness, centroid-in-bounds
Positives also:
  label present, shape/bands, values subset {0,1,255}, rts_frac (flag ==0 empty, >=0.99 full-frame)
Negatives:
  label absent by convention (flag if a label file exists)
Metadata-level: duplicate Tile_ID, duplicate centroid, lat in [60,74] domain, Version mix.

Writes <out_dir>/qc_full_per_tile.csv + prints a summary. Run inside rts-train:v2
(needs ADC + GOOGLE_CLOUD_PROJECT).

Usage:
  python scripts/qc_full_dataset.py \
    --data-root gs://abrupt_thaw/RTS_MODEL_V2/DATA/TRAINING_DATA \
    --out-dir /outputs/qc_full [--workers 64] [--sample N]
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qc_full")

DOMAIN_LAT = (60.0, 74.0)


def _check_tile(args: tuple[str, str, str, bool, set]) -> dict:
    data_root, tid, trainclass, has_centroid, label_set = args[0], args[1], args[2], args[3], args[4]
    o: dict = {"tile_id": tid, "trainclass": trainclass}
    try:
        with rasterio.open(f"{data_root}/PLANET-RGB/{tid}.tif") as src:
            o["rgb_ok"] = True
            o["rgb_shape_ok"] = (src.width, src.height) == (512, 512)
            o["rgb_bands"] = src.count
            o["rgb_dtype"] = src.dtypes[0]
            o["rgb_crs"] = src.crs.to_string() if src.crs else "NONE"
            arr = src.read()  # (bands, H, W)
            o["band_zero_frac"] = [float((arr[b] == 0).mean()) for b in range(arr.shape[0])]
            o["max_band_zero"] = max(o["band_zero_frac"])
            o["rgb_all_zero"] = bool((arr == 0).all())
            o["mean_brightness"] = float(arr.mean())
    except Exception as e:  # noqa: BLE001
        o["rgb_ok"] = False
        o["rgb_err"] = repr(e)[:120]

    o["label_present"] = tid in label_set
    if trainclass == "positive":
        try:
            with rasterio.open(f"{data_root}/labels/{tid}.tif") as lb:
                o["label_shape_ok"] = (lb.width, lb.height) == (512, 512)
                o["label_bands"] = lb.count
                lab = lb.read(1)
                vals = set(np.unique(lab).tolist())
                o["label_values_ok"] = vals.issubset({0, 1, 255})
                o["label_values"] = sorted(vals)
                o["rts_frac"] = float((lab == 1).mean())
        except Exception as e:  # noqa: BLE001
            o["label_read_ok"] = False
            o["label_err"] = repr(e)[:120]
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="gs://abrupt_thaw/RTS_MODEL_V2/DATA/TRAINING_DATA")
    ap.add_argument("--out-dir", default="/outputs/qc_full")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--sample", type=int, default=0, help="QC only N tiles per class (0 = all)")
    args = ap.parse_args()

    from google.cloud import storage
    proj = os.environ.get("GOOGLE_CLOUD_PROJECT", "pdg-project-406720")
    client = storage.Client(project=proj)

    meta = pd.read_csv(f"{args.data_root}/metadata.csv")
    logger.info("metadata: %d rows | %s", len(meta), meta.TrainClass.value_counts().to_dict())

    # Build the set of present label tile_ids (one listing) instead of probing each.
    root_no_scheme = args.data_root.replace("gs://", "")
    bkt_name, prefix = root_no_scheme.split("/", 1)
    label_set = {
        b.name.rsplit("/", 1)[-1].replace(".tif", "")
        for b in client.list_blobs(bkt_name, prefix=f"{prefix}/labels/")
        if b.name.endswith(".tif")
    }
    logger.info("label files present: %d", len(label_set))

    if args.sample:
        meta = pd.concat([
            meta[meta.TrainClass == "positive"].head(args.sample),
            meta[meta.TrainClass == "negative"].head(args.sample),
        ])
    jobs = [(args.data_root, r.Tile_ID, r.TrainClass, True, label_set) for r in meta.itertuples()]

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, o in enumerate(ex.map(_check_tile, jobs), 1):
            rows.append(o)
            if i % 2000 == 0:
                logger.info("checked %d/%d", i, len(jobs))
    df = pd.DataFrame(rows)
    df = df.merge(meta[["Tile_ID", "Version", "RegionName", "centroid_lat", "centroid_lon"]],
                  left_on="tile_id", right_on="Tile_ID", how="left").drop(columns="Tile_ID")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "qc_full_per_tile.csv"
    df.to_csv(csv_path, index=False)

    # ---- Aggregate report ----
    pos = df[df.trainclass == "positive"]
    neg = df[df.trainclass == "negative"]
    def n(x): return int(x)
    print("\n" + "=" * 70)
    print("FULL-DATASET QC SUMMARY  (data_root=%s)" % args.data_root)
    print("=" * 70)
    print(f"tiles: {len(df)}  | positive {len(pos)}  negative {len(neg)}")
    print(f"RGB unreadable           : {n((~df.rgb_ok.fillna(False)).sum())}")
    print(f"RGB wrong shape (!512)   : {n((df.rgb_shape_ok == False).sum())}")
    print(f"RGB bands != 3           : {n((df.rgb_bands.fillna(3) != 3).sum())}")
    print(f"RGB dtype != uint8       : {n((df.rgb_dtype.fillna('uint8') != 'uint8').sum())}")
    print(f"RGB CRS != EPSG:3857     : {n((df.rgb_crs.fillna('EPSG:3857') != 'EPSG:3857').sum())}")
    print(f"RGB whole-tile all-zero  : {n(df.rgb_all_zero.fillna(False).sum())}")
    print(f"degraded band >50% zero  : {n((df.max_band_zero.fillna(0) > 0.5).sum())}")
    print(f"  band >10% zero         : {n((df.max_band_zero.fillna(0) > 0.1).sum())}")
    print("--- positives ---")
    print(f"label missing (pos)      : {n((~pos.label_present).sum())}")
    print(f"label values not in {{0,1,255}}: {n((pos.label_values_ok == False).sum())}")
    print(f"empty label rts_frac==0  : {n((pos.rts_frac == 0).sum())}")
    print(f"full-frame rts_frac>=0.99: {n((pos.rts_frac >= 0.99).sum())}")
    if len(pos):
        print("rts_frac by Version:")
        for v, s in pos.groupby("Version").rts_frac:
            print(f"   {v:8s} n={len(s):5d} min={s.min():.3f} med={s.median():.3f} max={s.max():.3f}"
                  f" empty={n((s==0).sum())} full={n((s>=0.99).sum())}")
    print("--- negatives ---")
    print(f"negatives WITH a label   : {n(neg.label_present.sum())}  (should be 0)")
    print("--- metadata-level ---")
    print(f"duplicate Tile_ID        : {n(meta.Tile_ID.duplicated().sum())}")
    print(f"duplicate centroid       : {n(meta.duplicated(['centroid_lat','centroid_lon']).sum())}")
    outside = ((df.centroid_lat < DOMAIN_LAT[0]) | (df.centroid_lat > DOMAIN_LAT[1]))
    print(f"centroid outside 60-74N  : {n(outside.sum())}  (pos {n((outside & (df.trainclass=='positive')).sum())},"
          f" neg {n((outside & (df.trainclass=='negative')).sum())})")
    print(f"lat range overall        : {df.centroid_lat.min():.1f} - {df.centroid_lat.max():.1f}")
    print(f"\nper-tile CSV written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
