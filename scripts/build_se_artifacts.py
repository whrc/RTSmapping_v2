"""Build the SE artifacts — global PCA(3) basis + RTS-mean prototype — ONCE.

Outputs `se_artifacts.npz` = {pca_components (3,64), pca_mean (64,), prototype (64,),
meta}. These MUST travel with the data team's 2025 inference EXTRA generation so
`SE_PCA`/`SE_PROTO` are derived from the identical basis + prototype (CLAUDE Rule 3).

Method (self-contained; numpy SVD, no sklearn):
  * global PCA: sample SE pixels from a random subset of tiles → SVD(3) on centered data.
  * RTS prototype: gather SE vectors at labeled RTS pixels (label==1) over a random subset
    of positive tiles → mean → unit-normalize.

Run in rts-train Docker with earthengine-api + ADC mounted. Uses the same SE fetch
(`data.extra_channels.fetch_se_raw`) as tile generation, so artifacts and tiles agree.

Usage:
  python scripts/build_se_artifacts.py --year 2024 \
     --metadata /outputs/v1.0/data_local/metadata.csv \
     --rgb-dir  /outputs/v1.0/data_local/PLANET-RGB \
     --labels-dir /outputs/v1.0/data_local/labels \
     --out /outputs/v1.0/staging/se_artifacts/se_artifacts.npz
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `data.*`

from data.extra_channels import (  # noqa: E402
    SE_N_BANDS, fetch_se_raw, init_ee, tile_grid,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_se_artifacts")


def _tile_se(rgb_path: Path, year: int) -> np.ndarray:
    """Fetch (64,H,W) SE co-registered to the RGB tile's EPSG:3857 bbox."""
    with rasterio.open(rgb_path) as src:
        bounds = tuple(src.bounds)
    return fetch_se_raw(bounds, tile_grid(bounds), year)


def _rts_mask(label_path: Path) -> np.ndarray | None:
    """(H,W) bool mask of RTS pixels (label==1), or None if absent/empty."""
    if not label_path.exists():
        return None
    with rasterio.open(label_path) as ds:
        m = ds.read(1) == 1
    return m if m.any() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--rgb-dir", required=True, type=Path)
    ap.add_argument("--labels-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--project", default="abruptthawmapping")
    ap.add_argument("--pca-tiles", type=int, default=120, help="random tiles sampled for PCA fit")
    ap.add_argument("--pca-px-per-tile", type=int, default=2000)
    ap.add_argument("--proto-tiles", type=int, default=400, help="random positive tiles for the prototype")
    ap.add_argument("--proto-px-per-tile", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    init_ee(args.project)
    rng = random.Random(args.seed)
    meta = pd.read_csv(args.metadata, dtype={"Tile_ID": str})
    all_ids = meta["Tile_ID"].tolist()
    pos_ids = meta.loc[meta["TrainClass"] == "positive", "Tile_ID"].tolist()
    pca_ids = set(rng.sample(all_ids, min(args.pca_tiles, len(all_ids))))
    proto_ids = set(rng.sample(pos_ids, min(args.proto_tiles, len(pos_ids))))
    union = sorted(pca_ids | proto_ids)
    logger.info("SE artifacts: %d PCA tiles, %d prototype tiles (%d unique fetches)",
                len(pca_ids), len(proto_ids), len(union))

    def one(tid: str):
        rgb = args.rgb_dir / f"{tid}.tif"
        if not rgb.exists():
            return tid, None, None
        se = _tile_se(rgb, args.year)                  # (64,H,W)
        flat = se.reshape(SE_N_BANDS, -1).T            # (H*W, 64)
        pca_px = proto_px = None
        if tid in pca_ids:
            finite = flat[np.isfinite(flat).all(axis=1)]
            if len(finite):
                k = min(args.pca_px_per_tile, len(finite))
                idx = np.random.default_rng(abs(hash(tid)) % (2**32)).choice(len(finite), k, replace=False)
                pca_px = finite[idx]
        if tid in proto_ids:
            mask = _rts_mask(args.labels_dir / f"{tid}.tif")
            if mask is not None:
                rts = flat[mask.reshape(-1)]
                rts = rts[np.isfinite(rts).all(axis=1)]
                if len(rts):
                    k = min(args.proto_px_per_tile, len(rts))
                    idx = np.random.default_rng((abs(hash(tid)) + 1) % (2**32)).choice(len(rts), k, replace=False)
                    proto_px = rts[idx]
        return tid, pca_px, proto_px

    pca_samples: list[np.ndarray] = []
    proto_samples: list[np.ndarray] = []
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, t): t for t in union}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                _tid, pca_px, proto_px = fut.result()
                ok += 1
                if pca_px is not None:
                    pca_samples.append(pca_px)
                if proto_px is not None:
                    proto_samples.append(proto_px)
            except Exception as e:  # noqa: BLE001
                fail += 1
                logger.warning("fetch failed for %s: %s", futs[fut], e)
            if i % 50 == 0:
                logger.info("  %d/%d fetched (ok=%d fail=%d)", i, len(union), ok, fail)

    if not pca_samples or not proto_samples:
        raise SystemExit(f"insufficient samples (pca={len(pca_samples)} proto={len(proto_samples)})")

    # Global PCA(3) via SVD on centered pixel sample.
    X = np.vstack(pca_samples).astype(np.float64)
    pca_mean = X.mean(axis=0)
    Xc = X - pca_mean
    _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    components = vt[:3]
    explained = (s[:3] ** 2) / (s ** 2).sum()
    logger.info("Global PCA: %d px, explained variance (PC1-3) = %s",
                len(X), np.round(explained, 4).tolist())

    # RTS prototype: mean of RTS-pixel SE vectors, unit-normalized.
    P = np.vstack(proto_samples).astype(np.float64)
    mean_vec = P.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm < 1e-12:
        raise SystemExit("prototype has near-zero norm")
    prototype = mean_vec / norm
    cos_sims = (P / np.linalg.norm(P, axis=1, keepdims=True)) @ prototype
    logger.info("RTS prototype: %d px, norm=%.4f, RTS cosine mean=%.4f (5th pct=%.4f)",
                len(P), norm, float(cos_sims.mean()), float(np.percentile(cos_sims, 5)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        pca_components=components.astype(np.float32),
        pca_mean=pca_mean.astype(np.float32),
        prototype=prototype.astype(np.float32),
        meta=np.array({"year": args.year, "seed": args.seed,
                       "pca_px": len(X), "proto_px": len(P),
                       "explained_variance": explained.tolist()}, dtype=object),
    )
    logger.info("Wrote %s", args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
