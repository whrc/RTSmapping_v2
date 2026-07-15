"""Build the v2.1 SSL pretraining corpus (spec: pretraining/pretraining.md §2).

Samples 4-ch (RGB+NDVI) 512x512 tiles over the S2-covered south footprint, writes one
.npz per tile + manifest.csv + normalization_stats.json. Tile reads reuse the inference
readers verbatim (CLAUDE Rule 3). Materialization is embarrassingly parallel; this script
does a single-process build (use --shard/--n-shards to fan out across processes/VMs).

Usage (5k pilot):
  python scripts/build_pretrain_corpus.py \
    --quad-index /mnt/outputs/inference/quad_index_2025q3.csv \
    --s2-index   /mnt/outputs/inference/s2_index_2025_south.csv \
    --domain-tiles /mnt/outputs/inference/tiles_2025q3_domain_full.csv \
    --regions-geojson gs://rts-mapping-v2/training/v1.0/circumpolar_subregions.geojson \
    --splits-yaml     gs://rts-mapping-v2/training/v1.0/splits.yaml \
    --out-dir /mnt/outputs/v2.1/PRETRAIN_CORPUS_PILOT --n-target 5000
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    _adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if _adc.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_adc)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.normalization import WelfordStats, build_stats_dict, save_stats  # noqa: E402
from data.splits import load_splits_yaml  # noqa: E402
from inference.quad_index import load_quad_index  # noqa: E402
from inference.s2_index import load_s2_index  # noqa: E402
from inference.tiles import _BBoxIndex, read_ndvi_tile, read_tile  # noqa: E402
from pretraining import corpus as corpus_mod  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# Region names that must never appear in the corpus (evaluation regions).
_EVAL_SPLITS = ["val_realistic", "test_realistic"]


def build_candidates(args) -> pd.DataFrame:
    """Domain tiles restricted to the S2 footprint, minus the eval regions."""
    s2_index = load_s2_index(args.s2_index)
    tiles = pd.read_csv(args.domain_tiles)
    logger.info("Domain grid: %d tiles; filtering to S2 footprint…", len(tiles))
    tiles = corpus_mod.filter_to_s2_footprint(tiles, s2_index)
    logger.info("S2-covered candidates: %d tiles", len(tiles))

    splits = load_splits_yaml(args.splits_yaml)
    eval_names = [r for s in _EVAL_SPLITS for r in splits.get(s, [])]
    exclusion = corpus_mod.load_exclusion_polygons(args.regions_geojson, eval_names)
    before = len(tiles)
    tiles = corpus_mod.drop_excluded(tiles, exclusion)
    logger.info("Excluded %d tiles intersecting val/test regions; %d remain",
                before - len(tiles), len(tiles))
    return tiles


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quad-index", required=True)
    p.add_argument("--s2-index", required=True)
    p.add_argument("--domain-tiles", required=True)
    p.add_argument("--regions-geojson", required=True)
    p.add_argument("--splits-yaml", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--n-target", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-nodata-frac", type=float, default=0.5)
    p.add_argument("--shard", type=int, default=0, help="this shard index")
    p.add_argument("--n-shards", type=int, default=1, help="total shards")
    args = p.parse_args()

    setup_logging()
    tiles_dir = args.out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates(args)
    sample = corpus_mod.stratified_sample(candidates, args.n_target, seed=args.seed)
    logger.info("Sampled %d tiles across %d strata",
                len(sample), len(np.unique(corpus_mod.stratum_labels(sample))))

    # Shard split for multi-process builds (deterministic by row order).
    if args.n_shards > 1:
        sample = sample.iloc[args.shard::args.n_shards]
        logger.info("Shard %d/%d → %d tiles", args.shard, args.n_shards, len(sample))

    quad_index = load_quad_index(args.quad_index)
    s2_index = load_s2_index(args.s2_index)
    quad_bbox = _BBoxIndex(quad_index)
    s2_bbox = _BBoxIndex(s2_index)

    rgb_stats = WelfordStats(channel_names=["R", "G", "B"])
    ndvi_stats = WelfordStats(channel_names=["ndvi"])
    manifest_rows: list[dict] = []
    n_written = n_rejected = 0

    for row in tqdm(sample.itertuples(), total=len(sample), desc="tiles"):
        bbox = (row.minx, row.miny, row.maxx, row.maxy)
        rgb, nodata = read_tile(bbox, quad_index, hits=quad_bbox.hits(bbox))
        ndvi = read_ndvi_tile(bbox, s2_index, hits=s2_bbox.hits(bbox))
        if not corpus_mod.quality_ok(rgb, nodata, ndvi, args.max_nodata_frac):
            n_rejected += 1
            continue
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        np.savez_compressed(tiles_dir / f"{row.tile_id}.npz",
                            rgb=rgb_u8, ndvi=ndvi.astype(np.float16))
        # Stats over valid pixels only (NoData RGB is 0; NaN NDVI excluded).
        valid = ~nodata
        rgb_stats.update(rgb[:, valid])          # (3, N_valid)
        fin = np.isfinite(ndvi)
        ndvi_stats.update(ndvi[fin][None, :])    # (1, N_finite)
        manifest_rows.append({
            "tile_id": row.tile_id, "minx": row.minx, "miny": row.miny,
            "maxx": row.maxx, "maxy": row.maxy,
            "stratum": int(corpus_mod.stratum_labels(pd.DataFrame([{
                "minx": row.minx, "miny": row.miny,
                "maxx": row.maxx, "maxy": row.maxy}]))[0]),
            "nodata_frac": float(nodata.mean()),
        })
        n_written += 1

    manifest = pd.DataFrame(manifest_rows)
    suffix = "" if args.n_shards == 1 else f".shard{args.shard:03d}"
    manifest.to_csv(args.out_dir / f"manifest{suffix}.csv", index=False)
    stats = build_stats_dict(rgb_stats, ndvi_stats, dataset_version="v2.1-pretrain",
                             n_tiles_used=n_written)
    save_stats(stats, args.out_dir / f"normalization_stats{suffix}.json")

    logger.info("Wrote %d tiles (%d rejected) → %s", n_written, n_rejected, tiles_dir)
    logger.info("RGB mean %s std %s; NDVI mean %.4f std %.4f",
                [round(m, 2) for m in rgb_stats.means()],
                [round(s, 2) for s in rgb_stats.stds()],
                ndvi_stats.means()[0], ndvi_stats.stds()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
