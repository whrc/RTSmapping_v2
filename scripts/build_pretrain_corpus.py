"""Build the v2.1 SSL pretraining corpus (spec: pretraining/pretraining.md §2).

Samples 4-ch (RGB+NDVI) 512x512 tiles over the S2-covered south footprint, writes one
.npz per tile + manifest + normalization_stats.json. Tile reads reuse the inference readers
verbatim (CLAUDE Rule 3).

Two-step build so the expensive candidate step (3.6 GB domain CSV load + S2 footprint filter
+ eval-region exclusion, ~13 min, GBs of RAM) runs ONCE, not per shard:

  # 1. plan (once): sample tiles → sample_manifest.csv, then exit
  python scripts/build_pretrain_corpus.py --plan-only --n-target 300000 \
    --quad-index …/quad_index_2025q3.csv --s2-index …/s2_index_2025_south.csv \
    --domain-tiles …/tiles_2025q3_domain_full.csv \
    --regions-geojson gs://…/circumpolar_subregions.geojson \
    --splits-yaml gs://…/splits.yaml --out-dir /mnt/outputs/v2.1/PRETRAIN_CORPUS

  # 2. materialize (parallel, N shards): each reads sample_manifest.csv + writes its slice
  for k in $(seq 0 63); do python scripts/build_pretrain_corpus.py --from-sample \
      --shard $k --n-shards 64 --quad-index … --s2-index … \
      --out-dir /mnt/outputs/v2.1/PRETRAIN_CORPUS & done; wait

Single-process build (pilot): omit --plan-only/--from-sample and pass --n-target.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

from data.normalization import WelfordStats, save_stats  # noqa: E402
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


def plan(args) -> int:
    """Build candidates, stratified-sample, write sample_manifest.csv (tile_id + bbox)."""
    candidates = build_candidates(args)
    sample = corpus_mod.stratified_sample(candidates, args.n_target, seed=args.seed)
    sample = sample[["tile_id", "minx", "miny", "maxx", "maxy"]].copy()
    sample["stratum"] = corpus_mod.stratum_labels(sample)
    out = args.out_dir / "sample_manifest.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    logger.info("Planned %d tiles across %d strata → %s",
                len(sample), sample["stratum"].nunique(), out)
    return 0


def materialize(args, sample: pd.DataFrame) -> int:
    """Read + write the npz for each tile in ``sample``; write shard manifest + partial stats."""
    tiles_dir = args.out_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
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
        np.savez_compressed(tiles_dir / f"{row.tile_id}.npz",
                            rgb=np.clip(rgb, 0, 255).astype(np.uint8),
                            ndvi=ndvi.astype(np.float16))
        valid = ~nodata                          # stats over valid pixels only
        rgb_stats.update(rgb[:, valid])
        ndvi_stats.update(ndvi[np.isfinite(ndvi)][None, :])
        manifest_rows.append({"tile_id": row.tile_id, "minx": row.minx, "miny": row.miny,
                              "maxx": row.maxx, "maxy": row.maxy,
                              "stratum": getattr(row, "stratum", 0),
                              "nodata_frac": float(nodata.mean())})
        n_written += 1

    suffix = "" if args.n_shards == 1 else f".shard{args.shard:03d}"
    pd.DataFrame(manifest_rows).to_csv(args.out_dir / f"manifest{suffix}.csv", index=False)
    # Partial stats as {n, mean, std} per channel-group → merge() pools them exactly.
    partial = {"n": n_written,
               "rgb": {"mean": rgb_stats.means(), "std": rgb_stats.stds()},
               "ndvi": {"mean": ndvi_stats.means(), "std": ndvi_stats.stds()}}
    (args.out_dir / f"partial_stats{suffix}.json").write_text(json.dumps(partial))
    logger.info("Shard %s: wrote %d tiles (%d rejected) → %s",
                suffix or "single", n_written, n_rejected, tiles_dir)
    if args.n_shards == 1:
        _write_final_stats(args.out_dir, [partial])
    return 0


def _pool(groups: list[dict], key: str, n_ch: int) -> tuple[list[float], list[float]]:
    """Pool per-shard (n, mean, std) into a global (mean, std) per channel."""
    total = sum(g["n"] for g in groups) or 1
    means, stds = [], []
    for c in range(n_ch):
        m = sum(g["n"] * g[key]["mean"][c] for g in groups) / total
        ex2 = sum(g["n"] * (g[key]["std"][c] ** 2 + g[key]["mean"][c] ** 2) for g in groups) / total
        means.append(m)
        stds.append(math.sqrt(max(ex2 - m * m, 0.0)))
    return means, stds


def _write_final_stats(out_dir: Path, partials: list[dict]) -> None:
    """Pool per-shard partial stats into the normalization_stats.json schema."""
    from datetime import datetime, timezone
    rgb_m, rgb_s = _pool(partials, "rgb", 3)
    nd_m, nd_s = _pool(partials, "ndvi", 1)
    n = sum(g["n"] for g in partials)
    stats = {
        "dataset_version": "v2.1-pretrain",
        "computed_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_tiles_used": n,
        "rgb": {"channel_names": ["R", "G", "B"], "mean": rgb_m, "std": rgb_s},
        "extra": {"channel_names": ["ndvi"], "mean": nd_m, "std": nd_s},
    }
    save_stats(stats, out_dir / "normalization_stats.json")
    logger.info("Final stats over %d tiles: RGB mean %s; NDVI mean %.4f",
                n, [round(v, 2) for v in rgb_m], nd_m[0])


def merge(args) -> int:
    """Combine all shard manifests + partial stats → manifest.csv + normalization_stats.json."""
    parts = sorted(args.out_dir.glob("manifest.shard*.csv"))
    pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(
        args.out_dir / "manifest.csv", index=False)
    partials = [json.loads(p.read_text()) for p in sorted(args.out_dir.glob("partial_stats.shard*.json"))]
    _write_final_stats(args.out_dir, partials)
    logger.info("Merged %d shards → manifest.csv + normalization_stats.json", len(parts))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--quad-index")
    p.add_argument("--s2-index")
    p.add_argument("--domain-tiles")
    p.add_argument("--regions-geojson")
    p.add_argument("--splits-yaml")
    p.add_argument("--n-target", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-nodata-frac", type=float, default=0.5)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--plan-only", action="store_true", help="write sample_manifest.csv and exit")
    p.add_argument("--from-sample", action="store_true",
                   help="materialize a shard from an existing sample_manifest.csv")
    p.add_argument("--merge", action="store_true", help="combine shard outputs, then exit")
    args = p.parse_args()
    setup_logging()

    if args.merge:
        return merge(args)
    if args.plan_only:
        return plan(args)
    if args.from_sample:
        sample = pd.read_csv(args.out_dir / "sample_manifest.csv")
        sample = sample.iloc[args.shard::args.n_shards]
        logger.info("Shard %d/%d → %d tiles", args.shard, args.n_shards, len(sample))
        return materialize(args, sample)

    # Single-process default: plan + materialize in one go.
    candidates = build_candidates(args)
    sample = corpus_mod.stratified_sample(candidates, args.n_target, seed=args.seed)
    sample = sample[["tile_id", "minx", "miny", "maxx", "maxy"]].copy()
    sample["stratum"] = corpus_mod.stratum_labels(sample)
    return materialize(args, sample)


if __name__ == "__main__":
    raise SystemExit(main())
