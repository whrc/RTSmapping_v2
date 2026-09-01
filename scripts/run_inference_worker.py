"""Queue worker for the dual-fleet inference run (plan Phase 1).

One process per GPU (`CUDA_VISIBLE_DEVICES` pins it). The worker loads the
deployment package(s) once, then repeatedly claims the next free shard from the
GCS queue (`inference/claim.py`), runs inference over that shard's tile list
(`inference.runner.run_inference`), writes probability COGs under
``<base>/probs/<shard_id>/`` + a per-shard manifest under ``<base>/logs/``, and
marks the shard done. Fast A100s take more shards, slow L4s fewer — the queue
self-balances. Crash/preemption is safe: done markers are the source of truth
and a stale claim is reclaimed by another worker.

Run 8 of these per VM (one per L4/A100):
    for g in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$g python scripts/run_inference_worker.py \
        --config configs/deployment.yaml \
        --base gs://rts-arctic-usw1/inference/2025q3_south \
        --quad-index gs://.../quad_index_2025q3.csv \
        --s2-index   gs://.../s2_index_2025_south.csv \
        --package gs://.../pkgs/seed42 --package gs://.../pkgs/seed43 \
        --package gs://.../pkgs/seed44 &
    done
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

# GDAL /vsigs/ + google-cloud auth via ADC (same bootstrap as train.py).
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    _adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if _adc.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_adc)

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.claim import ClaimStore, default_worker_id  # noqa: E402
from inference.runner import build_context, run_inference, run_metadata  # noqa: E402
from inference.writer import Manifest  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _read_text(path: str) -> str:
    """Read text from a local path or gs:// URI."""
    if str(path).startswith("gs://"):
        import gcsfs
        with gcsfs.GCSFileSystem(token="google_default").open(path[5:], "r") as f:
            return f.read()
    return Path(path).read_text()


def load_shard_ids(base: str) -> list[str]:
    """Read the shard universe from ``<base>/shards/index.json``."""
    index = json.loads(_read_text(f"{base.rstrip('/')}/shards/index.json"))
    return [s["shard_id"] for s in index["shards"]]


def work_loop(store: ClaimStore, shard_ids: Iterable[str],
              process_shard: Callable[[str], None], stale_after_s: float = 1800.0,
              max_shards: Optional[int] = None,
              heartbeat_every_s: float = 240.0) -> int:
    """Drain the queue: claim -> process -> mark_done until no shard is free.

    GPU/torch-free so it is unit-testable. ``process_shard`` does the actual
    inference for one shard id; ``mark_done`` runs only after it returns, so a
    crash mid-shard leaves a reclaimable claim, never a false done marker.
    Returns the number of shards this worker completed.

    A daemon thread heartbeats the claim every ``heartbeat_every_s`` for the
    whole life of the shard. The progress-callback heartbeat alone fires only
    every 512 processed tiles, so a cold worker's first tick can lag the claim
    by many minutes — the 2026-07-05 pre-flight drill saw an *active* L4
    worker's shard reclaimed because its heartbeat was 238 s old before the
    first tile completed. Time-based heartbeats decouple staleness from
    throughput; set ``heartbeat_every_s=0`` to disable (tests).
    """
    import threading

    shard_ids = list(shard_ids)
    n = 0
    while max_shards is None or n < max_shards:
        sid = store.claim_next(shard_ids, stale_after_s)
        if sid is None:
            break
        logger.info("claimed shard %s", sid)
        stop = threading.Event()
        beater = None
        if heartbeat_every_s > 0:
            def _beat(sid=sid, stop=stop):
                while not stop.wait(heartbeat_every_s):
                    try:
                        store.heartbeat(sid)
                    except Exception:  # noqa: BLE001 - never kill the shard for a beat
                        logger.warning("heartbeat failed for %s", sid, exc_info=True)
            beater = threading.Thread(target=_beat, daemon=True)
            beater.start()
        try:
            process_shard(sid)
        finally:
            stop.set()
            if beater is not None:
                beater.join(timeout=5)
        store.mark_done(sid)
        n += 1
        logger.info("done shard %s (%d this worker)", sid, n)
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/deployment.yaml")
    p.add_argument("--base", required=True,
                   help="gs:// base prefix for the run (holds shards/, claims/, "
                        "done/, probs/, logs/)")
    p.add_argument("--quad-index", required=True)
    p.add_argument("--s2-index", default=None)
    p.add_argument("--package", required=True, action="append")
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--stale-after-s", type=float, default=1800.0,
                   help="reclaim a claim whose heartbeat is older than this (s)")
    p.add_argument("--worker-id", default=None)
    p.add_argument("--max-shards", type=int, default=None,
                   help="stop after N shards (testing / partial runs)")
    args = p.parse_args()
    setup_logging()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_cfg = load_config(args.config)
    ctx = build_context(run_cfg, args.package, args.quad_index, args.s2_index, device)
    meta = run_metadata(ctx, device)

    base = args.base.rstrip("/")
    if not base.startswith("gs://"):
        raise ValueError(f"--base must be a gs:// URI, got {base}")
    bucket_name, base_prefix = base[5:].split("/", 1)
    from google.cloud import storage
    bucket = storage.Client().bucket(bucket_name)
    worker_id = args.worker_id or default_worker_id()
    store = ClaimStore(bucket, base_prefix, worker_id=worker_id)

    shard_ids = load_shard_ids(base)
    logger.info("worker %s: %d shards in the universe, device=%s",
                worker_id, len(shard_ids), device)

    def process_shard(sid: str) -> None:
        tiles = pd.read_csv(io.StringIO(_read_text(f"{base}/shards/{sid}.csv")))
        manifest = Manifest(f"{base}/logs/{sid}.json", meta)
        run_inference(ctx, tiles, f"{base}/probs/{sid}", manifest, device,
                      num_workers=args.num_workers, scale=args.scale,
                      progress_cb=lambda _n, _t: store.heartbeat(sid))

    n = work_loop(store, shard_ids, process_shard, stale_after_s=args.stale_after_s,
                  max_shards=args.max_shards)
    logger.info("worker %s finished: %d shards completed", worker_id, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
