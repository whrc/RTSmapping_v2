"""Shared inference setup + per-tile loop (inference.md §8).

Both the single-shot CLI (`scripts/inference.py`) and the queue worker
(`scripts/run_inference_worker.py`) use this module so the model setup, NDVI
windowing, ensemble fusion, NoData handling, and COG writing are defined exactly
once (CLAUDE Rule 3). The CLI runs `run_inference` once over a whole tile list;
the worker calls it per claimed shard, reusing one `InferenceContext`.
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from inference.predictor import (
    assert_runtime_matches_package, load_deployment_package, predict_probs,
    predict_probs_ensemble,
)
from inference.quad_index import load_quad_index
from inference.s2_index import load_s2_index
from inference.tiles import InferenceTileDataset
from inference.writer import NODATA_PROB, Manifest, write_probability_tile
from utils.watchdog import start_stall_watchdog

logger = logging.getLogger(__name__)


def _collate(items: list[dict]) -> dict:
    """Stack a batch, keeping per-tile metadata as lists."""
    return {
        "tile_id": [it["tile_id"] for it in items],
        "image": torch.from_numpy(np.stack([it["image"] for it in items])),
        "nodata_mask": np.stack([it["nodata_mask"] for it in items]),
        "all_nodata": [it["all_nodata"] for it in items],
        "bounds": [tuple(it["bounds"]) for it in items],
    }


def _collate_multiscale(items: list[dict]) -> dict:
    """Stack a multi-scale batch (dataset yields per-scale image + valid mask)."""
    scales = list(items[0]["images"].keys())
    return {
        "tile_id": [it["tile_id"] for it in items],
        "images": {s: torch.from_numpy(np.stack([it["images"][s] for it in items]))
                   for s in scales},
        "valid": {s: np.stack([it["valid"][s] for it in items]) for s in scales},
        "all_nodata": [it["all_nodata"] for it in items],
        "bounds": [tuple(it["bounds"]) for it in items],
    }


def _make_loader(dataset, batch_size: int, num_workers: int, collate_fn) -> DataLoader:
    """Build the inference DataLoader with a fork-safe worker start method.

    ``build_context`` and the queue worker create ``storage.Client()`` (gRPC
    background threads) in the parent *before* the loader spawns workers. The
    default Linux ``fork`` copies those locked gRPC mutexes into every child, so
    a worker can deadlock on its first GCS-adjacent call — the probabilistic
    Banks GPU-0 hang where one shard never produced a tile. ``forkserver`` starts
    each worker from a clean server process, so no parent thread/CUDA state
    crosses. Workers read imagery via GDAL ``/vsigs/`` only and need no GCS client
    of their own, and the dataset is picklable (DataFrames + lazily-built trees).
    """
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "forkserver"
    return DataLoader(dataset, **kwargs)


# Stall watchdog lives in utils/watchdog.py — shared with the acquisition order
# loop (planetscope-download/), which has the same silent-hang failure mode.
#
# Here it is defence-in-depth behind the ``forkserver`` fix in _make_loader: if a
# DataLoader worker still wedges (any cause), the main thread blocks inside
# ``for batch in loader`` while the claim's heartbeat thread keeps the shard alive
# forever — the exact way Banks stranded a shard. Exiting 3 lets the claim go
# stale so the launch script's per-GPU loop restarts the worker, and the stalled
# shard is later reclaimed and resumed from its manifest.


def _crop_center_upsample(arr: np.ndarray, out_size: int, frac: float) -> np.ndarray:
    """Crop the centre ``frac`` fraction of ``arr`` and bilinear-resize to out_size.

    A scale-s(<1) prediction covers 1/s× the 1× footprint; its centre s-fraction is
    the 1× tile (inference.md §6.3 "crop center"), resized back to the output grid.
    """
    h, w = arr.shape
    ch, cw = max(1, int(round(h * frac))), max(1, int(round(w * frac)))
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    center = arr[r0:r0 + ch, c0:c0 + cw]
    up = F.interpolate(torch.from_numpy(center.astype(np.float32))[None, None],
                       size=(out_size, out_size), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def fuse_scale_probs(scale_probs: dict[float, np.ndarray],
                     scale_valid: dict[float, np.ndarray], out_size: int) -> np.ndarray:
    """§7.3 fusion: per-pixel arithmetic mean over valid scales on the 1× grid.

    Each scale's prob (already temperature-scaled per §7.3) is mapped to the 1×
    output grid — scale 1.0 as-is; scale s<1 centre-cropped (fraction s) and
    bilinear-upsampled — and averaged over the scales valid at that pixel. Pixels
    valid only at 1× keep the 1× value (the §6.3 graceful degradation). 1.0 must be
    present (the base grid); its valid mask defines the tile footprint. Returns the
    fused prob (out_size, out_size); pixels outside the 1× footprint are NaN.
    """
    acc = np.zeros((out_size, out_size), np.float32)
    cnt = np.zeros((out_size, out_size), np.float32)
    for s, prob in scale_probs.items():
        if s == 1.0:
            pm, vm = prob, scale_valid[s]
        else:
            pm = _crop_center_upsample(prob, out_size, s)
            vm = _crop_center_upsample(scale_valid[s].astype(np.float32), out_size, s) > 0.5
        acc[vm] += pm[vm]
        cnt[vm] += 1.0
    fused = np.full((out_size, out_size), np.nan, np.float32)
    nz = cnt > 0
    fused[nz] = acc[nz] / cnt[nz]
    return fused


def weights_sha256(package: str) -> str:
    """SHA256 of a package's weights.pth (local or gs://) for the manifest."""
    path = f"{package.rstrip('/')}/weights.pth"
    h = hashlib.sha256()
    if path.startswith("gs://"):
        import gcsfs
        f = gcsfs.GCSFileSystem(token="google_default").open(path[5:], "rb")
    else:
        f = open(path, "rb")
    with f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class InferenceContext:
    """Heavy, run-wide setup loaded once and reused for every shard/tile list."""

    models: list
    pkg: dict
    dep_cfg: dict
    run_cfg: dict
    quad_index: pd.DataFrame
    s2_index: Optional[pd.DataFrame]
    extra_bands: list
    ensemble: bool
    package_paths: list[str]


def build_context(config: dict, packages: list[str], quad_index_path: str,
                  s2_index_path: Optional[str], device: torch.device) -> InferenceContext:
    """Load packages + indices and validate the ensemble/calibration contract.

    Mirrors the §8.2 init: load each deployment package, assert ensemble members
    share calibration + channel layout, assert runtime matches the package (§14),
    and load the quad index (+ S2 index when the package declares EXTRA=NDVI).
    """
    pkgs = [load_deployment_package(pp, device) for pp in packages]
    pkg = pkgs[0]                       # reference for stats / model_cfg / dep_cfg
    dep_cfg = pkg["dep_cfg"]
    ensemble = len(pkgs) > 1
    if ensemble:
        # Members must share the calibration + channel layout (fusion is on the
        # final calibrated prob); otherwise the fused threshold is meaningless.
        for other in pkgs[1:]:
            for k in ("temperature", "threshold", "tta", "precision"):
                if other["dep_cfg"].get(k) != dep_cfg.get(k):
                    raise ValueError(f"ensemble member {k} mismatch: "
                                     f"{other['dep_cfg'].get(k)} != {dep_cfg.get(k)}")
            if other["n_channels"] != pkg["n_channels"]:
                raise ValueError("ensemble member channel count mismatch")
        logger.info("ENSEMBLE inference: %d members, T=%.4f thr=%.3f tta=%s",
                    len(pkgs), dep_cfg["temperature"], dep_cfg["threshold"],
                    dep_cfg.get("tta", "none"))
    assert_runtime_matches_package(config, dep_cfg)

    # EXTRA=NDVI is windowed from the bulk S2 composites on the fly (inference.md §5).
    extra_bands = (pkg["model_cfg"].get("channels") or {}).get("extra") or []
    s2_index = None
    if extra_bands:
        if not s2_index_path:
            raise ValueError("package declares EXTRA channels but no s2_index was "
                             "provided (needed to window NDVI from the S2 composites)")
        s2_index = load_s2_index(s2_index_path)

    quad_index = load_quad_index(quad_index_path)
    return InferenceContext(
        models=[p["model"] for p in pkgs], pkg=pkg, dep_cfg=dep_cfg, run_cfg=config,
        quad_index=quad_index, s2_index=s2_index, extra_bands=extra_bands,
        ensemble=ensemble, package_paths=[str(pp) for pp in packages])


def run_metadata(ctx: InferenceContext, device: torch.device) -> dict:
    """Assemble the §9.4 inference_log.json run metadata for this context."""
    dep_cfg, run_cfg = ctx.dep_cfg, ctx.run_cfg
    return {
        "model_version": "+".join(Path(pp.rstrip("/")).name for pp in ctx.package_paths),
        "deployment_package_path": ctx.package_paths,
        "model_checkpoint_sha": [weights_sha256(pp) for pp in ctx.package_paths],
        "ensemble_members": len(ctx.models),
        "inference_date": datetime.now(timezone.utc).isoformat(),
        "scales_used": dep_cfg.get("scales", [1.0]),
        "tta_config": dep_cfg.get("tta", "none"),
        "precision": dep_cfg.get("precision"),
        "torch_compile": bool(dep_cfg.get("torch_compile", False)),
        "threshold": dep_cfg["threshold"],
        "temperature": dep_cfg["temperature"],
        "stride_px": run_cfg["inference"]["stride_px"],
        "overlap_aggregation": "gaussian_weighted_mean",
        "fusion_sigma_px": run_cfg["inference"]["fusion_sigma_px"],
        "gpu_type": (torch.cuda.get_device_name(device)
                     if device.type == "cuda" else "cpu"),
    }


def _predict_batch(ctx: InferenceContext, images: torch.Tensor, dep_cfg: dict) -> np.ndarray:
    """Forward one batch → clamped prob array (H,W per tile), ensemble or single."""
    if ctx.ensemble:
        probs = predict_probs_ensemble(ctx.models, images,
                                       temperature=dep_cfg["temperature"],
                                       tta=dep_cfg.get("tta", "none"),
                                       precision=dep_cfg.get("precision", "fp32"))
    else:
        probs = predict_probs(ctx.pkg["model"], images,
                              temperature=dep_cfg["temperature"],
                              tta=dep_cfg.get("tta", "none"),
                              precision=dep_cfg.get("precision", "fp32"))
    return probs.clamp_(0.0, 1.0).cpu().numpy()  # §10.1 range guard


class _ProbWriter:
    """Background prob-COG writer (inference.md §8.2).

    The per-tile COG write — a temp-file COG + a single GCS upload — is a
    latency-bound round-trip. Done synchronously in the batch loop it stalls the
    GPU: on an A100 the write, not the read, was the throughput bottleneck
    (benchmark 2026-07-07: GCS-write 2.8 t/s → local/async write ~29–36 t/s).
    Writes run in a thread pool (uploads are I/O-bound → threads overlap them
    with the next batch's read+compute). Bounded in-flight applies backpressure;
    a tile is marked ``done`` only after its write **succeeds** (a crash leaves
    unwritten tiles unmarked → reprocessed on resume). Not thread-safe: only the
    owning loop calls ``submit``/``flush`` (manifest marking stays single-thread).
    """

    def __init__(self, manifest: "Manifest", dtype: str,
                 max_workers: int = 16, max_inflight: int = 512) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._manifest = manifest
        self._dtype = dtype
        self._max_inflight = max_inflight
        self._pending: dict = {}  # future -> tile_id

    def _reap(self, futures) -> None:
        for fut in futures:
            tile_id = self._pending.pop(fut)
            fut.result()  # re-raise a failed write on the owning thread
            self._manifest.mark(tile_id, "done")

    def submit(self, path: str, prob: np.ndarray, bounds, tile_id: str) -> None:
        fut = self._pool.submit(write_probability_tile, path, prob, bounds,
                                dtype=self._dtype)
        self._pending[fut] = tile_id
        self._reap([f for f in self._pending if f.done()])  # non-blocking sweep
        while len(self._pending) >= self._max_inflight:  # backpressure
            done, _ = wait(list(self._pending), return_when=FIRST_COMPLETED)
            self._reap(done)

    def flush(self) -> None:
        """Wait for all writes to finish, marking each done; then shut down."""
        while self._pending:
            done, _ = wait(list(self._pending), return_when=FIRST_COMPLETED)
            self._reap(done)
        self._pool.shutdown(wait=True)


def _run_inference_multiscale(ctx: InferenceContext, todo: pd.DataFrame, out: str,
                              manifest: Manifest, device: torch.device, num_workers: int,
                              scales: list[float],
                              progress_cb: Optional[Callable[[int, int], None]]) -> dict:
    """Multi-scale inference (inference.md §6.3/§7.3): predict each scale, fuse per
    tile, write one fused probability COG. ``scales`` must contain 1.0 (base grid)."""
    if 1.0 not in scales:
        raise ValueError(f"multiscale inference requires 1.0 in scales, got {scales}")
    pkg, dep_cfg = ctx.pkg, ctx.dep_cfg
    output_dtype = ctx.run_cfg["inference"].get("output_dtype", "float32")
    dataset = InferenceTileDataset(todo, ctx.quad_index, pkg["stats"],
                                   s2_index=ctx.s2_index, extra_bands=ctx.extra_bands,
                                   scales=scales)
    loader = _make_loader(dataset, ctx.run_cfg["inference"]["batch_size"],
                          num_workers, _collate_multiscale)
    logger.info("MULTISCALE inference: scales=%s (§7.3 arithmetic-mean fusion)", scales)

    writer = _ProbWriter(manifest, output_dtype,
                         max_workers=ctx.run_cfg["inference"].get("write_threads", 16))
    t0, n_done = time.time(), 0
    last_active = [time.time()]
    stop_watchdog = start_stall_watchdog(
        last_active, ctx.run_cfg["inference"].get("stall_timeout_s", 900.0), out)
    for batch in loader:
        last_active[0] = time.time()
        keep = [i for i, nd in enumerate(batch["all_nodata"]) if not nd]
        for i, nd in enumerate(batch["all_nodata"]):
            if nd:
                manifest.mark(batch["tile_id"][i], "all_nodata")
        if keep:
            scale_probs = {s: _predict_batch(ctx, batch["images"][s][keep].to(device), dep_cfg)
                           for s in scales}
            out_size = scale_probs[1.0].shape[-1]
            for j, i in enumerate(keep):
                per = {s: scale_probs[s][j] for s in scales}
                valid = {s: batch["valid"][s][i] for s in scales}
                fused = fuse_scale_probs(per, valid, out_size)
                # §5.3: output NoData = the 1× footprint; within it, 1× always
                # contributes so fused is finite (0.5× only adds where also valid).
                fused = np.where(valid[1.0], fused, NODATA_PROB).astype(np.float32)
                tile_id = batch["tile_id"][i]
                writer.submit(f"{out}/{tile_id}.tif", fused, batch["bounds"][i],
                              tile_id)
        n_done += len(batch["tile_id"])
        rate = n_done / (time.time() - t0)
        if n_done % 512 < len(batch["tile_id"]):
            logger.info("%d/%d tiles (%.1f tiles/s, ETA %.1f h)", n_done, len(todo),
                        rate, (len(todo) - n_done) / rate / 3600)
            if progress_cb is not None:
                progress_cb(n_done, len(todo))
    writer.flush()
    stop_watchdog()
    manifest.save()
    logger.info("Done (multiscale): %s", manifest.counts())
    return manifest.counts()


def run_inference(ctx: InferenceContext, tiles: pd.DataFrame, output: str,
                  manifest: Manifest, device: torch.device, num_workers: int = 8,
                  scale: float = 1.0,
                  progress_cb: Optional[Callable[[int, int], None]] = None) -> dict:
    """Run §8.2 inference over ``tiles``, writing probability COGs under ``output``.

    Resumable via ``manifest`` (skips tiles already recorded). ``progress_cb`` is
    invoked as ``(n_done, n_total)`` at each progress-log point — the queue worker
    uses it to heartbeat its shard claim. Returns the manifest counts.
    """
    out = output.rstrip("/")
    todo = tiles[~tiles["tile_id"].astype(str).isin(manifest.completed)]
    logger.info("%d tiles total, %d already done, %d to process",
                len(tiles), len(tiles) - len(todo), len(todo))
    if todo.empty:
        manifest.save()
        return manifest.counts()

    pkg, dep_cfg = ctx.pkg, ctx.dep_cfg
    output_dtype = ctx.run_cfg["inference"].get("output_dtype", "float32")
    scales = dep_cfg.get("scales") or [1.0]
    if len(scales) > 1:  # inference.md §6.3/§7.3 multi-scale fusion
        return _run_inference_multiscale(ctx, todo, out, manifest, device,
                                         num_workers, scales, progress_cb)
    dataset = InferenceTileDataset(todo, ctx.quad_index, pkg["stats"], scale=scale,
                                   s2_index=ctx.s2_index, extra_bands=ctx.extra_bands)
    loader = _make_loader(dataset, ctx.run_cfg["inference"]["batch_size"],
                          num_workers, _collate)

    writer = _ProbWriter(manifest, output_dtype,
                         max_workers=ctx.run_cfg["inference"].get("write_threads", 16))
    t0, n_done = time.time(), 0
    last_active = [time.time()]
    stop_watchdog = start_stall_watchdog(
        last_active, ctx.run_cfg["inference"].get("stall_timeout_s", 900.0), out)
    for batch in loader:
        last_active[0] = time.time()
        keep = [i for i, all_nd in enumerate(batch["all_nodata"]) if not all_nd]
        for i, all_nd in enumerate(batch["all_nodata"]):
            if all_nd:
                manifest.mark(batch["tile_id"][i], "all_nodata")
        if keep:
            probs = _predict_batch(ctx, batch["image"][keep].to(device), dep_cfg)
            for j, i in enumerate(keep):
                prob = probs[j]
                prob[batch["nodata_mask"][i]] = NODATA_PROB  # §5.3 output mask
                tile_id = batch["tile_id"][i]
                writer.submit(f"{out}/{tile_id}.tif", np.ascontiguousarray(prob),
                              batch["bounds"][i], tile_id)
        n_done += len(batch["tile_id"])
        rate = n_done / (time.time() - t0)
        if n_done % 512 < len(batch["tile_id"]):
            logger.info("%d/%d tiles (%.1f tiles/s, ETA %.1f h)",
                        n_done, len(todo), rate, (len(todo) - n_done) / rate / 3600)
            if progress_cb is not None:
                progress_cb(n_done, len(todo))

    writer.flush()
    stop_watchdog()
    manifest.save()
    logger.info("Done: %s", manifest.counts())
    return manifest.counts()
