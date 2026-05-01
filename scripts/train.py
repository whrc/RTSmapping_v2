"""Phase 1 training entry point.

Orchestrates: data pipeline (reused from Phase 0) -> model + loss + optimizer
-> two-phase LR schedule -> curriculum batch sampling -> BF16/FP16 mixed
precision -> EMA -> validation + visualizations + MLflow logging ->
checkpointing + early stopping.

Run:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/smoke.yaml  # real-data smoke on L4

Resume:
    python scripts/train.py --config configs/baseline.yaml \
        --resume runs/rts-v2/resume_latest-0050.pth

Override specific keys (dotted notation; repeatable):
    python scripts/train.py --config configs/baseline.yaml \
        --override seed=43 --override mlflow.run_name=baseline-seed43
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Make sibling top-level packages importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import RTSDataset, parse_extra_spec  # noqa: E402
from data.normalization import load_stats, stats_to_arrays  # noqa: E402
from data.sampler import BalancedBatchSampler, ratio_for_epoch, parse_curriculum_schedule  # noqa: E402
from data.splits import get_tile_ids, load_metadata, load_splits_yaml  # noqa: E402
from data.transforms import build_eval_transforms, build_train_transforms  # noqa: E402
from losses import build_loss  # noqa: E402
from models import build_model  # noqa: E402
from training import (  # noqa: E402
    checkpoint as ckpt_mod,
    early_stopping as es_mod,
    ema as ema_mod,
    freeze as freeze_mod,
    metrics as metrics_mod,
    mlflow_utils,
    scheduler as scheduler_mod,
    visualizations as viz,
)
from utils.config import load_config, resolve_path  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argparse + config
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTS Phase 1 training entry point.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--resume", type=Path, default=None,
                   help="resume_latest-*.pth from a previous run")
    p.add_argument("--override", action="append", default=[],
                   help="dotted.key=value; repeatable")
    p.add_argument("--device", default=None,
                   help="cuda | cpu; default auto-detect")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to write checkpoints; default ./runs/<run_name>")
    return p.parse_args()


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `--override key.path=value` strings onto cfg (in place)."""
    for entry in overrides:
        if "=" not in entry:
            raise ValueError(f"Bad --override entry: {entry!r} (expected key=value)")
        key, raw = entry.split("=", 1)
        # Type-coerce numeric / boolean literals; leave strings as-is.
        val: Any = raw
        if raw.lower() in {"true", "false"}:
            val = raw.lower() == "true"
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    pass
        _set_nested(cfg, key.split("."), val)
        logger.info("override: %s = %r", key, val)
    return cfg


def _set_nested(cfg: dict, path: list[str], value: Any) -> None:
    node = cfg
    for part in path[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[path[-1]] = value


# ---------------------------------------------------------------------------
# Device and precision
# ---------------------------------------------------------------------------


@dataclass
class PrecisionSetup:
    requested: str              # what config asked for
    effective: str              # what we're actually running
    autocast_dtype: torch.dtype | None
    use_scaler: bool
    scaler: torch.amp.GradScaler | None


def _configure_precision(cfg: dict, device: torch.device) -> PrecisionSetup:
    """Pick BF16/FP16/FP32 based on config + hardware support.

    BF16 is preferred on A100/H100 (no scaler needed, wider exponent range).
    Fall back to FP16 on L4; log a warning if the config asked for BF16 but the
    GPU doesn't support it (plan risk #8).
    """
    requested = cfg["training"].get("precision", "bf16").lower()
    if device.type != "cuda":
        logger.info("Non-CUDA device (%s): forcing fp32", device)
        return PrecisionSetup("fp32", "fp32", None, False, None)

    if requested == "bf16":
        if torch.cuda.is_bf16_supported():
            return PrecisionSetup("bf16", "bf16", torch.bfloat16, False, None)
        logger.warning("BF16 requested but this GPU lacks native support; using fp16")
        requested = "fp16"

    if requested == "fp16":
        scaler = torch.amp.GradScaler("cuda")
        return PrecisionSetup("fp16", "fp16", torch.float16, True, scaler)

    return PrecisionSetup("fp32", "fp32", None, False, None)


# ---------------------------------------------------------------------------
# Data / sampler / loaders
# ---------------------------------------------------------------------------


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy and Python RNGs per-worker (plan risk #13).

    Without this, all DataLoader workers share the same NumPy RNG state and
    produce correlated augmentations. torch.initial_seed() is already
    per-worker-offset by PyTorch.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _filter_train_positive_subset(
    train_ids: list[str],
    metadata: pd.DataFrame,
    subset_pct: int,
) -> list[str]:
    """Deterministic seeded subsample of positive train tiles; negatives untouched.

    Used by Phase 0 §3.2 LR range test (30 %) and Phase 2 §5.1 (25/50/75/100 %).
    Seed is fixed at 42 so subset composition is stable across runs.
    """
    if not 1 <= subset_pct <= 100:
        raise ValueError(f"train_positive_subset_pct must be 1..100; got {subset_pct}")
    class_by_id = metadata.set_index("Tile_id")["TrainClass"].to_dict()
    positives = sorted(tid for tid in train_ids if class_by_id.get(tid) == "Positive")
    negatives = [tid for tid in train_ids if class_by_id.get(tid) == "Negative"]
    n_keep = max(1, round(len(positives) * subset_pct / 100.0))
    rng = random.Random(42)
    kept_positives = sorted(rng.sample(positives, n_keep))
    return kept_positives + negatives


def _setup_data(cfg: dict) -> dict:
    """Build datasets, samplers, and loaders for train + val_realistic.

    Returns a dict with keys: metadata, train_ds, train_loader, train_sampler,
    val_ds, val_loader, stats, extra_channels, normalization_stats_path.
    """
    data_root = cfg["data"]["data_root"]
    tile_size = int(cfg["data"]["tile_size"])
    ignore_idx = int(cfg["data"]["label_ignore_index"])
    boundary = cfg["loss"]["boundary_handling"]
    boundary_w = int(cfg["loss"].get("boundary_ignore_width", 3))

    metadata = load_metadata(resolve_path(data_root, cfg["data"]["metadata_csv"]))
    splits = load_splits_yaml(resolve_path(data_root, cfg["data"]["splits_yaml"]))

    extra_channels = parse_extra_spec(cfg["channels"].get("extra", []))
    stats_path = cfg["data"]["normalization_stats_path"]
    # Stats may not exist yet for the synthetic smoke; Dataset handles that path.
    # Only swallow FileNotFoundError — corrupt JSON / schema mismatches must surface.
    try:
        stats = load_stats(stats_path)
    except FileNotFoundError:
        stats = None
        logger.warning("Normalization stats not found at %s; using unit stats", stats_path)

    tr_aug = build_train_transforms(tile_size, cfg["augmentation"])
    ev_aug = build_eval_transforms()

    train_ids = get_tile_ids("train", metadata, splits)
    val_ids = get_tile_ids("val_realistic", metadata, splits)
    logger.info("Tile counts: train=%d, val_realistic=%d", len(train_ids), len(val_ids))

    # Positive-subset filter (Phase 0 §3.2 LR test, Phase 2 §5.1 data scale).
    # Deterministic: seed is fixed at 42 regardless of cfg.seed so subset
    # composition is stable across multi-seed reruns.
    subset_pct = cfg.get("splits", {}).get("train_positive_subset_pct")
    if subset_pct is not None and int(subset_pct) < 100:
        train_ids = _filter_train_positive_subset(train_ids, metadata, int(subset_pct))
        logger.info("Filtered train tiles to %d%% positive subset → %d tiles",
                    int(subset_pct), len(train_ids))

    def _make_ds(tile_ids, transform):
        return RTSDataset(
            tile_ids=tile_ids,
            metadata=metadata,
            data_root=data_root,
            rgb_dir=cfg["data"]["rgb_dir"],
            extra_dir=cfg["data"]["extra_dir"],
            labels_dir=cfg["data"]["labels_dir"],
            extra_channels=extra_channels,
            norm_stats_path=stats_path if stats is not None else None,
            transform=transform,
            tile_size=tile_size,
            label_ignore_index=ignore_idx,
            boundary_handling=boundary,
            boundary_ignore_width=boundary_w,
        )

    train_ds = _make_ds(train_ids, tr_aug)
    val_ds = _make_ds(val_ids, ev_aug)

    bs = int(cfg["training"]["batch_size"])
    n_workers = int(cfg["training"]["num_workers"])
    pin = bool(cfg["training"]["pin_memory"])
    pref = int(cfg["training"]["prefetch_factor"])
    persist = bool(cfg["training"]["persistent_workers"]) and n_workers > 0
    drop_last = bool(cfg["training"]["drop_last"])

    train_sampler = BalancedBatchSampler(
        tile_ids=train_ids,
        metadata=metadata,
        batch_size=bs,
        schedule=cfg["sampling"]["curriculum_schedule"],
        seed=int(cfg["seed"]),
        epoch=1,
        drop_last=drop_last,
    )
    # Seeded generator covers any shuffling in the main process; per-worker
    # numpy/random seeds are handled by _worker_init_fn.
    loader_generator = torch.Generator().manual_seed(int(cfg["seed"]))
    loader_kwargs = dict(
        num_workers=n_workers,
        pin_memory=pin,
        persistent_workers=persist,
        worker_init_fn=_worker_init_fn,
        generator=loader_generator,
    )
    if n_workers > 0:
        loader_kwargs["prefetch_factor"] = pref

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, drop_last=False, **loader_kwargs,
    )

    return {
        "metadata": metadata,
        "splits": splits,
        "train_ds": train_ds,
        "train_loader": train_loader,
        "train_sampler": train_sampler,
        "val_ds": val_ds,
        "val_loader": val_loader,
        "val_ids": val_ids,
        "stats": stats,
        "extra_channels": extra_channels,
        "normalization_stats_path": stats_path,
    }


# ---------------------------------------------------------------------------
# Positive-tile pre-warming (plan risk #11)
# ---------------------------------------------------------------------------


def _prewarm_positive_tiles(ds: RTSDataset) -> None:
    """Read every positive tile once to warm the gcsfuse/filesystem cache.

    Synchronous — runs before epoch 1. For typical positive counts (few hundred
    tiles) this completes in seconds over gcsfuse with the cache flags set per
    configs/baseline.yaml training.gcsfuse.
    """
    pos_ids = [tid for tid in ds.tile_ids if ds.is_positive(tid)]
    if not pos_ids:
        logger.info("No positive tiles to pre-warm")
        return
    logger.info("Pre-warming %d positive tiles", len(pos_ids))
    t0 = time.time()
    for tid in pos_ids:
        try:
            ds._read_rgb(tid)
            ds._read_label(tid)
            if ds.extra_channels:
                ds._read_extra(tid)
        except Exception as e:  # noqa: BLE001
            logger.warning("Pre-warm failed on tile %s: %s", tid, e)
    logger.info("Pre-warm done in %.2fs", time.time() - t0)


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    precision: PrecisionSetup,
    device: torch.device,
    *,
    grad_clip_norm: float,
    ema: ema_mod.EMAModel | None,
    exposure_counter: dict[str, int],
    per_step_lr_setter: Callable[..., None] | None = None,
    global_step_offset: int = 0,
    total_steps: int = 1,
    lr_history: list[tuple[int, float, float]] | None = None,
) -> dict[str, float]:
    """Run one training epoch. Returns per-epoch averages.

    `per_step_lr_setter`, if provided, is called as
    `setter(optimizer, step=global_step, total_steps=total_steps)` before each
    step — used by the lr_range_test scheduler. `lr_history`, if provided,
    accumulates `(global_step, lr, loss)` triples for post-run plotting.
    """
    model.train()
    running_loss = 0.0
    running_n = 0
    nan_count = 0
    scaler_halves = 0
    scaler_prev_scale: float | None = None
    if precision.scaler is not None:
        scaler_prev_scale = precision.scaler.get_scale()

    for step, batch in enumerate(loader):
        global_step = global_step_offset + step
        if per_step_lr_setter is not None:
            per_step_lr_setter(optimizer, step=global_step, total_steps=total_steps)
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        tile_ids = batch["tile_id"]

        # Exposure log — per-tile sample counter (plan risk #9).
        for tid in tile_ids:
            exposure_counter[tid] = exposure_counter.get(tid, 0) + 1

        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.amp.autocast(device_type=device.type, dtype=precision.autocast_dtype)
            if precision.autocast_dtype is not None
            else nullcontext()
        )
        with autocast_ctx:
            logits = model(images)
            loss = loss_fn(logits, labels)

        if not torch.isfinite(loss):
            nan_count += 1
            logger.warning("Non-finite loss at step %d; skipping update", step)
            continue

        if precision.scaler is not None:
            precision.scaler.scale(loss).backward()
            precision.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            precision.scaler.step(optimizer)
            precision.scaler.update()
            # Track chronic scaler halving (plan risk #8).
            cur = precision.scaler.get_scale()
            if scaler_prev_scale is not None and cur < scaler_prev_scale:
                scaler_halves += 1
            scaler_prev_scale = cur
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        bs = images.size(0)
        loss_val = float(loss.detach().cpu())
        running_loss += loss_val * bs
        running_n += bs

        if lr_history is not None:
            lr_history.append((global_step, float(optimizer.param_groups[0]["lr"]), loss_val))

    avg_loss = running_loss / max(1, running_n)
    out = {"train_loss": avg_loss, "train_nan_steps": float(nan_count)}
    if precision.scaler is not None:
        out["scaler_scale"] = float(precision.scaler.get_scale())
        out["scaler_halves_this_epoch"] = float(scaler_halves)
    return out


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    precision: PrecisionSetup,
    device: torch.device,
    cfg: dict,
) -> tuple[dict[str, float], metrics_mod.ValidationAccumulator, list[dict]]:
    """Run validation, returning (metrics, accumulator, preview_records).

    preview_records is a list of {tile_id, image, label, prob} entries collected
    from the validation loop to be used by the prediction-preview figure. The
    caller filters to the fixed preview-tile set after this function returns.
    """
    model.eval()
    acc = metrics_mod.ValidationAccumulator(cfg)
    preview_records: list[dict] = []
    total_loss = 0.0
    total_n = 0

    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=precision.autocast_dtype)
        if precision.autocast_dtype is not None
        else nullcontext()
    )

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        tile_ids = batch["tile_id"]

        with autocast_ctx:
            logits = model(images)
            loss = loss_fn(logits, labels)

        # Cast logits to float32 for metric computation (BF16 probs cause tiny
        # numeric noise in PR-AUC).
        logits_f32 = logits.float()
        acc.update(logits_f32, labels, tile_ids)

        probs = torch.sigmoid(logits_f32).squeeze(1).cpu().numpy()
        images_cpu = images.detach().cpu().numpy()
        labels_cpu = labels.detach().cpu().numpy()
        for i, tid in enumerate(tile_ids):
            preview_records.append({
                "tile_id": tid,
                "image": images_cpu[i],
                "label": labels_cpu[i].astype(np.int64),
                "prob": probs[i],
            })

        total_loss += float(loss.detach().cpu()) * images.size(0)
        total_n += images.size(0)

    metrics = acc.compute()
    metrics["val_loss"] = total_loss / max(1, total_n)
    return metrics, acc, preview_records


# ---------------------------------------------------------------------------
# Visualization helpers (driven by cfg + accumulator + preview records)
# ---------------------------------------------------------------------------


def _select_preview_tiles(cfg: dict, metadata: pd.DataFrame, val_ids: list[str]) -> list[str]:
    """Return the fixed 3+3 preview tile IDs (pass-1 heuristic)."""
    picked = viz.pick_preview_tiles_pass1(
        metadata.set_index("Tile_id"),
        val_ids,
        n_positive=3,
        n_negative=3,
        seed=int(cfg["seed"]),
    )
    return picked["positive"] + picked["negative"]


def _rotate_artifacts(artifact_dir: Path, glob: str, keep: int) -> None:
    """Delete older files matching glob; keep newest N (plan risk #19)."""
    files = sorted(artifact_dir.glob(glob))
    while len(files) > keep:
        victim = files.pop(0)
        victim.unlink()


def _render_and_log_figures(
    epoch: int,
    preview_records: list[dict],
    preview_tile_ids: list[str],
    acc: metrics_mod.ValidationAccumulator,
    metrics: dict[str, float],
    stats: dict | None,
    extra_channels,
    out_dir: Path,
) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Preview tiles (ordered per preview_tile_ids).
    preview = [r for tid in preview_tile_ids
               for r in preview_records if r["tile_id"] == tid]
    if preview and stats is not None:
        mean, std = stats_to_arrays(stats, with_extra=bool(extra_channels))
        preview_png = fig_dir / f"preview_epoch_{epoch:04d}.png"
        viz.prediction_preview_grid(preview, mean[:3], std[:3], preview_png)
        mlflow.log_artifact(str(preview_png))

    # PR curves across ratios (use accumulator's cached logit/label data).
    logits_by_ratio: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    pos = [t for t in acc._tiles if t.is_positive_tile]
    neg = [t for t in acc._tiles if not t.is_positive_tile]
    if pos:
        for r in acc.ratios:
            needed = r * len(pos)
            if not neg:
                continue
            idx = (acc._rng.choice(len(neg), size=needed, replace=False)
                   if len(neg) >= needed else
                   acc._rng.choice(len(neg), size=needed, replace=True))
            sub = [neg[i] for i in idx]
            logits = np.concatenate([t.valid_logits for t in pos + sub])
            labels = np.concatenate([t.valid_labels for t in pos + sub])
            logits_by_ratio[r] = (logits, labels)
    if logits_by_ratio:
        pr_png = fig_dir / f"pr_curves_epoch_{epoch:04d}.png"
        viz.pr_curves_at_ratios(logits_by_ratio, pr_png)
        mlflow.log_artifact(str(pr_png))

    # Probability histogram (all valid val pixels).
    all_probs = np.concatenate([1.0 / (1.0 + np.exp(-t.valid_logits)) for t in acc._tiles])
    hist_png = fig_dir / f"prob_hist_epoch_{epoch:04d}.png"
    viz.probability_histogram(all_probs, hist_png)
    mlflow.log_artifact(str(hist_png))

    # Confusion matrix.
    # Reconstruct counts from accumulator (obj_tp is object-level; pixel TP/FP/FN
    # already tracked internally). For tn at pixel level, derive from total valid pixels.
    total_valid = sum(t.valid_logits.size for t in acc._tiles)
    tn = max(0, total_valid - acc.pixel_tp - acc.pixel_fp - acc.pixel_fn)
    cm_png = fig_dir / f"confusion_epoch_{epoch:04d}.png"
    viz.confusion_matrix_pixel(acc.pixel_tp, acc.pixel_fp, acc.pixel_fn, tn, cm_png)
    mlflow.log_artifact(str(cm_png))

    # Keep-last-10 per figure type.
    for pattern in ("preview_epoch_*.png", "pr_curves_epoch_*.png",
                    "prob_hist_epoch_*.png", "confusion_epoch_*.png"):
        _rotate_artifacts(fig_dir, pattern, 10)


# ---------------------------------------------------------------------------
# Main training orchestration
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    cfg = load_config(args.config)
    _apply_overrides(cfg, args.override)

    # Output directory.
    run_name = cfg["mlflow"].get("run_name", "run")
    out_dir = args.out_dir or Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(level="INFO", log_file=str(out_dir / "train.log"))
    logger.info("Starting train.py with cfg=%s out_dir=%s", args.config, out_dir)

    deterministic = bool(cfg.get("deterministic", False))
    if not deterministic and run_name.startswith("final"):
        logger.warning(
            "run_name=%r looks like a final-lock run but cfg.deterministic=false; "
            "set deterministic: true in the final_seed* config for reproducible CUDNN.",
            run_name,
        )
    seed_everything(int(cfg["seed"]), deterministic=deterministic)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)
    precision = _configure_precision(cfg, device)
    logger.info("Precision: requested=%s effective=%s use_scaler=%s",
                precision.requested, precision.effective, precision.use_scaler)

    # Data
    data = _setup_data(cfg)
    preview_tile_ids = _select_preview_tiles(cfg, data["metadata"], data["val_ids"])

    # Model + loss + optimizer
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)

    wd = float(cfg["optimizer"]["weight_decay"])
    frozen_lr = float(cfg["lr_schedule"]["frozen_lr"])
    base_lr = float(cfg["lr_schedule"]["base_lr"])
    backbone_mult = float(cfg["lr_schedule"]["backbone_lr_multiplier"])

    param_groups = freeze_mod.build_param_groups(
        model, decoder_lr=frozen_lr, backbone_lr=frozen_lr,
        weight_decay=wd,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=frozen_lr)
    set_lrs = scheduler_mod.make_lr_setter(cfg)
    is_range_test = scheduler_mod.is_lr_range_test(cfg)
    grad_clip = float(cfg["optimizer"]["gradient_clip_norm"])

    # Phase 1: backbone frozen.
    freeze_mod.freeze_backbone(model)
    ema: ema_mod.EMAModel | None = None  # constructed at unfreeze
    lr_history: list[tuple[int, float, float]] | None = [] if is_range_test else None

    # Checkpointing + early stopping.
    ckpt_mgr = ckpt_mod.CheckpointManager(out_dir / "checkpoints", keep_last_n=3)
    es = es_mod.EarlyStopping(
        metric_name=cfg["training"]["early_stopping"]["metric"],
        patience=int(cfg["training"]["early_stopping"]["patience"]),
        min_delta=float(cfg["training"]["early_stopping"]["min_delta"]),
        start_epoch=int(cfg["training"]["early_stopping"]["start_epoch"]),
        smoothing_window=int(cfg["training"]["early_stopping"].get("smoothing_window", 3)),
    )

    max_epochs = int(cfg["training"]["max_epochs"])
    val_frequency = int(cfg["training"]["val_frequency"])
    freeze_epochs = int(cfg["lr_schedule"]["freeze_backbone_epochs"])

    # MLflow
    mlflow_run = mlflow_utils.setup_mlflow(cfg)
    mlflow_utils.log_config_artifact(cfg, out_dir)
    mlflow_utils.log_requirements_frozen(out_dir)

    # Resume (if requested). Restores EMA shadow weights when applicable so
    # post-resume validation does not silently fall back to live weights.
    start_epoch = 1
    if args.resume is not None:
        start_epoch, ema = _resume_from(
            args.resume, model, optimizer, precision, es, cfg,
            freeze_epochs=freeze_epochs,
        )
        if ema is not None and start_epoch > freeze_epochs + 1:
            freeze_mod.unfreeze_backbone(model)
        logger.info("Resumed from %s at epoch %d (ema=%s)",
                    args.resume, start_epoch, ema is not None)

    # Pre-warm positive tiles once (epoch 1 only).
    if start_epoch == 1:
        _prewarm_positive_tiles(data["train_ds"])

    exposure_counter: dict[str, int] = {}
    nan_events: list[dict] = []
    t_start = time.time()

    # For lr_range_test, ramp LR per-step across the entire run.
    steps_per_epoch = len(data["train_loader"])
    total_steps = steps_per_epoch * max_epochs

    try:
        for epoch in range(start_epoch, max_epochs + 1):
            # Unfreeze transition.
            if epoch == freeze_epochs + 1 and ema is None:
                freeze_mod.unfreeze_backbone(model)
                ema = ema_mod.EMAModel(model, decay=float(cfg["ema"]["decay"]))
                logger.info("Unfroze backbone; EMA initialised (decay=%g)", ema.decay)

            # Per-epoch LR update (skipped for range-test mode where LR moves per-step).
            if not is_range_test:
                set_lrs(optimizer, epoch)
            current_lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
            current_ratio = ratio_for_epoch(
                parse_curriculum_schedule(cfg["sampling"]["curriculum_schedule"]),
                epoch,
            )

            # Curriculum step.
            data["train_sampler"].set_epoch(epoch)

            # Train.
            epoch_t0 = time.time()
            global_step_offset = (epoch - 1) * steps_per_epoch
            train_metrics = _train_one_epoch(
                model, data["train_loader"], loss_fn, optimizer, precision, device,
                grad_clip_norm=grad_clip, ema=ema, exposure_counter=exposure_counter,
                per_step_lr_setter=set_lrs if is_range_test else None,
                global_step_offset=global_step_offset,
                total_steps=total_steps,
                lr_history=lr_history,
            )
            epoch_t = time.time() - epoch_t0

            # Log training scalars.
            scalars: dict[str, float] = {
                **train_metrics,
                "lr_decoder": current_lrs.get("decoder", 0.0),
                "lr_backbone": current_lrs.get("backbone", 0.0),
                "curriculum_neg_per_pos": float(current_ratio),
                "epoch_time_s": epoch_t,
            }
            if device.type == "cuda":
                scalars["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                torch.cuda.reset_peak_memory_stats()
            mlflow_utils.log_metrics_step(scalars, epoch)
            logger.info("epoch=%d train_loss=%.4f time=%.1fs ratio=1:%d",
                        epoch, train_metrics["train_loss"], epoch_t, current_ratio)
            if train_metrics.get("train_nan_steps", 0) > 0:
                nan_events.append({"epoch": epoch, "nan_steps": int(train_metrics["train_nan_steps"])})

            # Validation cadence.
            if epoch % val_frequency != 0 and epoch != max_epochs:
                continue

            # Swap EMA in for validation (if EMA exists — post-unfreeze).
            swap_ctx = ema.swap_in(model) if ema is not None else nullcontext()
            with swap_ctx:
                val_metrics, acc, preview_records = _validate(
                    model, data["val_loader"], loss_fn, precision, device, cfg,
                )

            mlflow_utils.log_metrics_step(val_metrics, epoch)
            logger.info(
                "val epoch=%d pr_auc_geomean=%.4f pixel_iou=%.4f obj_f1=%.4f",
                epoch,
                val_metrics.get("val_realistic_pr_auc_geomean", 0.0),
                val_metrics.get("pixel_iou", 0.0),
                val_metrics.get("object_f1", 0.0),
            )

            # Figures
            _render_and_log_figures(
                epoch, preview_records, preview_tile_ids, acc, val_metrics,
                data["stats"], data["extra_channels"], out_dir,
            )

            # Early stopping + best-checkpoint tracking.
            is_best = es.update(epoch, val_metrics)
            smoothed = es.smoothed_value()
            if ckpt_mgr.update_best(smoothed) and ema is not None:
                # Save deployment checkpoint using EMA weights.
                with ema.swap_in(model):
                    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                trained_with = ckpt_mod.TrainedWith(
                    precision=precision.effective,
                    seed=int(cfg["seed"]),
                    config_sha=mlflow_utils.config_sha(cfg),
                )
                ckpt_mgr.save_deployment(
                    ema_state_dict=ema_state,
                    epoch=epoch,
                    best_metric=smoothed,
                    channel_names=["R", "G", "B"] + [c.name for c in data["extra_channels"]],
                    trained_with=trained_with,
                )

            # Resume snapshot (rotating).
            ckpt_mgr.save_resume(
                model=model,
                ema_state_dict=(ema.shadow.copy() if ema is not None else None),
                optimizer=optimizer,
                scheduler_state={"max_epochs": max_epochs},  # scheduler is stateless (pure function)
                scaler_state=(precision.scaler.state_dict() if precision.scaler is not None else None),
                epoch=epoch,
                early_stopping_state=es.state_dict(),
                rng_states={
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state().tolist(),
                },
            )

            if es.should_stop(epoch):
                break
    finally:
        # LR-range-test: dump (step, lr, loss) curve as a CSV artifact for analysis.
        if lr_history is not None and lr_history:
            lr_csv = out_dir / "lr_range_test.csv"
            with open(lr_csv, "w", encoding="utf-8") as f:
                f.write("global_step,lr,loss\n")
                for step_i, lr_i, loss_i in lr_history:
                    f.write(f"{step_i},{lr_i:.8e},{loss_i:.6f}\n")
            mlflow.log_artifact(str(lr_csv))
            logger.info("LR-range-test curve written to %s (%d steps)",
                        lr_csv, len(lr_history))

        # Exposure summary + run_summary.md.
        if exposure_counter:
            vals = np.array(list(exposure_counter.values()))
            mlflow.log_metrics({
                "exposure_max": float(vals.max()),
                "exposure_median": float(np.median(vals)),
                "exposure_p99": float(np.percentile(vals, 99)),
                "exposure_unique_tiles": float(vals.size),
            })

        duration = time.time() - t_start
        mlflow_utils.log_run_summary(
            cfg,
            final_metrics={k: v for k, v in es.state_dict().items() if isinstance(v, (int, float))},
            training_duration_s=duration,
            nan_events=nan_events,
            tmp_dir=out_dir,
        )
        mlflow.end_run()
        logger.info("Done in %.1fs", duration)

    return 0


def _resume_from(
    resume_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    precision: PrecisionSetup,
    es: es_mod.EarlyStopping,
    cfg: dict,
    *,
    freeze_epochs: int,
) -> tuple[int, ema_mod.EMAModel | None]:
    """Restore from a resume_latest-*.pth snapshot.

    Returns (next_epoch_to_run, ema_or_None). If the saved epoch is past the
    backbone-freeze boundary, an EMAModel is reconstructed from the saved
    `ema_state_dict` so post-resume validation continues using EMA weights
    rather than silently falling back to live weights (training.md §10.2 step 4).
    """
    sd = torch.load(resume_path, map_location="cpu", weights_only=False)
    if sd.get("checkpoint_type") != "resume":
        raise ValueError(f"Not a resume checkpoint: {resume_path}")
    model.load_state_dict(sd["live_state_dict"])
    optimizer.load_state_dict(sd["optimizer_state_dict"])
    if sd.get("scaler_state") and precision.scaler is not None:
        precision.scaler.load_state_dict(sd["scaler_state"])
    es.load_state_dict(sd["early_stopping_state"])
    rng = sd.get("rng_states", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.set_rng_state(torch.tensor(rng["torch"], dtype=torch.uint8))

    saved_epoch = int(sd["epoch"])
    next_epoch = saved_epoch + 1

    ema: ema_mod.EMAModel | None = None
    if saved_epoch > freeze_epochs and sd.get("ema_state_dict") is not None:
        # Reconstruct EMA so validation/best-checkpoint comparisons stay on
        # EMA weights instead of silently using live weights.
        ema = ema_mod.EMAModel(model, decay=float(cfg["ema"]["decay"]))
        ema.shadow = {k: v.detach().clone() for k, v in sd["ema_state_dict"].items()}
    return next_epoch, ema


if __name__ == "__main__":
    sys.exit(main())
