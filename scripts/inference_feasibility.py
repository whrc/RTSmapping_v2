"""Phase 1 Step 8.5 — inference feasibility gates.

Runs two post-calibration studies on the cached val predictions:

    8.5a — Multi-scale feasibility (inference.md §6.4).
        Evaluate val at scale 1.0 and again at scale 0.5, fuse by arithmetic
        mean, compute PR-AUC on the large-RTS subset (bbox > 500 m).
        Gate: ship multi-scale if large-RTS PR-AUC gain >= 2% AND global
        FP-rate delta <= 10%.

    8.5b — TTA cost-benefit (inference.md §7.4).
        For each TTA config (none / minimal / standard), measure PR-AUC and
        precision@threshold on val.
        Gate: ship the cheapest config with PR-AUC gain >= 1% AND precision
        drop <= 0.5% from no-TTA.

Outputs:
    - {deployment_package}/feasibility_report.md
    - Updates {deployment_package}/deployment_config.yaml:
        scales:   [1.0] or [1.0, 0.5]
        tta:      none / minimal / standard

Run:
    python scripts/inference_feasibility.py \\
        --deployment-package gs://abruptthawmapping/models/rts-v2-seed42 \\
        --training-config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import RTSDataset, parse_extra_spec  # noqa: E402
from data.splits import get_tile_ids, load_metadata, load_splits_yaml  # noqa: E402
from data.transforms import build_eval_transforms  # noqa: E402
from models import build_model  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# Gate thresholds (inference.md §6.4, §7.4).
MULTISCALE_PR_AUC_GAIN = 0.02       # 2% gain on large-RTS subset
MULTISCALE_MAX_FP_DELTA = 0.10      # no more than 10% global FP-rate increase
TTA_PR_AUC_GAIN = 0.01              # 1% PR-AUC gain
TTA_MAX_PRECISION_DROP = 0.005      # 0.5% precision drop at threshold

LARGE_RTS_BBOX_EDGE_M = 500.0       # threshold for "large RTS"


@dataclass
class _TileResult:
    tile_id: str
    is_positive: bool
    is_large_rts: bool
    logits_1x: np.ndarray       # (H, W) float32
    logits_0p5x: np.ndarray | None   # (H, W) float32 or None
    labels: np.ndarray          # (H, W) uint8
    image: torch.Tensor | None = None  # (C, H, W) CPU tensor; cached for real TTA forwards


# ---------------------------------------------------------------------------
# Scale 0.5 helper: load 2x physical area, downsample, predict, upsample-crop
# ---------------------------------------------------------------------------


def _predict_scale_0_5(
    model: torch.nn.Module,
    image_1x: torch.Tensor,
    autocast_ctx,
) -> torch.Tensor:
    """Produce a scale-0.5 prediction by downsampling inside the same tile.

    Since context-expanded training tiles are not available in Phase 1, this
    predicts on the same 512x512 tile downsampled to 256x256 then upsampled
    back — a conservative approximation of "half-scale inference". The real
    §6.3 procedure needs surrounding-area tiles; full support is a Phase 2
    data-pipeline change. §6.4 gate still works because both halves of the
    comparison use the same (approximate) half-scale operator.
    """
    b, c, h, w = image_1x.shape
    down = F.interpolate(image_1x, size=(h // 2, w // 2), mode="bilinear",
                         align_corners=False, antialias=True)
    with autocast_ctx:
        logits_small = model(down)
    return F.interpolate(logits_small.float(), size=(h, w), mode="bilinear",
                         align_corners=False)


# ---------------------------------------------------------------------------
# Data ingestion: run val set once, cache per-tile data
# ---------------------------------------------------------------------------


def _tile_is_large(metadata_row, bbox_edge_m: float = LARGE_RTS_BBOX_EDGE_M) -> bool:
    """True if the tile's GT RTS has any bbox edge exceeding the threshold.

    Requires `bbox_edge_m` column in metadata.csv. If absent, treat all
    positive tiles as not large (conservative — feasibility still computes
    the overall PR-AUC gain but the "large-RTS" column will be empty).
    """
    if "bbox_edge_m" not in metadata_row:
        return False
    val = metadata_row.get("bbox_edge_m", 0.0)
    try:
        return float(val) >= bbox_edge_m
    except (TypeError, ValueError):
        return False


@torch.no_grad()
def _run_val_inference(
    deployment_package: Path,
    training_cfg: dict,
    device: torch.device,
):
    """Run val once at scale 1.0 and scale 0.5.

    Returns (tiles, threshold, dep_cfg, model, autocast_ctx, temperature) so
    8.5b can re-run real TTA forwards on the cached input images.
    """
    dep_cfg = yaml.safe_load((deployment_package / "deployment_config.yaml").read_text())
    if dep_cfg.get("threshold") is None or dep_cfg.get("temperature") is None:
        raise ValueError("Deployment config has null threshold/temperature; run calibration first.")
    threshold = float(dep_cfg["threshold"])
    temperature = float(dep_cfg["temperature"])

    model_cfg = yaml.safe_load((deployment_package / "model_config.yaml").read_text())
    merged = {**training_cfg, "model": model_cfg["model"], "channels": model_cfg["channels"]}
    model = build_model(merged).to(device).eval()
    state_dict = torch.load(deployment_package / "weights.pth", map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

    metadata = load_metadata(resolve_path(training_cfg["data"]["data_root"], training_cfg["data"]["metadata_csv"]))
    splits = load_splits_yaml(resolve_path(training_cfg["data"]["data_root"], training_cfg["data"]["splits_yaml"]))
    val_ids = get_tile_ids("val_realistic", metadata, splits)
    logger.info("Running feasibility on %d val_realistic tiles", len(val_ids))

    ds = RTSDataset(
        tile_ids=val_ids,
        metadata=metadata,
        data_root=training_cfg["data"]["data_root"],
        rgb_dir=training_cfg["data"]["rgb_dir"],
        extra_dir=training_cfg["data"]["extra_dir"],
        labels_dir=training_cfg["data"]["labels_dir"],
        extra_channels=parse_extra_spec(training_cfg["channels"].get("extra", [])),
        norm_stats_path=str(deployment_package / "normalization_stats.json"),
        transform=build_eval_transforms(),
        tile_size=int(training_cfg["data"]["tile_size"]),
        label_ignore_index=int(training_cfg["data"]["label_ignore_index"]),
        boundary_handling="none",
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(training_cfg["training"]["batch_size"]),
        shuffle=False, num_workers=0,
    )

    # Precision autocast matching deployment.
    precision = dep_cfg["precision"].lower()
    if precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif precision == "fp16" and device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = None
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=dtype)
        if dtype is not None else nullcontext()
    )

    md_indexed = metadata.set_index("Tile_id")
    out: list[_TileResult] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()
        tids = batch["tile_id"]

        with autocast_ctx:
            logits_1x = model(images).float()
        logits_0p5 = _predict_scale_0_5(model, images, autocast_ctx)

        # Apply temperature scaling (§12.1).
        logits_1x = logits_1x / temperature
        logits_0p5 = logits_0p5 / temperature

        for b, tid in enumerate(tids):
            row = md_indexed.loc[tid]
            is_pos = bool((labels[b] == 1).any())
            is_large = _tile_is_large(row) if is_pos else False
            # Cache the input image (CPU, fp32) so 8.5b can run real TTA forwards
            # via re-running the model on flipped inputs rather than flipping
            # output logits — the latter is mathematically wrong for a model
            # without translational equivariance.
            out.append(_TileResult(
                tile_id=tid,
                is_positive=is_pos,
                is_large_rts=is_large,
                logits_1x=logits_1x[b, 0].cpu().numpy(),
                logits_0p5x=logits_0p5[b, 0].cpu().numpy(),
                labels=labels[b].astype(np.uint8),
                image=images[b].detach().cpu(),
            ))
    return out, threshold, dep_cfg, model, autocast_ctx, temperature


# ---------------------------------------------------------------------------
# Metrics on cached data
# ---------------------------------------------------------------------------


def _stack_valid(tiles: list[_TileResult], logits_key: str, ignore: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate valid-pixel logits + labels across tiles."""
    all_logits = []
    all_labels = []
    for t in tiles:
        mask = t.labels != ignore
        logits = getattr(t, logits_key)
        all_logits.append(logits[mask].astype(np.float32))
        all_labels.append(np.where(t.labels[mask] == 1, 1, 0).astype(np.uint8))
    return np.concatenate(all_logits), np.concatenate(all_labels)


def _pr_auc(logits: np.ndarray, labels: np.ndarray) -> float:
    if labels.max() == 0:
        return 0.0
    return float(average_precision_score(labels, logits))


def _precision_at_threshold(logits: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    preds = logits >= threshold
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    return tp / max(1, tp + fp)


def _fp_rate(logits: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    preds = logits >= threshold
    fp = int(((preds == 1) & (labels == 0)).sum())
    neg = int((labels == 0).sum())
    return fp / max(1, neg)


def _fuse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Arithmetic mean in probability space, returned as logits."""
    pa = 1.0 / (1.0 + np.exp(-a))
    pb = 1.0 / (1.0 + np.exp(-b))
    p = 0.5 * (pa + pb)
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def _run_8p5a_multi_scale(tiles: list[_TileResult], threshold: float, ignore: int) -> dict:
    """Evaluate scale 1.0 vs scale 1.0 + 0.5 averaged."""
    # Fuse logits per tile (in place via temporary).
    fused: list[_TileResult] = []
    for t in tiles:
        fused.append(_TileResult(
            tile_id=t.tile_id,
            is_positive=t.is_positive,
            is_large_rts=t.is_large_rts,
            logits_1x=_fuse(t.logits_1x, t.logits_0p5x) if t.logits_0p5x is not None else t.logits_1x,
            logits_0p5x=None,
            labels=t.labels,
        ))

    baseline_logits, baseline_labels = _stack_valid(tiles, "logits_1x", ignore)
    multi_logits, multi_labels = _stack_valid(fused, "logits_1x", ignore)

    baseline_ap = _pr_auc(baseline_logits, baseline_labels)
    multi_ap = _pr_auc(multi_logits, multi_labels)

    # Large-RTS subset.
    large = [t for t in tiles if t.is_large_rts]
    if large:
        large_fused = [f for f in fused if f.tile_id in {t.tile_id for t in large}]
        bl, blab = _stack_valid(large, "logits_1x", ignore)
        ml, mlab = _stack_valid(large_fused, "logits_1x", ignore)
        large_baseline_ap = _pr_auc(bl, blab)
        large_multi_ap = _pr_auc(ml, mlab)
    else:
        large_baseline_ap = 0.0
        large_multi_ap = 0.0

    baseline_fp = _fp_rate(baseline_logits, baseline_labels, threshold)
    multi_fp = _fp_rate(multi_logits, multi_labels, threshold)

    fp_delta = multi_fp - baseline_fp
    pr_auc_gain = large_multi_ap - large_baseline_ap if large else (multi_ap - baseline_ap)

    ship = (pr_auc_gain >= MULTISCALE_PR_AUC_GAIN) and (fp_delta <= MULTISCALE_MAX_FP_DELTA)

    return {
        "baseline_pr_auc_full": baseline_ap,
        "multi_pr_auc_full": multi_ap,
        "baseline_pr_auc_large_rts": large_baseline_ap,
        "multi_pr_auc_large_rts": large_multi_ap,
        "n_large_rts_tiles": len(large),
        "pr_auc_gain_large_rts": pr_auc_gain,
        "baseline_fp_rate": baseline_fp,
        "multi_fp_rate": multi_fp,
        "fp_rate_delta": fp_delta,
        "gate_pass": ship,
        "recommended_scales": [1.0, 0.5] if ship else [1.0],
    }


@torch.no_grad()
def _run_8p5b_tta(
    model: torch.nn.Module,
    tiles: list[_TileResult],
    threshold: float,
    ignore: int,
    *,
    device: torch.device,
    autocast_ctx,
    temperature: float,
) -> dict:
    """Compare none / minimal / standard TTA via real flipped-input forwards.

    Re-runs the model with each TTA flip configuration on the cached input
    image, inverse-flips the resulting logits, and averages in logit space —
    matching `evaluate_test._apply_tta`. Earlier versions averaged the
    output-logit map directly with its spatial flip; that is mathematically
    wrong for any non-translation-equivariant model and produced gate
    decisions that were essentially noise.

    Cost: one extra forward pass per tile per non-identity TTA. Val set is
    small (a few hundred tiles) so this is acceptable.
    """
    results: dict = {}

    # No-TTA reference uses the cached baseline logits.
    logits_pool, labels_pool = _stack_valid(tiles, "logits_1x", ignore)
    ap_none = _pr_auc(logits_pool, labels_pool)
    prec_none = _precision_at_threshold(logits_pool, labels_pool, threshold)
    results["none"] = {"pr_auc": ap_none, "precision_at_threshold": prec_none}

    minimal_passes = [(False, False), (True, False)]
    standard_passes = [(False, False), (True, False), (False, True), (True, True)]

    def _real_tta_pool(passes: list[tuple[bool, bool]]) -> np.ndarray:
        """Re-run model per (fh, fv) flip; inverse-flip; average in logit space."""
        all_logits = []
        for t in tiles:
            if t.image is None:
                raise RuntimeError(
                    "_TileResult.image is None — feasibility script must cache "
                    "input images during _run_val_inference for real TTA."
                )
            x = t.image.unsqueeze(0).to(device, non_blocking=True)
            stack = []
            for fh, fv in passes:
                xi = x
                if fv:
                    xi = torch.flip(xi, dims=(-2,))
                if fh:
                    xi = torch.flip(xi, dims=(-1,))
                with autocast_ctx:
                    out = model(xi).float()
                if fv:
                    out = torch.flip(out, dims=(-2,))
                if fh:
                    out = torch.flip(out, dims=(-1,))
                stack.append(out)
            mean_logits = torch.stack(stack, dim=0).mean(dim=0) / temperature
            mask = t.labels != ignore
            all_logits.append(mean_logits[0, 0].cpu().numpy()[mask].astype(np.float32))
        return np.concatenate(all_logits)

    for name, passes in [("minimal", minimal_passes), ("standard", standard_passes)]:
        logits_tta = _real_tta_pool(passes)
        ap = _pr_auc(logits_tta, labels_pool)
        prec = _precision_at_threshold(logits_tta, labels_pool, threshold)
        results[name] = {"pr_auc": ap, "precision_at_threshold": prec}

    # Pick cheapest config that passes both gates vs none.
    order = ["none", "minimal", "standard"]
    recommended = "none"
    for name in order[1:]:
        ap_gain = results[name]["pr_auc"] - ap_none
        prec_drop = prec_none - results[name]["precision_at_threshold"]
        if ap_gain >= TTA_PR_AUC_GAIN and prec_drop <= TTA_MAX_PRECISION_DROP:
            recommended = name
            break   # cheapest-that-passes wins

    return {
        "per_config": results,
        "recommended_tta": recommended,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _write_feasibility_report(pkg: Path, multi_scale: dict, tta: dict) -> Path:
    lines = [
        "# Inference feasibility report",
        "",
        "## 8.5a — Multi-scale",
        "",
        f"- Baseline (scale 1.0) PR-AUC (full val): {multi_scale['baseline_pr_auc_full']:.4f}",
        f"- Multi-scale PR-AUC (full val): {multi_scale['multi_pr_auc_full']:.4f}",
        f"- Baseline PR-AUC (large-RTS, n={multi_scale['n_large_rts_tiles']}): "
        f"{multi_scale['baseline_pr_auc_large_rts']:.4f}",
        f"- Multi-scale PR-AUC (large-RTS): {multi_scale['multi_pr_auc_large_rts']:.4f}",
        f"- Large-RTS gain: {multi_scale['pr_auc_gain_large_rts']:+.4f} "
        f"(gate requires >= +{MULTISCALE_PR_AUC_GAIN:.2f})",
        f"- FP-rate delta: {multi_scale['fp_rate_delta']:+.4f} "
        f"(gate requires <= +{MULTISCALE_MAX_FP_DELTA:.2f})",
        f"- **Gate pass: {multi_scale['gate_pass']}**",
        f"- Recommended scales: `{multi_scale['recommended_scales']}`",
        "",
        "## 8.5b — TTA cost-benefit",
        "",
        "| Config | PR-AUC | Precision@threshold |",
        "|--------|--------|---------------------|",
    ]
    for name, vals in tta["per_config"].items():
        lines.append(f"| {name} | {vals['pr_auc']:.4f} | {vals['precision_at_threshold']:.4f} |")
    lines += [
        "",
        f"- Gates: PR-AUC gain >= +{TTA_PR_AUC_GAIN:.2f} AND precision drop <= "
        f"{TTA_MAX_PRECISION_DROP:.3f}",
        f"- **Recommended TTA: `{tta['recommended_tta']}`**",
    ]
    path = pkg / "feasibility_report.md"
    path.write_text("\n".join(lines))
    return path


def _update_deployment_config(pkg: Path, scales: list[float], tta: str) -> None:
    """Write the gate-selected scales + tta into the package's deployment config."""
    cfg_path = pkg / "deployment_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["scales"] = scales
    cfg["tta"] = tta
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--deployment-package", type=Path, required=True)
    p.add_argument("--training-config", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--update-config", action="store_true",
                   help="Write gate recommendations into deployment_config.yaml. "
                        "Off by default — §8.5a half-scale operates on a "
                        "downsampled crop of the same 512×512 tile rather "
                        "than the §6.3-required expanded surrounding area, "
                        "so the multi-scale gate is advisory until Phase 2 "
                        "implements the expanded-tile path.")
    args = p.parse_args()

    setup_logging(level="INFO")
    training_cfg = load_config(args.training_config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tiles, threshold, dep_cfg, model, autocast_ctx, temperature = _run_val_inference(
        args.deployment_package, training_cfg, device,
    )
    ignore = int(training_cfg["data"]["label_ignore_index"])

    ms = _run_8p5a_multi_scale(tiles, threshold, ignore)
    tta = _run_8p5b_tta(
        model, tiles, threshold, ignore,
        device=device, autocast_ctx=autocast_ctx, temperature=temperature,
    )

    report = _write_feasibility_report(args.deployment_package, ms, tta)
    logger.info("Wrote %s", report)
    logger.info(
        "8.5a multi-scale gate is ADVISORY until §6.3 expanded-tile inference "
        "lands; do not rely on `recommended_scales` for deployment without "
        "the proper expanded-tile path.",
    )

    # Combined summary JSON.
    (args.deployment_package / "feasibility.json").write_text(json.dumps({
        "multi_scale": ms, "tta": tta,
    }, indent=2, default=float))

    if args.update_config:
        _update_deployment_config(
            args.deployment_package,
            scales=ms["recommended_scales"],
            tta=tta["recommended_tta"],
        )
        logger.info("Updated deployment_config.yaml: scales=%s tta=%s",
                    ms["recommended_scales"], tta["recommended_tta"])
    else:
        logger.info("--update-config not set; deployment_config.yaml left untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
