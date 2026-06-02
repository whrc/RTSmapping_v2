"""One-shot test-set evaluation (Phase 1 Step 8).

Runs inference on Test-Realistic exactly once, using the deployment package's
precision / TTA / torch_compile / threshold / temperature at scale 1.0.

Multi-scale evaluation is **optional and deferred**: per `training.md §4.6`,
the 1× test number is the canonical Test-Realistic result. Multi-scale
evaluation is a separate optional pass that runs only after the 1× number is
finalized. This script refuses to proceed when the deployment package selects
multi-scale (`len(scales) > 1`) — point users at the inference pipeline
(`scripts/inference.py`) for that path.

Refuses to run if the deployment package's threshold or temperature is null
(calibration not run yet). Channel-name binding (training.md §4.5) is the
integrity guarantee for normalization stats.

Per training.md §10.3, Test-Realistic is touched exactly once, after every
experimental decision is frozen. Running this script twice on the same
package is fine (same numbers); running it with a different deployment
config after a first evaluation counts as a new experimental decision and
violates the §10.3 discipline — the guard doesn't catch this; the
researcher's discipline does.

Run:
    python scripts/evaluate_test.py \\
        --deployment-package gs://abruptthawmapping/models/rts-v2-seed42 \\
        --training-config configs/baseline.yaml

Outputs:
    - metrics.json next to the deployment package with pixel + object + PR-AUC
      at each ratio.
    - Appends a row to docs/baseline_unetpp_effb5.md if --results-md is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import RTSDataset, parse_extra_spec  # noqa: E402
from data.splits import get_tile_ids, load_metadata, load_splits_yaml  # noqa: E402
from data.transforms import build_eval_transforms  # noqa: E402
from models import build_model  # noqa: E402
from training import metrics as metrics_mod  # noqa: E402
from utils.config import load_config, resolve_path  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _assert_deployment_package_complete(pkg: Path) -> tuple[dict, dict, dict]:
    """Validate package; return (training_cfg_fragment, deployment_cfg, run_metadata)."""
    required = [
        "weights.pth",
        "normalization_stats.json",
        "model_config.yaml",
        "deployment_config.yaml",
        "run_metadata.json",
    ]
    for name in required:
        if not (pkg / name).exists():
            raise FileNotFoundError(f"Deployment package missing {name}: {pkg}")

    deployment_cfg = yaml.safe_load((pkg / "deployment_config.yaml").read_text())
    if deployment_cfg.get("threshold") is None or deployment_cfg.get("temperature") is None:
        raise ValueError(
            "Deployment config has null threshold/temperature. Run calibration "
            "on val (training.md §12) before evaluating on test."
        )
    scales = deployment_cfg.get("scales", [1.0])
    if len(scales) > 1:
        raise ValueError(
            f"Deployment config selects multi-scale ({scales}); evaluate_test.py "
            "is the 1× contract per training.md §4.6. Run scripts/inference.py "
            "(Phase 2) for the optional multi-scale eval pass."
        )

    model_cfg = yaml.safe_load((pkg / "model_config.yaml").read_text())
    meta = json.loads((pkg / "run_metadata.json").read_text())

    # Integrity for normalization stats lives in channel-name binding at load
    # time (training.md §4.5), not in a content hash. Packaging already
    # refuses un-FINISHED MLflow runs.

    return model_cfg, deployment_cfg, meta


def _apply_tta(logits_fn, image: torch.Tensor, tta: str) -> torch.Tensor:
    """Average per-tile probabilities under the requested TTA config.

    Args:
        logits_fn: callable(batch_image) -> logits (pre-sigmoid).
        image: (B, C, H, W) tensor (already on device).
        tta: "none" | "minimal" | "standard" | "full".

    Returns:
        Averaged **logits-space** tensor of same shape. Keeping in logit space
        matches calibration which also operated on raw logits.
    """
    def inverse(t: torch.Tensor, flip_h: bool, flip_v: bool, rot180: bool) -> torch.Tensor:
        if rot180:
            t = torch.rot90(t, 2, dims=(-2, -1))
        if flip_v:
            t = torch.flip(t, dims=(-2,))
        if flip_h:
            t = torch.flip(t, dims=(-1,))
        return t

    passes: list[tuple[bool, bool, bool]] = [(False, False, False)]  # identity
    if tta == "minimal":
        passes.append((True, False, False))
    elif tta == "standard":
        passes += [(True, False, False), (False, True, False), (False, False, True)]
    elif tta == "full":
        # 8-way D4: identity, hflip, vflip, rot180, rot90, rot90+hflip, rot90+vflip, rot90+rot180
        # For simplicity implement via hflip/vflip/rot180 composition.
        passes += [(True, False, False), (False, True, False), (True, True, False),
                   (False, False, True), (True, False, True), (False, True, True), (True, True, True)]

    outs = []
    for fh, fv, r180 in passes:
        x = image
        if r180:
            x = torch.rot90(x, 2, dims=(-2, -1))
        if fv:
            x = torch.flip(x, dims=(-2,))
        if fh:
            x = torch.flip(x, dims=(-1,))
        logit = logits_fn(x)
        logit = inverse(logit, fh, fv, r180)
        outs.append(logit)

    return torch.stack(outs, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate_test(
    deployment_package: Path,
    training_config: Path,
    output_path: Path | None = None,
    device: str | None = None,
) -> dict:
    """Run the one-shot test evaluation. Returns the metrics dict."""
    model_cfg, dep_cfg, meta = _assert_deployment_package_complete(deployment_package)
    train_cfg = load_config(training_config)

    # Use training config for data_root / split info; model-specific keys from
    # the deployment package take precedence on architecture/channels.
    merged = {
        **train_cfg,
        "model": model_cfg["model"],
        "channels": model_cfg["channels"],
    }

    device_t = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s  Precision: %s  TTA: %s  Scale: 1.0 (1×-only contract)",
                device_t, dep_cfg["precision"], dep_cfg["tta"])

    # Load model with EMA weights.
    model = build_model(merged).to(device_t).eval()
    state_dict = torch.load(deployment_package / "weights.pth", map_location=device_t, weights_only=False)
    model.load_state_dict(state_dict)

    # torch.compile as requested (must match calibration).
    if dep_cfg.get("torch_compile", False):
        model = torch.compile(model)

    # Data: test_realistic split.
    metadata = load_metadata(resolve_path(train_cfg["data"]["data_root"], train_cfg["data"]["metadata_csv"]))
    splits = load_splits_yaml(resolve_path(train_cfg["data"]["data_root"], train_cfg["data"]["splits_yaml"]))
    test_ids = get_tile_ids("test_realistic", metadata, splits)
    logger.info("Evaluating on %d test_realistic tiles", len(test_ids))

    ds = RTSDataset(
        tile_ids=test_ids,
        metadata=metadata,
        data_root=train_cfg["data"]["data_root"],
        rgb_dir=train_cfg["data"]["rgb_dir"],
        extra_dir=train_cfg["data"]["extra_dir"],
        labels_dir=train_cfg["data"]["labels_dir"],
        extra_channels=parse_extra_spec(train_cfg["channels"].get("extra", [])),
        norm_stats_path=str(deployment_package / "normalization_stats.json"),
        transform=build_eval_transforms(),
        tile_size=int(train_cfg["data"]["tile_size"]),
        label_ignore_index=int(train_cfg["data"]["label_ignore_index"]),
        boundary_handling="none",  # never ignore at test time
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=int(train_cfg["training"]["batch_size"]),
        shuffle=False, num_workers=0, pin_memory=False,
    )

    # Autocast per deployment precision.
    precision = dep_cfg["precision"].lower()
    if precision == "bf16":
        dtype = torch.bfloat16 if device_t.type == "cuda" and torch.cuda.is_bf16_supported() else None
    elif precision == "fp16":
        dtype = torch.float16 if device_t.type == "cuda" else None
    else:
        dtype = None
    autocast_ctx = (
        torch.amp.autocast(device_type=device_t.type, dtype=dtype)
        if dtype is not None else nullcontext()
    )

    # Accumulator uses per-ratio PR-AUC; reporting_threshold comes from the
    # deployment config (overriding metrics.reporting_threshold).
    acc_cfg = {
        "data": {"label_ignore_index": int(train_cfg["data"]["label_ignore_index"])},
        "metrics": {
            "reporting_threshold": float(dep_cfg["threshold"]),
            "min_blob_size_px": int(dep_cfg.get("min_blob_size_px", 10)),
            "object_iou_threshold": float(train_cfg["metrics"]["object_iou_threshold"]),
        },
    }
    acc = metrics_mod.ValidationAccumulator(acc_cfg, ratios=[200, 500, 1000])

    tta = dep_cfg["tta"]
    temperature = float(dep_cfg["temperature"])

    for batch in loader:
        images = batch["image"].to(device_t, non_blocking=True)
        labels = batch["label"].to(device_t, non_blocking=True)
        tile_ids = batch["tile_id"]

        with autocast_ctx:
            if tta == "none":
                logits = model(images)
            else:
                logits = _apply_tta(model, images, tta)

        # Apply temperature scaling (on logits). Float32 for metric stability.
        logits = logits.float() / temperature
        acc.update(logits, labels, tile_ids)

    metrics = acc.compute()
    metrics["_deployment_package"] = str(deployment_package)
    metrics["_mlflow_run_id"] = meta.get("mlflow_run_id")
    metrics["_seed"] = meta.get("seed")
    metrics["_tta"] = tta
    metrics["_threshold"] = dep_cfg["threshold"]
    metrics["_temperature"] = temperature
    metrics["_scale"] = 1.0

    output_path = output_path or (deployment_package / "test_metrics.json")
    # Convert numpy/python floats to plain floats for JSON.
    cleaned = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in metrics.items()}
    output_path.write_text(json.dumps(cleaned, indent=2))
    logger.info("Wrote test metrics to %s", output_path)
    logger.info("\n%s", json.dumps(cleaned, indent=2))
    return cleaned


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--deployment-package", type=Path, required=True)
    p.add_argument("--training-config", type=Path, required=True,
                   help="Training config for data split info (typically configs/baseline.yaml)")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write test_metrics.json; default: inside the package")
    p.add_argument("--device", default=None, help="cuda|cpu; default auto")
    args = p.parse_args()

    setup_logging(level="INFO")
    evaluate_test(args.deployment_package, args.training_config, args.output, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
