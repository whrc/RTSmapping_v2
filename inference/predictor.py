"""Deployment-package loading and probability prediction (inference.md §7, §8.2).

Order of operations per §7.3 (probability-space TTA fusion — intentionally
different from scripts/evaluate_test.py, whose logit-space averaging matches
the calibration definition):

    for each TTA pass: logits -> /temperature -> sigmoid -> inverse transform
    -> arithmetic mean over passes.

Phase 1 runs scales [1.0] only; multi-scale is gated by §6.4 and not built.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from data.normalization import load_stats, stats_to_arrays
from models import build_model
from utils.config import load_config

logger = logging.getLogger(__name__)

# (hflip, vflip, rot180) passes per TTA config — inference.md §7.1.
TTA_PASSES: dict[str, list[tuple[bool, bool, bool]]] = {
    "none": [(False, False, False)],
    "minimal": [(False, False, False), (True, False, False)],
    "standard": [(False, False, False), (True, False, False),
                 (False, True, False), (False, False, True)],
    "full": [(fh, fv, r) for fh in (False, True) for fv in (False, True)
             for r in (False, True)],
}


def load_deployment_package(package_dir: str | Path, device: torch.device) -> dict:
    """Load and validate a deployment package (inference.md §2.2, §8.2 step 1).

    Returns dict with: model (eval mode, on device), model_cfg, dep_cfg,
    stats, mean, std, n_channels.

    Raises on the §5.1 channel-name-binding mismatch.
    """
    pkg = str(package_dir).rstrip("/")
    model_cfg = load_config(f"{pkg}/model_config.yaml")
    dep_cfg = load_config(f"{pkg}/deployment_config.yaml")
    stats = load_stats(f"{pkg}/normalization_stats.json")

    # §5.1 channel-name binding — abort on mismatch.
    if stats["rgb"]["channel_names"] != ["R", "G", "B"]:
        raise ValueError(f"normalization_stats.json rgb channel_names "
                         f"{stats['rgb']['channel_names']} != ['R','G','B']")
    extra_spec = (model_cfg.get("channels") or {}).get("extra") or []
    if extra_spec:
        expected = [c["name"] for c in extra_spec]
        got = stats.get("extra", {}).get("channel_names")
        if got != expected:
            raise ValueError(f"EXTRA channel binding mismatch: stats {got} "
                             f"!= model_config {expected}")
    with_extra = bool(extra_spec)
    mean, std = stats_to_arrays(stats, with_extra=with_extra)

    if dep_cfg.get("temperature") is None or dep_cfg.get("threshold") is None:
        raise ValueError(f"{pkg}/deployment_config.yaml: threshold/temperature "
                         "is null — package was assembled without calibration")
    scales = dep_cfg.get("scales", [1.0])
    if scales != [1.0]:
        raise NotImplementedError(
            f"scales={scales}: multi-scale inference is gated by inference.md "
            "§6.4 and not implemented in Phase 1")

    model = build_model(model_cfg)
    weights_path = f"{pkg}/weights.pth"
    if weights_path.startswith("gs://"):
        import gcsfs
        with gcsfs.GCSFileSystem(token="google_default").open(weights_path[5:], "rb") as f:
            state = torch.load(f, map_location="cpu", weights_only=True)
    else:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    if dep_cfg.get("torch_compile", False):
        model = torch.compile(model)

    logger.info("Deployment package loaded: %s (tta=%s, T=%.4f, thr=%.4f, %s)",
                pkg, dep_cfg.get("tta"), dep_cfg["temperature"],
                dep_cfg["threshold"], dep_cfg.get("precision"))
    return {"model": model, "model_cfg": model_cfg, "dep_cfg": dep_cfg,
            "stats": stats, "mean": mean, "std": std,
            "n_channels": 3 + len(extra_spec)}


def assert_runtime_matches_package(run_cfg: dict, dep_cfg: dict) -> None:
    """Abort if runtime config disagrees with the package's calibration-bound
    values (inference.md §14 'Calibration-deployment mismatch')."""
    for key in ("precision", "tta", "torch_compile", "scales", "temperature", "threshold"):
        # null in the runtime config means "defer to the package" (the repo's
        # deployment.yaml carries null threshold/temperature until calibration).
        if run_cfg.get(key) is not None and run_cfg[key] != dep_cfg.get(key):
            raise ValueError(
                f"Runtime config {key}={run_cfg[key]!r} != deployment package "
                f"{dep_cfg.get(key)!r}; calibration would be invalid. Fix the "
                "config or repackage with re-run calibration.")


def _autocast(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    images: torch.Tensor,
    temperature: float,
    tta: str = "none",
    precision: str = "fp32",
) -> torch.Tensor:
    """Probability maps for a batch per §7.3.

    Args:
        images: (B, C, H, W) normalized tensor on the model's device.

    Returns:
        (B, H, W) float32 probabilities in [0, 1].
    """
    if tta not in TTA_PASSES:
        raise ValueError(f"Unknown tta config {tta!r}; one of {list(TTA_PASSES)}")
    device = images.device
    acc: torch.Tensor | None = None
    for fh, fv, r180 in TTA_PASSES[tta]:
        x = images
        if r180:
            x = torch.rot90(x, 2, dims=(-2, -1))
        if fv:
            x = torch.flip(x, dims=(-2,))
        if fh:
            x = torch.flip(x, dims=(-1,))
        with _autocast(precision, device):
            logits = model(x)
        # Temperature on logits, then sigmoid — §7.3; float32 for stability.
        probs = torch.sigmoid(logits.float() / temperature)
        if fh:
            probs = torch.flip(probs, dims=(-1,))
        if fv:
            probs = torch.flip(probs, dims=(-2,))
        if r180:
            probs = torch.rot90(probs, 2, dims=(-2, -1))
        acc = probs if acc is None else acc + probs
    return (acc / len(TTA_PASSES[tta])).squeeze(1)
