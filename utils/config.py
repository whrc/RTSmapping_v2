"""YAML config loading for the RTS pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a nested dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If the file is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg).__name__}: {path}")

    # Optional single-level inheritance: `base: baseline.yaml` (path relative
    # to this file) pulls in the base config, with this file's keys deep-merged
    # on top. Lets experiment configs hold only their 2-10 line delta instead
    # of a full ~169-line copy (the v2.0 copy-paste configs drifted exactly
    # this way). One level only: a base may not itself declare a base.
    base_ref = cfg.pop("base", None)
    if base_ref is not None:
        base_path = (path.parent / base_ref).resolve()
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path} (from {path})")
        with base_path.open("r") as f:
            base = yaml.safe_load(f)
        if not isinstance(base, dict):
            raise ValueError(f"Base config root must be a mapping: {base_path}")
        if "base" in base:
            raise ValueError(f"Chained bases are not supported: {base_path} "
                             f"declares its own 'base'")
        cfg = _deep_merge(base, cfg)
    return cfg


# Keys allowed at the top level of a TRAINING config (the schema scripts/train.py reads).
# Anything else is almost always a mis-nested override that train.py silently ignores —
# the classic being `early_stopping:` at the top level instead of under `training:`
# (train.py reads cfg["training"]["early_stopping"]). Derived from base_v2_fast/phase0c.
# NOTE: inference/deploy/preview/pretrain configs have their own schemas and must NOT be
# passed through validate_training_config — only scripts/train.py calls it.
TRAINING_TOP_LEVEL_KEYS = frozenset({
    "seed", "deterministic", "data", "channels", "splits", "normalization",
    "sampling", "augmentation", "loss", "training", "optimizer", "lr_schedule",
    "ema", "model", "metrics", "mlflow", "base", "_config_path",
})

# Known mis-nestings: top-level key -> where it actually belongs (for a targeted hint).
_MISPLACED_TRAINING_KEYS = {
    "early_stopping": "training.early_stopping",
    "max_epochs": "training.max_epochs",
    "batch_size": "training.batch_size",
    "val_frequency": "training.val_frequency",
    "num_workers": "training.num_workers",
    "precision": "training.precision",
}


def validate_training_config(cfg: dict[str, Any], path: str | Path | None = None) -> None:
    """Reject a training config carrying unknown / mis-nested top-level keys.

    scripts/train.py reads a fixed schema (see ``TRAINING_TOP_LEVEL_KEYS``); any other
    top-level key is silently dropped at train time. The recurring offender is
    ``early_stopping:`` placed at the top level instead of under ``training:`` — the
    override is ignored, the run inherits the base schedule, and it quietly burns
    GPU-hours in the overfit tail. Fail loudly at load time instead of at ep300.

    Args:
        cfg: The merged training config (post ``load_config`` + CLI overrides).
        path: Optional source path, for a clearer error message.

    Raises:
        ValueError: Listing every offending top-level key, with a fix hint where known.
    """
    stray = [k for k in cfg if k not in TRAINING_TOP_LEVEL_KEYS]
    if not stray:
        return
    where = f" in {path}" if path else ""
    lines = [f"Unknown top-level key(s){where}: {sorted(stray)}",
             "scripts/train.py silently ignores top-level keys outside its schema, so this "
             "override would be a no-op. Fix the nesting:"]
    for k in sorted(stray):
        dest = _MISPLACED_TRAINING_KEYS.get(k)
        lines.append(f"  - '{k}' -> move under '{dest}'" if dest
                     else f"  - '{k}' is not a recognized training-config key")
    raise ValueError("\n".join(lines))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base` (override wins; lists replace)."""
    merged = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write a config dict to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def resolve_path(data_root: str, subpath: str) -> str:
    """Join a `data_root` (file path or `gs://` URI) with a subpath.

    Used by entry-point scripts to compose paths relative to the dataset root
    without duplicating the rstrip-and-slash idiom.
    """
    return f"{data_root.rstrip('/')}/{subpath}"


def vectorize_min_blob_px(dep_cfg: dict[str, Any], default: int = 0) -> int:
    """Read the LEGACY vectorization-stage pixel floor from a deployment config.

    This is **not** the shipped product's minimum mapping unit. The delivered
    South inventory is cut with the geodesic MMU (``vectorize_region.py
    --min-area-m2``, in m²); this pixel count only ever produced the superseded
    `south_rts.gpkg`. See `post-inference/south_products.md` §"Size parameters".

    Reads ``vectorize_min_blob_px``, falling back to the pre-2026-08-12 key name
    ``min_blob_size_px`` — still present in the deployment packages already
    written to GCS — with a warning.

    Args:
        dep_cfg: A parsed deployment config (``deployment.yaml`` or a package's
            ``deployment_config.yaml``).
        default: Value to return when neither key is present.

    Returns:
        The pixel floor, or ``default`` if unset.
    """
    if "vectorize_min_blob_px" in dep_cfg:
        return int(dep_cfg["vectorize_min_blob_px"])
    if "min_blob_size_px" in dep_cfg:
        logging.getLogger(__name__).warning(
            "deployment config uses the legacy key 'min_blob_size_px' (%s); it was "
            "renamed to 'vectorize_min_blob_px' on 2026-08-12 to separate it from the "
            "eval-stage 'metrics.min_blob_size_px' and the shipped geodesic "
            "--min-area-m2 MMU. Reading it anyway.", dep_cfg["min_blob_size_px"])
        return int(dep_cfg["min_blob_size_px"])
    return default


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Fetch a nested key like 'data.paths.rgb' from cfg; raise KeyError if missing.

    Prefer this over cfg['data']['paths']['rgb'] so missing keys give a useful message.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing required config key: {dotted_key}")
        node = node[part]
    return node
