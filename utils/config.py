"""YAML config loading for the RTS pipeline."""

from __future__ import annotations

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
