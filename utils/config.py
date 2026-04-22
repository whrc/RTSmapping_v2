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
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write a config dict to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


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
