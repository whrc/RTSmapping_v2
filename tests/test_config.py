"""Tests for utils/config.py — base-config inheritance and merge semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import _deep_merge, load_config


def _write(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_load_config_without_base_unchanged(tmp_path):
    p = _write(tmp_path / "a.yaml", {"seed": 42, "training": {"lr": 0.001}})
    assert load_config(p) == {"seed": 42, "training": {"lr": 0.001}}


def test_base_merge_nested_override(tmp_path):
    _write(tmp_path / "baseline.yaml",
           {"seed": 42, "training": {"lr": 0.001, "epochs": 300},
            "mlflow": {"run_name": "baseline", "experiment_name": "rts"}})
    p = _write(tmp_path / "exp.yaml",
               {"base": "baseline.yaml", "seed": 44,
                "mlflow": {"run_name": "exp_seed44"}})
    cfg = load_config(p)
    assert cfg["seed"] == 44                              # overridden
    assert cfg["training"] == {"lr": 0.001, "epochs": 300}  # inherited
    assert cfg["mlflow"]["run_name"] == "exp_seed44"      # nested override
    assert cfg["mlflow"]["experiment_name"] == "rts"      # nested sibling kept
    assert "base" not in cfg                              # key consumed


def test_base_merge_lists_replace_not_concat(tmp_path):
    _write(tmp_path / "baseline.yaml", {"metrics": {"ratios": [5, 10, 20]}})
    p = _write(tmp_path / "exp.yaml",
               {"base": "baseline.yaml", "metrics": {"ratios": [5]}})
    assert load_config(p)["metrics"]["ratios"] == [5]


def test_missing_base_raises(tmp_path):
    p = _write(tmp_path / "exp.yaml", {"base": "nope.yaml", "seed": 1})
    with pytest.raises(FileNotFoundError, match="Base config not found"):
        load_config(p)


def test_chained_base_rejected(tmp_path):
    _write(tmp_path / "root.yaml", {"seed": 1})
    _write(tmp_path / "mid.yaml", {"base": "root.yaml", "seed": 2})
    p = _write(tmp_path / "leaf.yaml", {"base": "mid.yaml", "seed": 3})
    with pytest.raises(ValueError, match="Chained bases"):
        load_config(p)


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}
    merged = _deep_merge(base, override)
    assert merged == {"a": {"b": 1, "c": 2}}
    assert base == {"a": {"b": 1}} and override == {"a": {"c": 2}}
