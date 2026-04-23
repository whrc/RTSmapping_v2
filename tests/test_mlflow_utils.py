"""Unit tests for pure-function utilities in training.mlflow_utils.

MLflow-side integration (setup_mlflow, log_artifact) is exercised via the
synthetic train-smoke test in Step 7a, using a local file-backend store.
"""

from __future__ import annotations

from training.mlflow_utils import _flatten_params, config_sha


def test_flatten_params_nested_dict():
    cfg = {"a": {"b": {"c": 1}}, "d": [1, 2, 3], "e": "x"}
    flat = _flatten_params(cfg)
    assert flat["a.b.c"] == "1"
    assert flat["d"] == "[1, 2, 3]"
    assert flat["e"] == "x"


def test_flatten_params_truncates_long_values():
    big = "x" * 1000
    flat = _flatten_params({"k": big})
    assert len(flat["k"]) == 500


def test_config_sha_deterministic_and_order_independent():
    a = {"b": 1, "a": 2, "nested": {"y": 3, "x": 4}}
    b = {"a": 2, "b": 1, "nested": {"x": 4, "y": 3}}
    assert config_sha(a) == config_sha(b)


def test_config_sha_changes_on_value_difference():
    a = {"a": 1}
    b = {"a": 2}
    assert config_sha(a) != config_sha(b)
