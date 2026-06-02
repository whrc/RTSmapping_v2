"""Unit tests for scripts/package_model._assert_calibration_complete.

Full end-to-end packaging requires an MLflow run, which is exercised via the
train-smoke test. Here we just verify the calibration-guard contract: a
deployment config with null threshold or temperature must be rejected
(plan Step 8 gate).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from package_model import _assert_calibration_complete  # noqa: E402


def test_null_threshold_rejected():
    cfg = {"threshold": None, "temperature": 1.2}
    with pytest.raises(ValueError, match="threshold"):
        _assert_calibration_complete(cfg)


def test_null_temperature_rejected():
    cfg = {"threshold": 0.5, "temperature": None}
    with pytest.raises(ValueError, match="temperature"):
        _assert_calibration_complete(cfg)


def test_both_null_rejected_together():
    cfg = {"threshold": None, "temperature": None}
    with pytest.raises(ValueError, match="threshold"):
        _assert_calibration_complete(cfg)


def test_both_set_accepted():
    cfg = {"threshold": 0.5, "temperature": 1.2}
    # Should not raise.
    _assert_calibration_complete(cfg)
