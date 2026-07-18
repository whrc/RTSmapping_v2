"""Unit tests for training.early_stopping.EarlyStopping."""

from __future__ import annotations

import pytest

from training.early_stopping import EarlyStopping


def _stopper(**overrides) -> EarlyStopping:
    kwargs = dict(
        metric_name="m",
        patience=3,
        min_delta=0.0,
        start_epoch=2,
        smoothing_window=3,
    )
    kwargs.update(overrides)
    return EarlyStopping(**kwargs)


def _feed(stopper, epoch_metric_pairs):
    """Drive the stopper through a sequence of (epoch, metric_value) pairs.

    Returns the final (smoothed, best_smoothed, no_improve, stopped) snapshot.
    """
    for ep, m in epoch_metric_pairs:
        stopper.update(ep, {"m": m})
        stopper.should_stop(ep)
    return (
        stopper.smoothed_value(),
        stopper.best_smoothed,
        stopper.no_improve_count,
        stopper.stopped,
    )


def test_monotone_increase_always_improving():
    s = _stopper()
    _feed(s, [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)])
    assert s.no_improve_count == 0
    assert s.stopped is False


def test_plateau_triggers_stop_after_patience():
    s = _stopper(patience=2, start_epoch=1)
    # Smoothing window = 3, so after three equal values smoothed plateaus.
    for ep, v in [(1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5), (5, 0.5)]:
        s.update(ep, {"m": v})
        stopped = s.should_stop(ep)
        if ep == 1:
            assert s.no_improve_count == 0  # first-ever gets the "best" trophy
        elif ep <= 3:
            # While smoothed is still rising (early fills), may remain improving.
            pass
    # Eventually the smoothed plateau exhausts patience.
    assert s.stopped is True


def test_start_epoch_gates_stopping_but_not_best_tracking():
    s = _stopper(patience=1, start_epoch=5)
    _feed(s, [(1, 0.9), (2, 0.0), (3, 0.0), (4, 0.0)])  # clear degradation
    # No stop allowed until epoch 5.
    assert s.stopped is False
    # Best still tracked from the start.
    assert s.best_smoothed >= 0.0
    # At epoch 5, with patience already exhausted, next failed update stops.
    s.update(5, {"m": 0.0})
    stopped = s.should_stop(5)
    assert stopped is True


def test_min_delta_ignores_noise():
    s = _stopper(patience=3, min_delta=0.01, start_epoch=1, smoothing_window=1)
    s.update(1, {"m": 0.5})
    # Tiny improvement below min_delta -> not a new best.
    s.update(2, {"m": 0.505})
    assert s.no_improve_count == 1
    # Larger improvement above min_delta -> resets counter.
    s.update(3, {"m": 0.6})
    assert s.no_improve_count == 0


def test_missing_metric_key_raises():
    s = _stopper()
    with pytest.raises(KeyError, match="metric_name"):
        s.update(1, {"something_else": 0.5})


def test_state_dict_restores_observation_state():
    """Resume carries observation state across — history, best, counters."""
    s1 = _stopper(patience=4, min_delta=0.001, start_epoch=7)
    _feed(s1, [(1, 0.1), (2, 0.2), (3, 0.15)])
    sd = s1.state_dict()

    s2 = _stopper(patience=4, min_delta=0.001, start_epoch=7)
    s2.load_state_dict(sd)
    assert s2.best_smoothed == s1.best_smoothed
    assert s2.best_epoch == s1.best_epoch
    assert s2.no_improve_count == s1.no_improve_count
    assert s2.stopped == s1.stopped
    assert list(s2._history) == list(s1._history)


def test_load_state_dict_keeps_config_hyperparameters():
    """Hyperparameters come from config on resume, never from the checkpoint.

    Regression: the v2.1 gate resume silently kept the checkpoint's start_epoch=101
    after the config was lowered to 65, burning ~15 extra epochs per seed.
    """
    s1 = _stopper(patience=8, min_delta=0.005, start_epoch=101)
    _feed(s1, [(1, 0.1), (2, 0.2), (3, 0.15)])
    sd = s1.state_dict()
    assert sd["start_epoch"] == 101  # checkpoint carries the stale value

    s2 = _stopper(patience=5, min_delta=0.001, start_epoch=65)  # new config
    s2.load_state_dict(sd)
    assert s2.start_epoch == 65
    assert s2.patience == 5
    assert s2.min_delta == 0.001
    # ...while observation state still came across
    assert s2.best_smoothed == s1.best_smoothed


def test_lowered_start_epoch_takes_effect_after_resume():
    """End-to-end: a resumed run stops on the new, lower start_epoch."""
    s1 = _stopper(patience=2, min_delta=0.0, start_epoch=100)
    _feed(s1, [(10, 0.9), (20, 0.5), (30, 0.5)])  # peak at 10, then flat
    assert not s1.stopped  # floored off by start_epoch=100

    s2 = _stopper(patience=2, min_delta=0.0, start_epoch=40)
    s2.load_state_dict(s1.state_dict())
    s2.update(40, {"m": 0.5})
    assert s2.should_stop(40)  # patience already exhausted, now past start_epoch
