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


def test_state_dict_roundtrip():
    s1 = _stopper(patience=4, min_delta=0.001, start_epoch=7)
    _feed(s1, [(1, 0.1), (2, 0.2), (3, 0.15)])
    sd = s1.state_dict()

    s2 = _stopper()  # different initial params
    s2.load_state_dict(sd)
    assert s2.metric_name == "m"
    assert s2.patience == 4
    assert s2.min_delta == 0.001
    assert s2.start_epoch == 7
    assert s2.best_smoothed == s1.best_smoothed
    assert list(s2._history) == list(s1._history)
