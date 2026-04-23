"""Early stopping on a smoothed validation metric.

Smoothing: moving average over the last `smoothing_window` validations
(default 3 — see configs/baseline.yaml training.early_stopping.smoothing_window).
This smoothed value is **also** the source of truth for best-checkpoint
selection (plan risk #6 — consistency between stopping and checkpointing).

Gating: `start_epoch` delays stopping until the curriculum reaches the
realistic prevalence (epoch 101 in baseline — matches 1:20 curriculum end).
Before that, validations still run and best-so-far checkpoints still update;
only stopping is suppressed (training.md §10.2).
"""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Patience-based early stopping on a smoothed higher-is-better metric.

    Args:
        metric_name: Key in the metrics dict to monitor.
        patience: Consecutive validations without smoothed improvement before stop.
        min_delta: Minimum smoothed-metric gain to count as an improvement.
        start_epoch: First epoch at which stopping decisions are allowed.
        smoothing_window: Moving-average window on raw metric values.
    """

    def __init__(
        self,
        metric_name: str,
        patience: int,
        min_delta: float,
        start_epoch: int,
        smoothing_window: int = 3,
    ):
        self.metric_name = metric_name
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.start_epoch = int(start_epoch)
        self.smoothing_window = int(smoothing_window)

        self._history: deque[float] = deque(maxlen=self.smoothing_window)
        self.best_smoothed: float = float("-inf")
        self.best_epoch: int = -1
        self.no_improve_count: int = 0
        self.stopped: bool = False

    def smoothed_value(self) -> float:
        """Current moving-average of observed raw metric values.

        Returns -inf before any observations — never "best" on an empty history.
        """
        if not self._history:
            return float("-inf")
        return sum(self._history) / len(self._history)

    def update(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Ingest one validation's metrics. Returns True iff this is a new best.

        Does NOT trigger stopping by itself — call `should_stop(epoch)` after.
        """
        if self.metric_name not in metrics:
            raise KeyError(
                f"EarlyStopping.metric_name={self.metric_name!r} not in metrics dict; "
                f"available: {sorted(metrics)}"
            )
        self._history.append(float(metrics[self.metric_name]))
        current = self.smoothed_value()
        is_best = current > self.best_smoothed + self.min_delta
        if is_best:
            self.best_smoothed = current
            self.best_epoch = epoch
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
        logger.info(
            "EarlyStopping epoch=%d smoothed=%.5f best=%.5f (epoch=%d) no_improve=%d%s",
            epoch, current, self.best_smoothed, self.best_epoch,
            self.no_improve_count, " [new best]" if is_best else "",
        )
        return is_best

    def should_stop(self, epoch: int) -> bool:
        """Return True when patience has been exhausted and epoch >= start_epoch."""
        if epoch < self.start_epoch:
            return False
        if self.no_improve_count >= self.patience:
            self.stopped = True
            logger.info(
                "Early stopping at epoch=%d (no improvement for %d validations; "
                "best smoothed=%.5f at epoch=%d)",
                epoch, self.no_improve_count, self.best_smoothed, self.best_epoch,
            )
            return True
        return False

    def state_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "start_epoch": self.start_epoch,
            "smoothing_window": self.smoothing_window,
            "history": list(self._history),
            "best_smoothed": self.best_smoothed,
            "best_epoch": self.best_epoch,
            "no_improve_count": self.no_improve_count,
            "stopped": self.stopped,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.metric_name = sd["metric_name"]
        self.patience = sd["patience"]
        self.min_delta = sd["min_delta"]
        self.start_epoch = sd["start_epoch"]
        self.smoothing_window = sd["smoothing_window"]
        self._history = deque(sd["history"], maxlen=self.smoothing_window)
        self.best_smoothed = sd["best_smoothed"]
        self.best_epoch = sd["best_epoch"]
        self.no_improve_count = sd["no_improve_count"]
        self.stopped = sd["stopped"]
