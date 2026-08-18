"""Process-level stall watchdog.

A long-running loop that wedges — a DataLoader worker deadlocked on a fork, an
HTTP call with no timeout on a half-open socket — is worse than one that
crashes: it holds its claim, keeps its heartbeat fresh, and reports no error,
so nothing notices until someone looks. This turns a silent hang into a loud
non-zero exit that a supervisor can restart.

The caller keeps a one-element list holding the epoch seconds of its last real
progress and bumps it each unit of work; the watchdog thread ``os._exit``s the
process when that timestamp goes stale.

Used by ``inference/runner.py`` (per-batch tile progress) and
``planetscope-download/order_basemaps.py`` (per-order progress).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


def start_stall_watchdog(last_active: list[float], timeout_s: float,
                         label: str, exit_code: int = 3) -> Callable[[], None]:
    """Kill this process if ``last_active[0]`` goes ``timeout_s`` seconds stale.

    Args:
        last_active: One-element list holding ``time.time()`` of the last unit
            of work completed. The caller bumps ``last_active[0]`` as it goes.
        timeout_s: Seconds of no progress before exiting. ``<= 0`` disables the
            watchdog and returns a no-op stop function (tests, single-shot CLI).
        label: Identifier for the log line — the shard id, the year, whatever
            makes the message locatable.
        exit_code: Status to exit with; the default 3 is what the inference
            supervisor treats as "restartable stall".

    Returns:
        A function that stops the watchdog thread.

    Note:
        Uses ``os._exit`` deliberately: a wedged process cannot be trusted to
        unwind cleanly, and ``sys.exit`` from a daemon thread would not reach
        the blocked main thread at all.
    """
    if not timeout_s or timeout_s <= 0:
        return lambda: None
    stop = threading.Event()

    def _watch() -> None:
        while not stop.wait(min(30.0, timeout_s / 4)):
            idle = time.time() - last_active[0]
            if idle > timeout_s:
                logger.critical(
                    "STALL: no progress for %.0fs (%s) — exiting %d for supervised "
                    "restart", idle, label, exit_code)
                os._exit(exit_code)

    threading.Thread(target=_watch, daemon=True).start()
    return stop.set
