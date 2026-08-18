"""Stall watchdog (utils/watchdog.py).

Guards the failure mode that motivated the module: a loop that hangs rather
than crashing holds its claim and reports no error, so nothing notices. These
assert it fires on a real stall, stays out of the way during progress, and can
be disabled outright.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.watchdog import start_stall_watchdog


def test_disabled_is_noop():
    assert start_stall_watchdog([0.0], 0, "x")() is None  # timeout<=0 -> no-op stop fn


def test_does_not_kill_while_progressing():
    """A short timeout must not fire while last_active keeps moving."""
    last = [time.time()]
    stop = start_stall_watchdog(last, 0.5, "x")
    for _ in range(8):          # ~0.8s of steady progress, each tick < timeout
        time.sleep(0.1)
        last[0] = time.time()
    stop()                      # still alive: os._exit never called
    assert True


def _run_stalled(exit_code_arg: str) -> int:
    """Drive a stalled watchdog in a subprocess so os._exit can't kill pytest."""
    import subprocess
    import textwrap
    code = textwrap.dedent(f"""
        import time
        from utils.watchdog import start_stall_watchdog
        start_stall_watchdog([time.time() - 100.0], 0.2, "stall"{exit_code_arg})
        time.sleep(10)  # watchdog should os._exit well before this returns
        print("NOT_REACHED")
    """)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=30,
                          cwd=str(Path(__file__).resolve().parent.parent))
    assert "NOT_REACHED" not in proc.stdout
    return proc.returncode


def test_exits_process_on_hard_stall():
    assert _run_stalled("") == 3          # default: what the inference supervisor expects


def test_exit_code_is_configurable():
    assert _run_stalled(", exit_code=7") == 7
