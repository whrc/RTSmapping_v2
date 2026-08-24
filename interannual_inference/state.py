"""Per-year run state: which stages have run, and what they produced.

One JSON file per year under ``<work>/state/<year>.json``, mirrored best-effort to
GCS so the record survives the VM. The file is the campaign's single source of
truth for *progress*; the artifacts themselves (quad index, tile list, shard queue,
probability COGs) remain the source of truth for *content*.

Writes are atomic (tmp + ``os.replace``) because a stage that dies mid-write must
not leave a state file that cannot be parsed — that would strand the whole year.

Status values:
    pending  — not started
    running  — a process is working on it (see ``heartbeat_at``)
    done     — finished, evidence recorded
    failed   — exited non-zero; ``exit_code`` + ``log`` say where to look
    blocked  — a human gate has been reached and not yet signed off
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

STATUSES = ("pending", "running", "done", "failed", "blocked")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state_path(work: Path, year: int) -> Path:
    """Path of the state file for `year` under the campaign work dir."""
    return Path(work) / "state" / f"{year}.json"


def log_path(work: Path, year: int, stage: str) -> Path:
    """Path of the log file for one stage of one year."""
    return Path(work) / "logs" / str(year) / f"{stage}.log"


def skeleton(year: int, stage_names: list[str]) -> dict:
    """A fresh state dict with every stage pending."""
    return {
        "year": year,
        "created_at": _now(),
        "updated_at": _now(),
        "stages": {n: {"status": "pending"} for n in stage_names},
    }


def load(work: Path, year: int, stage_names: list[str]) -> dict:
    """Read the year's state, creating the skeleton if it does not exist yet.

    Stages added to the table after a state file was written appear as pending
    rather than raising — the stage list is allowed to grow mid-campaign.
    """
    p = state_path(work, year)
    if not p.is_file():
        return skeleton(year, stage_names)
    st = json.loads(p.read_text())
    for n in stage_names:
        st["stages"].setdefault(n, {"status": "pending"})
    return st


def save(work: Path, year: int, st: dict, mirror_uri: str | None = None) -> None:
    """Atomically write the state file, then best-effort mirror it to GCS."""
    st["updated_at"] = _now()
    p = state_path(work, year)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, p)
    if mirror_uri:
        _mirror(p, f"{mirror_uri.rstrip('/')}/{year}.json")


def _mirror(local: Path, uri: str) -> None:
    """Copy the state file to GCS. Never raises — a lost mirror must not fail a stage."""
    try:
        from google.cloud import storage
        bucket, _, key = uri[5:].partition("/")
        storage.Client().bucket(bucket).blob(key).upload_from_filename(str(local))
    except Exception as exc:  # noqa: BLE001 - mirroring is best-effort by design
        logger.debug("state mirror to %s failed: %s", uri, exc)


def set_stage(work: Path, year: int, stage: str, stage_names: list[str],
              status: str, mirror_uri: str | None = None, **fields: Any) -> dict:
    """Set one stage's status (plus any extra fields) and persist.

    Args:
        status: one of STATUSES.
        **fields: merged into the stage entry (evidence, exit_code, log, ...).

    Returns:
        The updated state dict.
    """
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")
    st = load(work, year, stage_names)
    entry = st["stages"].setdefault(stage, {})
    entry["status"] = status
    entry.update(fields)
    if status == "running":
        entry["started_at"] = _now()
        entry["heartbeat_at"] = _now()
        entry.pop("finished_at", None)
    elif status in ("done", "failed"):
        entry["finished_at"] = _now()
    save(work, year, st, mirror_uri)
    return st


def heartbeat(work: Path, year: int, stage: str, stage_names: list[str]) -> None:
    """Refresh a running stage's heartbeat so the alerter can tell live from dead."""
    st = load(work, year, stage_names)
    entry = st["stages"].get(stage)
    if entry and entry.get("status") == "running":
        entry["heartbeat_at"] = _now()
        save(work, year, st)


def start_heartbeat(work: Path, year: int, stage: str, stage_names: list[str],
                    period_s: float = 60.0) -> Callable[[], None]:
    """Run a daemon thread refreshing the stage heartbeat; returns a stop callable.

    Mirrors the acquisition loop's heartbeat (planetscope-download/order_basemaps.py)
    so `interannual_inference/alert.py` can apply the same two-signal liveness rule.
    """
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(period_s):
            try:
                heartbeat(work, year, stage, stage_names)
            except Exception as exc:  # noqa: BLE001 - a heartbeat must never kill the stage
                logger.debug("heartbeat failed: %s", exc)

    threading.Thread(target=_loop, daemon=True, name=f"hb-{year}-{stage}").start()
    return stop.set


def age_s(iso: str | None) -> float | None:
    """Seconds since an ISO timestamp, or None if absent/unparseable."""
    if not iso:
        return None
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    except ValueError:
        return None


def elapsed_s(entry: dict) -> float | None:
    """Wall-clock seconds a stage ran (or has been running)."""
    start = entry.get("started_at")
    if not start:
        return None
    end = entry.get("finished_at")
    try:
        t0 = datetime.fromisoformat(start)
        t1 = datetime.fromisoformat(end) if end else datetime.now(timezone.utc)
    except ValueError:
        return None
    return (t1 - t0).total_seconds()
