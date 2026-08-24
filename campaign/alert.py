"""Tell a human when the campaign needs one. Runs from cron.

Most failures need nobody: the inference supervisor restarts crashed workers, the
GEE export is resumable, and the acquisition loop self-heals. Four situations do
need a person, and all four are silent:

  * a stage FAILED,
  * a human GATE was reached (drift check, QC),
  * a running stage has gone quiet (stale heartbeat AND no progress),
  * a year finished every stage.

Liveness takes two signals, not one — the same rule as the acquisition alerter.
A stale heartbeat alone is not enough: a detached stage legitimately sits still
while GEE queues or while the inference run lists its shards at startup. So a
stage is "stuck" only when its heartbeat is stale *and* its probe has not advanced.

State is kept in <work>/alerts_seen.json so each incident is announced once, not
every ten minutes.

Usage:
    python campaign/alert.py             # post what is new
    python campaign/alert.py --dry-run   # print instead
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from campaign import notify, stages as S, state as ST  # noqa: E402
from campaign.run_stage import CONFIG, load_config  # noqa: E402
from campaign.status import probe_stage  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def seen_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["work"]) / "alerts_seen.json"


def load_seen(cfg: dict) -> dict:
    p = seen_path(cfg)
    return json.loads(p.read_text()) if p.is_file() else {}


def save_seen(cfg: dict, seen: dict) -> None:
    p = seen_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(seen, indent=1))


def stuck(entry: dict, progress: dict | None, prev: dict | None, cfg: dict) -> bool:
    """Two-signal liveness: stale heartbeat AND no forward progress."""
    hb = ST.age_s(entry.get("heartbeat_at"))
    if hb is None or hb < cfg["alerts"]["stale_after_s"]:
        return False
    if progress is None:
        return True
    if prev is None or prev.get("done") is None:
        return False   # first sighting: nothing to compare against yet
    if progress.get("done", 0) > prev["done"]:
        return False
    return (time.time() - prev.get("at", 0)) > cfg["alerts"]["no_progress_after_s"]


def collect(cfg: dict, years: list[int]) -> tuple[list[tuple[str, str]], dict]:
    """Find what needs announcing. Returns (incidents, updated seen-state).

    Each incident is (key, message); `key` makes it announce-once.
    """
    work = Path(cfg["paths"]["work"])
    seen = load_seen(cfg)
    out: list[tuple[str, str]] = []

    for y in years:
        st = ST.load(work, y, S.NAMES)
        for stage in S.ORDER:
            e = st["stages"].get(stage.name, {"status": "pending"})
            key_base = f"{y}:{stage.name}"

            if e["status"] == "failed":
                k = f"{key_base}:failed:{e.get('finished_at', '')}"
                if k not in seen:
                    out.append((k, f":x: *{y} / {stage.name} FAILED* (rc={e.get('exit_code')})\n"
                                   f"log: `{e.get('log')}`"))

            elif e["status"] == "blocked":
                k = f"{key_base}:gate:{e.get('finished_at', '')}"
                if k not in seen:
                    out.append((k, f":pause_button: *{y} / {stage.name} needs sign-off*\n"
                                   f"{stage.note}\nevidence: `{e.get('evidence', {})}`"))

            elif e["status"] == "running":
                prog = probe_stage(stage, y, cfg)
                prev = seen.get(f"{key_base}:progress")
                if stuck(e, prog, prev, cfg):
                    k = f"{key_base}:stuck:{e.get('started_at', '')}"
                    if k not in seen:
                        hb = ST.age_s(e.get("heartbeat_at")) or 0
                        out.append((k, f":warning: *{y} / {stage.name} looks stuck* — "
                                       f"heartbeat {hb / 60:.0f} min old and no progress.\n"
                                       f"log: `{e.get('log')}`"))
                if prog is not None and (prev is None or prog.get("done", 0) > prev.get("done", -1)):
                    seen[f"{key_base}:progress"] = {"done": prog.get("done"), "at": time.time()}

        if all(st["stages"].get(n, {}).get("status") == "done" for n in S.NAMES):
            k = f"{y}:complete"
            if k not in seen:
                out.append((k, f":white_check_mark: *{y} is complete* — every campaign stage done."))

    return out, seen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    setup_logging(level="WARNING")
    cfg = load_config(args.config)

    incidents, seen = collect(cfg, cfg["years"])
    if not incidents:
        save_seen(cfg, seen)
        return 0

    url = notify.webhook_url(cfg)
    if url is None and not args.dry_run:
        logger.error("no Slack webhook configured (%s) — %d incident(s) unannounced",
                     cfg["alerts"]["webhook_file"], len(incidents))
        return 1

    for key, msg in incidents:
        if notify.post(url, msg, args.dry_run) or args.dry_run:
            seen[key] = time.time()
    save_seen(cfg, seen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
