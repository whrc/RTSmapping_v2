"""Walk one year through the pipeline, stopping wherever a human is needed.

The driver is deliberately not clever. It runs the stage table in order, skipping
what is done, waiting on what is detached, and STOPPING at anything that needs a
person — an unmet external prerequisite, a human gate, or a failure. It never
decides that a year looks good enough to continue past a gate.

Usage:
    python interannual_inference/drive.py --year 2022                 # run until blocked
    python interannual_inference/drive.py --year 2022 --until shard   # stop after a given stage
    python interannual_inference/drive.py --year 2022 --dry-run       # show what it would do
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interannual_inference import stages as S  # noqa: E402
from interannual_inference import state as ST  # noqa: E402
from interannual_inference.run_stage import (CONFIG, collect_evidence,  # noqa: E402
                                             load_config, run)
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

POLL_S = 300.0


def wait_for_detached(stage: S.Stage, year: int, cfg: dict, poll_s: float = POLL_S) -> bool:
    """Poll a detached stage's probe until it reaches its expected total.

    Returns True when complete. A detached stage has no exit code to wait on — the
    GEE export's launcher exits immediately and the inference supervisor is nohup'd —
    so completion is defined by the artifact count reaching the expected total.
    """
    work = Path(cfg["paths"]["work"])
    last_done, last_change = None, time.time()
    limit = cfg["alerts"]["no_progress_after_s"]
    while True:
        p = stage.probe(year, cfg) if stage.probe else None
        if p is None:
            logger.warning("%s/%s has no readable progress yet", year, stage.name)
        else:
            logger.info("%s/%s %s/%s (%.1f%%)%s", year, stage.name,
                        f"{p.get('done', 0):,}", f"{p.get('total', 0):,}", p.get("pct", 0.0),
                        f" ETA {p['eta_hours']:.1f}h" if p.get("eta_hours") else "")
            if p.get("done") != last_done:
                last_done, last_change = p.get("done"), time.time()
            if p.get("total") and p.get("done", 0) >= p["total"]:
                return True
            if time.time() - last_change > limit:
                logger.error("%s/%s has not advanced in %.1f h — stopping for a human",
                             year, stage.name, (time.time() - last_change) / 3600)
                return False
        ST.heartbeat(work, year, stage.name, S.NAMES)
        time.sleep(poll_s)


def drive(year: int, cfg: dict, until: str | None = None, dry_run: bool = False,
          poll_s: float = POLL_S) -> int:
    """Run `year` through the chain. Returns 0 if it reached the end or a clean stop."""
    work = Path(cfg["paths"]["work"])
    mirror = cfg["paths"].get("state_mirror")

    for stage in S.ORDER:
        st = ST.load(work, year, S.NAMES)
        entry = st["stages"].get(stage.name, {"status": "pending"})
        status = entry["status"]

        if status == "done":
            logger.info("%s/%s done — skipping", year, stage.name)
            if until and stage.name == until:
                return 0
            continue

        if status == "blocked":
            logger.warning("%s/%s is a HUMAN GATE awaiting sign-off. Evidence: %s\n"
                           "  %s\n  sign off: python interannual_inference/run_stage.py --year %d "
                           "--stage %s --sign-off",
                           year, stage.name, entry.get("evidence", {}), stage.note,
                           year, stage.name)
            return 0

        if status == "failed":
            logger.error("%s/%s FAILED earlier (rc=%s, log=%s) — fix it, then re-run with "
                         "--force", year, stage.name, entry.get("exit_code"), entry.get("log"))
            return 1

        unmet = S.unmet_prereqs(stage, st)
        if unmet:
            logger.info("%s/%s waiting on: %s — stopping here", year, stage.name,
                        ", ".join(unmet))
            return 0

        if stage.external:
            logger.info("%s/%s is run elsewhere (%s) — stopping here.\n"
                        "  mark it: python interannual_inference/run_stage.py --year %d --stage %s "
                        "--mark-done", year, stage.name, stage.note, year, stage.name)
            return 0

        if status == "running" and stage.detached:
            if dry_run:
                logger.info("%s/%s already running detached — would resume polling",
                            year, stage.name)
                continue
            logger.info("%s/%s already running detached — resuming the wait", year, stage.name)
        else:
            rc = run(year, stage.name, cfg, dry_run=dry_run)
            if dry_run:
                continue
            if rc != 0:
                return rc

        if stage.detached:
            if not wait_for_detached(stage, year, cfg, poll_s):
                return 1
            ev = collect_evidence(stage, year, cfg)
            ST.set_stage(work, year, stage.name, S.NAMES, "done",
                         mirror_uri=mirror, evidence=ev)
            logger.info("%s/%s DONE. Evidence: %s", year, stage.name, ev)

        # A gate that just ran is now blocked; re-read rather than assume.
        if ST.load(work, year, S.NAMES)["stages"][stage.name]["status"] == "blocked":
            logger.warning("%s/%s reached a HUMAN GATE — stopping", year, stage.name)
            return 0

        if until and stage.name == until:
            logger.info("%s: reached --until %s", year, until)
            return 0

    logger.info("%s: every stage done", year)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--until", default=None, help=f"stop after this stage ({', '.join(S.NAMES)})")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--poll-s", type=float, default=POLL_S,
                   help="seconds between progress polls of a detached stage")
    args = p.parse_args()
    setup_logging()
    if args.until:
        S.get(args.until)  # validate early
    return drive(args.year, load_config(args.config), until=args.until,
                 dry_run=args.dry_run, poll_s=args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
