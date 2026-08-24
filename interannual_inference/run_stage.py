"""Run ONE stage of ONE year, with prerequisite checks, logging and evidence capture.

This is the only place a stage command is actually executed. It refuses to run a
stage whose prerequisites are not done, refuses to re-run a finished stage without
``--force``, streams output to a per-stage log, and records what the stage produced
by reading its artifact afterwards.

Usage:
    python interannual_inference/run_stage.py --year 2022 --stage tile_grid
    python interannual_inference/run_stage.py --year 2022 --stage drift_check --force
    python interannual_inference/run_stage.py --year 2022 --stage acquire --mark-done   # external
    python interannual_inference/run_stage.py --year 2022 --stage drift_check --sign-off
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from interannual_inference import stages as S  # noqa: E402
from interannual_inference import state as ST  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: Path = CONFIG) -> dict:
    """Read the campaign config (config.yaml)."""
    return yaml.safe_load(path.read_text())


def collect_evidence(stage: S.Stage, year: int, cfg: dict) -> dict:
    """Read the stage's artifact for the ledger. Never raises — evidence is not the work."""
    if stage.evidence is None:
        return {}
    try:
        return stage.evidence(year, cfg) or {}
    except Exception as exc:  # noqa: BLE001 - a missing artifact is data, not a crash
        logger.warning("evidence for %s/%s unavailable: %s", year, stage.name, exc)
        return {"evidence_error": str(exc)}


def run(year: int, stage_name: str, cfg: dict, force: bool = False,
        dry_run: bool = False) -> int:
    """Execute one stage. Returns a process-style exit code."""
    stage = S.get(stage_name)
    work = Path(cfg["paths"]["work"])
    mirror = cfg["paths"].get("state_mirror")
    st = ST.load(work, year, S.NAMES)
    entry = st["stages"].get(stage_name, {})

    if entry.get("status") == "done" and not force:
        logger.info("%s/%s already done (%s) — nothing to do; --force to re-run",
                    year, stage_name, entry.get("evidence", {}))
        return 0

    unmet = S.unmet_prereqs(stage, st)
    if unmet:
        logger.error("%s/%s blocked: prerequisite(s) not done: %s",
                     year, stage_name, ", ".join(unmet))
        return 2

    if stage.cmd is None:
        logger.error("%s/%s is an external stage (%s) — this script does not run it. "
                     "Use --mark-done once it is finished.",
                     year, stage_name, stage.note or "run elsewhere")
        return 2

    argv = stage.cmd(year, cfg)
    if dry_run:
        print(" ".join(argv))
        return 0

    log = ST.log_path(work, year, stage_name)
    log.parent.mkdir(parents=True, exist_ok=True)
    ST.set_stage(work, year, stage_name, S.NAMES, "running",
                 mirror_uri=mirror, log=str(log), cmd=" ".join(argv))
    stop_hb = ST.start_heartbeat(work, year, stage_name, S.NAMES)
    logger.info("%s/%s START -> %s", year, stage_name, log)

    try:
        with open(log, "a") as fh:
            fh.write(f"\n=== {stage_name} {year} START ===\n{' '.join(argv)}\n")
            fh.flush()
            rc = subprocess.call(argv, stdout=fh, stderr=subprocess.STDOUT)
    finally:
        stop_hb()

    ev = collect_evidence(stage, year, cfg)
    if rc == 0:
        # A detached stage's launcher exiting 0 means the work was STARTED, not finished.
        status = "running" if stage.detached else ("blocked" if stage.gate else "done")
        ST.set_stage(work, year, stage_name, S.NAMES, status,
                     mirror_uri=mirror, evidence=ev, exit_code=0, log=str(log))
        if stage.detached:
            logger.info("%s/%s launched; work continues detached — poll with status.py",
                        year, stage_name)
        elif stage.gate:
            logger.info("%s/%s reached a HUMAN GATE. Evidence: %s\n  %s\n"
                        "  sign off with: --stage %s --sign-off",
                        year, stage_name, ev, stage.note, stage_name)
        else:
            logger.info("%s/%s DONE. Evidence: %s", year, stage_name, ev)
    else:
        ST.set_stage(work, year, stage_name, S.NAMES, "failed",
                     mirror_uri=mirror, evidence=ev, exit_code=rc, log=str(log))
        logger.error("%s/%s FAILED rc=%d — see %s", year, stage_name, rc, log)
    return rc


def mark(year: int, stage_name: str, cfg: dict, status: str) -> int:
    """Record an outcome we did not produce ourselves (external stage, or a sign-off)."""
    stage = S.get(stage_name)
    work = Path(cfg["paths"]["work"])
    ev = collect_evidence(stage, year, cfg)
    ST.set_stage(work, year, stage_name, S.NAMES, status,
                 mirror_uri=cfg["paths"].get("state_mirror"), evidence=ev)
    logger.info("%s/%s marked %s. Evidence: %s", year, stage_name, status, ev)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--stage", required=True, help=f"one of: {', '.join(S.NAMES)}")
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--force", action="store_true", help="re-run a stage already done")
    p.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--mark-done", action="store_true",
                   help="record an external stage as finished (acquire, merge, ...)")
    g.add_argument("--mark-failed", action="store_true", help="record a stage as failed")
    g.add_argument("--sign-off", action="store_true",
                   help="clear a human gate so dependent stages may run")
    args = p.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    if args.mark_done or args.sign_off:
        return mark(args.year, args.stage, cfg, "done")
    if args.mark_failed:
        return mark(args.year, args.stage, cfg, "failed")
    return run(args.year, args.stage, cfg, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
