"""Harvest run scores into the experiment ledger — the anti-drift mechanism.

The ledger (`docs/experiment_ledger.md`) is the experiments SSoT, but its `score`
column is *owned by this script*: each training run writes a structured
`run_summary.json` (see `training/mlflow_utils.log_run_summary`), and this script
reads `best_smoothed` from every run and rewrites the matching ledger row's score
cell in place. Scores can therefore never drift from what training actually logged.

Everything else in the ledger (family, note, findings, recipe, ...) is agent-edited;
this script touches only the score column and never the prose.

Usage:
    python scripts/sync_experiments.py                 # harvest + drift report
    python scripts/sync_experiments.py --backfill      # one-time: legacy .md -> .json, then harvest
    python scripts/sync_experiments.py --runs-dir /mnt/outputs/v1.0/runs

A run's directory name is the join key (== the `name` column in the ledger table).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER = REPO_ROOT / "docs" / "experiment_ledger.md"
DEFAULT_RUNS_DIR = Path("/mnt/outputs/v1.0/runs")

# The harvested run table is delimited by these markers in the ledger so the
# script edits only that block and leaves all other tables/prose untouched.
TABLE_BEGIN = "<!-- RUN-TABLE:BEGIN"
TABLE_END = "<!-- RUN-TABLE:END -->"


def _parse_md_metrics(md: str) -> dict:
    """Parse a legacy run_summary.md into the same dict shape as run_summary.json."""
    out: dict = {}
    m = re.search(r"-\s*Status:\s*\*\*(.+?)\*\*", md)
    if m:
        out["status"] = m.group(1).strip()
    m = re.search(r"-\s*Seed:\s*(\S+)", md)
    if m:
        out["seed"] = _coerce(m.group(1))
    for key, val in re.findall(r"^\|\s*([A-Za-z_]\w*)\s*\|\s*([^|]+?)\s*\|\s*$", md, re.M):
        if key.lower() == "metric":  # header row
            continue
        out[key] = _coerce(val)
    return out


def _coerce(s: str):
    s = s.strip()
    if s in ("True", "False"):
        return s == "True"
    try:
        f = float(s)
        return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f
    except ValueError:
        return s


def _parse_log_best(log: str) -> dict | None:
    """Recover best_smoothed from the last EarlyStopping line of a train.log.

    Runs that were killed (or never reached the summary step) have no
    run_summary.* but do log `... best=<x> (epoch=<e>` every epoch. The run is
    marked "incomplete" because the absence of a run_summary means training did
    not exit normally.
    """
    matches = re.findall(r"best=([\d.]+)\s*\(epoch=(\d+)", log)
    if not matches:
        return None
    best, epoch = matches[-1]
    return {"best_smoothed": float(best), "best_epoch": int(epoch), "status": "incomplete"}


def backfill_json(runs_dir: Path) -> int:
    """Write run_summary.json for any run dir lacking one.

    Source order: existing run_summary.md (parsed) > train.log (EarlyStopping best).
    """
    n = 0
    for run_dir in sorted(p for p in runs_dir.glob("*") if p.is_dir()):
        json_path = run_dir / "run_summary.json"
        if json_path.exists():
            continue
        md_path, log_path = run_dir / "run_summary.md", run_dir / "train.log"
        if md_path.exists():
            summary = _parse_md_metrics(md_path.read_text())
        elif log_path.exists():
            summary = _parse_log_best(log_path.read_text())
            if summary is None:
                continue
        else:
            continue
        summary.setdefault("run_name", run_dir.name)
        json_path.write_text(json.dumps(summary, indent=2, default=str))
        n += 1
    logger.info("Backfill: wrote %d run_summary.json.", n)
    return n


def harvest_scores(runs_dir: Path) -> dict[str, float | None]:
    """name -> best_smoothed (None if missing/non-finite)."""
    scores: dict[str, float | None] = {}
    for jp in sorted(runs_dir.glob("*/run_summary.json")):
        name = jp.parent.name
        try:
            data = json.loads(jp.read_text())
        except json.JSONDecodeError:
            logger.warning("  ! unreadable json: %s", jp)
            continue
        val = data.get("best_smoothed")
        scores[name] = val if isinstance(val, (int, float)) and math.isfinite(val) else None
    return scores


def _fmt(score: float | None) -> str:
    return f"{score:.4f}" if score is not None else "—"


def update_ledger(scores: dict[str, float | None], ledger: Path = LEDGER) -> None:
    """Rewrite the score cell of each row in the ledger's RUN-TABLE block.

    Reports drift (ledger != json), ledger rows with no run dir, and run dirs
    with no ledger row. ``ledger`` defaults to the v2.0 SSoT; pass the v2.1 ledger
    (`docs/experiment_ledger_v21.md`) to harvest that program's runs instead.
    """
    text = ledger.read_text()
    try:
        block = text.split(TABLE_BEGIN, 1)[1].split(TABLE_END, 1)[0]
    except IndexError:
        raise SystemExit(
            f"Could not find the {TABLE_BEGIN} ... {TABLE_END} block in {ledger}.\n"
            "The run table must be wrapped in those markers for sync to target it."
        )

    lines = block.splitlines()
    header_idx = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("|"))
    cols = [c.strip().lower() for c in lines[header_idx].strip().strip("|").split("|")]
    name_i, score_i = cols.index("name"), cols.index("score")

    seen: set[str] = set()
    drift, missing_dir = [], []
    new_lines = list(lines)
    for i in range(header_idx + 2, len(lines)):  # skip header + separator
        ln = lines[i]
        if not ln.lstrip().startswith("|"):
            continue
        cells = ln.split("|")  # keep outer empties to preserve formatting
        inner = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(inner) <= max(name_i, score_i):
            continue
        name = inner[name_i]
        seen.add(name)
        if name not in scores:
            missing_dir.append(name)
            continue
        new = _fmt(scores[name])
        old = inner[score_i]
        if old != new and not (old in ("—", "", "TBD") and scores[name] is None):
            drift.append((name, old, new))
        # rewrite the score cell (cells index = score_i + 1, accounting for leading empty)
        inner[score_i] = new
        new_lines[i] = "| " + " | ".join(inner) + " |"

    ledger.write_text(text.replace(block, "\n".join(new_lines)))

    extra_dirs = sorted(set(scores) - seen)
    logger.info("\n--- sync report ---")
    logger.info("ledger rows updated: %d", len(seen) - len(missing_dir))
    if drift:
        logger.info("\nSCORE CHANGES (%d):", len(drift))
        for name, old, new in drift:
            logger.info("  %-44s %s -> %s", name, old, new)
    if missing_dir:
        logger.info("\nLEDGER ROWS WITH NO RUN DIR (%d) — left as-is:", len(missing_dir))
        for name in missing_dir:
            logger.info("  %s", name)
    if extra_dirs:
        logger.info("\nRUN DIRS WITH NO LEDGER ROW (%d) — add a row if real:", len(extra_dirs))
        for name in extra_dirs:
            logger.info("  %-44s best_smoothed=%s", name, _fmt(scores[name]))
    if not (drift or missing_dir or extra_dirs):
        logger.info("clean — ledger fully in sync with run_summary.json.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                    help="canonical run dirs (default: %(default)s)")
    ap.add_argument("--backfill", action="store_true",
                    help="first convert legacy run_summary.md -> run_summary.json")
    ap.add_argument("--ledger", type=Path, default=LEDGER,
                    help="ledger .md to harvest into (default: the v2.0 SSoT; pass "
                         "docs/experiment_ledger_v21.md for the v2.1 program)")
    args = ap.parse_args()

    if not args.runs_dir.exists():
        raise SystemExit(f"runs dir not found: {args.runs_dir}")
    if args.backfill:
        backfill_json(args.runs_dir)
    scores = harvest_scores(args.runs_dir)
    logger.info("harvested %d run scores from %s", len(scores), args.runs_dir)
    update_ledger(scores, ledger=args.ledger)


if __name__ == "__main__":
    main()
