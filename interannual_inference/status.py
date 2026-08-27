"""One screen showing where every year of the campaign stands.

Prints a year x stage matrix. Running stages that expose a probe are queried live,
so `s2_export` and `infer` show real progress rather than just "running".

    ✓ done   ▶ running   ⏸ gate awaiting sign-off   ✗ failed   · pending

Usage:
    python interannual_inference/status.py                 # the matrix
    python interannual_inference/status.py --year 2022     # drill down into one year
    python interannual_inference/status.py --json out.json # machine-readable snapshot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interannual_inference import stages as S  # noqa: E402
from interannual_inference import state as ST  # noqa: E402
from interannual_inference.run_stage import CONFIG, load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

MARK = {"done": "✓", "running": "▶", "failed": "✗", "pending": "·", "blocked": "⏸"}


def probe_stage(stage: S.Stage, year: int, cfg: dict) -> dict | None:
    """Live progress for a running stage. Never raises — a probe is a convenience."""
    if stage.probe is None:
        return None
    try:
        return stage.probe(year, cfg)
    except Exception as exc:  # noqa: BLE001 - probing must not break the status view
        logger.debug("probe %s/%s failed: %s", year, stage.name, exc)
        return None


def snapshot(cfg: dict, years: list[int], live: bool = True) -> dict:
    """Collect every year's state, with live probes for running stages."""
    work = Path(cfg["paths"]["work"])
    out: dict = {"years": {}}
    for y in years:
        st = ST.load(work, y, S.NAMES)
        row = {}
        for stage in S.ORDER:
            e = dict(st["stages"].get(stage.name, {"status": "pending"}))
            if live and e["status"] == "running":
                p = probe_stage(stage, y, cfg)
                if p:
                    e["progress"] = p
            row[stage.name] = e
        out["years"][y] = row
    return out


def _cell(entry: dict) -> str:
    """One matrix cell: a mark, plus a percentage when we have live progress."""
    m = MARK.get(entry["status"], "?")
    p = entry.get("progress")
    if p and p.get("pct") is not None:
        return f"{m}{p['pct']:.0f}%"
    return m


def render_matrix(snap: dict) -> str:
    """The year x stage grid."""
    names = S.NAMES
    widths = [max(len(n), 6) for n in names]
    head = "year   " + " ".join(n[:w].ljust(w) for n, w in zip(names, widths))
    lines = [head, "-" * len(head)]
    for y, row in sorted(snap["years"].items()):
        cells = [_cell(row[n]).ljust(w) for n, w in zip(names, widths)]
        lines.append(f"{y}   " + " ".join(cells))
    return "\n".join(lines)


def render_year(year: int, row: dict) -> str:
    """Per-stage detail for one year."""
    lines = [f"=== {year} ==="]
    for stage in S.ORDER:
        e = row[stage.name]
        bits = [f"  {MARK.get(e['status'], '?')} {stage.name:<12} {e['status']:<8}"]
        el = ST.elapsed_s(e)
        if el:
            bits.append(f"{el / 3600:6.1f}h")
        p = e.get("progress")
        if p:
            bits.append(f"{p.get('done', 0):,}/{p.get('total', 0):,} ({p.get('pct', 0):.1f}%)")
            if p.get("eta_hours"):
                bits.append(f"ETA {p['eta_hours']:.1f}h")
            if p.get("stale_workers"):
                bits.append(f"⚠ {p['stale_workers']} stale worker(s)")
        if e.get("evidence"):
            bits.append(str(e["evidence"]))
        if e["status"] == "failed":
            bits.append(f"rc={e.get('exit_code')} log={e.get('log')}")
        if e["status"] == "blocked":
            bits.append(f"GATE — {stage.note}")
        lines.append("  ".join(bits))
        # A DETACHED stage's launcher exits by design (GEE tasks, nohup'd inference),
        # taking its heartbeat thread with it — so a stale heartbeat there is normal,
        # not a fault. Only flag it when progress has also stopped, which is the same
        # two-signal rule alert.py uses to avoid crying wolf.
        hb = ST.age_s(e.get("heartbeat_at"))
        advancing = bool(p) and (p.get("done") or 0) > 0
        if (e["status"] == "running" and hb is not None and hb > 1800
                and not (stage.detached and advancing)):
            lines.append(f"      ⚠ heartbeat {hb / 60:.0f} min old")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# PROGRESS.md — the committed, human-readable view of the same state
# --------------------------------------------------------------------------
PROGRESS_PATH = Path(__file__).resolve().parent / "PROGRESS.md"

_MD_MARK = {"done": "✅", "running": "🔄", "failed": "❌", "pending": "·", "blocked": "⏸"}


def _md_cell(entry: dict) -> str:
    """One markdown table cell: mark, plus live percentage where we have one."""
    m = _MD_MARK.get(entry["status"], "?")
    p = entry.get("progress")
    if entry["status"] == "running" and p and p.get("pct") is not None:
        return f"{m} {p['pct']:.0f}%"
    return m


def render_progress(snap: dict, cfg: dict) -> str:
    """Render PROGRESS.md: one row per year, one column per stage, plus evidence.

    Generated from the per-year state files, which stay the source of truth — this
    file is the readable view of them, never the other way round (CLAUDE.md SSoT).
    """
    from datetime import datetime, timezone

    names = S.NAMES
    head = "| year | " + " | ".join(names) + " |"
    rule = "|---" * (len(names) + 1) + "|"
    rows = []
    for y, row in sorted(snap["years"].items()):
        rows.append(f"| **{y}** | " + " | ".join(_md_cell(row[n]) for n in names) + " |")

    ev_lines = []
    for y, row in sorted(snap["years"].items()):
        bits = []
        for n in names:
            e = row[n]
            if e.get("evidence"):
                kv = ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}"
                               for k, v in e["evidence"].items() if v is not None)
                if kv:
                    bits.append(f"`{n}` {kv}")
            p = e.get("progress")
            if e["status"] == "running" and p:
                bits.append(f"`{n}` {p.get('done', 0):,}/{p.get('total', 0):,}"
                            + (f", ETA {p['eta_hours']:.0f}h" if p.get("eta_hours") else ""))
        if bits:
            ev_lines.append(f"- **{y}** — " + " · ".join(bits))

    ee = cfg.get("docker", {}).get("ee_projects") or {}
    ee_lines = [f"| {y} | `{ee[y]}` |" for y in sorted(ee)] if ee else []

    return "\n".join([
        "# Interannual run — progress",
        "",
        "> **Generated file — do not hand-edit.** Written from the per-year state at",
        "> `/mnt/outputs/interannual_inference/state/<year>.json`, which is the source of",
        "> truth. Refresh with `python interannual_inference/status.py --write-progress`;",
        "> `run_stage.py` also rewrites it after every stage transition.",
        "",
        f"Last updated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "✅ done · 🔄 running · ⏸ awaiting human sign-off · ❌ failed · · not started",
        "",
        head, rule, *rows,
        "",
        "## Evidence",
        "",
        *(ev_lines or ["_nothing recorded yet._"]),
        "",
        "## Earth Engine project per year",
        "",
        "Quota and concurrency are per-project, so each year exports on its own.",
        "",
        *(["| year | EE project |", "|---|---|", *ee_lines] if ee_lines else ["_none configured._"]),
        "",
    ])


def write_progress(cfg: dict, years: list[int], path: Path = PROGRESS_PATH,
                   live: bool = True) -> Path:
    """Regenerate PROGRESS.md from current state. Never raises — it is a view."""
    try:
        path.write_text(render_progress(snapshot(cfg, years, live=live), cfg))
    except Exception as exc:  # noqa: BLE001 - a stale view must not fail a stage
        logger.warning("could not write %s: %s", path, exc)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--year", type=int, default=None, help="drill into one year")
    p.add_argument("--json", type=Path, default=None, help="also write the snapshot here")
    p.add_argument("--write-progress", action="store_true",
                   help=f"regenerate {PROGRESS_PATH.name} (the committed progress view)")
    p.add_argument("--no-live", action="store_true",
                   help="skip live probes (fast, offline)")
    args = p.parse_args()
    setup_logging(level="WARNING")
    cfg = load_config(args.config)
    years = [args.year] if args.year else cfg["years"]

    snap = snapshot(cfg, years, live=not args.no_live)
    if args.json:
        args.json.write_text(json.dumps(snap, indent=1, default=str))
    if args.write_progress:
        # Always render every year, not just --year, so a drill-down cannot silently
        # blank the other rows of the committed file.
        p = write_progress(cfg, cfg["years"], live=not args.no_live)
        print(f"wrote {p}")
    if args.year:
        print(render_year(args.year, snap["years"][args.year]))
    else:
        print(render_matrix(snap))
        print("\n  ✓ done   ▶ running   ⏸ gate awaiting sign-off   ✗ failed   · pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
