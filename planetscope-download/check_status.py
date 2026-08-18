"""Show acquisition progress for every year, without attaching to tmux.

Reads the status JSON that order_basemaps.py refreshes about once a minute.
A heartbeat older than a few minutes on a run you believe is live means the
process died and the supervisor has not restarted it.

Usage:
    python planetscope-download/check_status.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Mirrors order_basemaps.DEFAULT_WORK — runtime outputs live outside the repo.
DEFAULT_STATUS_DIR = Path(
    os.environ.get("PSD_WORK", "/mnt/outputs/planetscope-download")) / "status"
STALE_AFTER_S = 300


def _age(iso: str) -> float:
    return (datetime.now(timezone.utc)
            - datetime.fromisoformat(iso)).total_seconds()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--status-dir", type=Path, default=DEFAULT_STATUS_DIR)
    args = p.parse_args()

    files = sorted(args.status_dir.glob("[0-9][0-9][0-9][0-9].json"))
    if not files:
        print(f"No status files in {args.status_dir} — nothing has been run yet.")
        return 0

    print(f"{'year':>6} {'done':>18} {'pct':>7} {'ordered':>9} {'skipped':>9} "
          f"{'failed':>7} {'ord/min':>8} {'eta_h':>7}  heartbeat")
    for f in files:
        s = json.loads(f.read_text())
        age = _age(s["heartbeat_at"])
        done = s["n_done"] >= s["n_total"]
        # A finished year stops heartbeating by design — only an *incomplete* run
        # going quiet means the process died without the supervisor restarting it.
        if done:
            flag = "  complete"
        elif age > STALE_AFTER_S:
            flag = "  STALE — process died? check logs/"
        else:
            flag = ""
        print(f"{s['year']:>6} {s['n_done']:>8,}/{s['n_total']:<9,} {s['pct_done']:>6.1f}% "
              f"{s['n_ordered']:>9,} {s['n_skipped']:>9,} {s['n_failed']:>7,} "
              f"{s['orders_per_min']:>8.1f} {s['eta_hours']:>7.1f}  {age / 60:.0f} min ago{flag}")

    failed = sorted(args.status_dir.glob("failed_orders_*.csv"))
    for f in failed:
        n = max(sum(1 for _ in f.open()) - 1, 0)
        if n:
            print(f"\n{n} failed quad{'s' if n != 1 else ''} recorded in {f} — sweep up with:\n"
                  f"  python planetscope-download/order_basemaps.py --year {f.stem[-4:]} \\\n"
                  f"      --grids planetscope-download/data/"
                  f"circumpolar_south_planet_basemap_grids_{f.stem[-4:]}.geojson \\\n"
                  f"      --retry-failed {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
