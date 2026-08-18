"""Alert when an acquisition year has stopped and needs a human.

The supervisor restarts crashes and stalls by itself, so most failures need
nobody. Three do not: an expired Planet key (`rc=2`, deliberately not retried),
a crash loop, and a VM reboot that takes the in-memory keys with it. Each of
those leaves the run idle and *silent* — and two of the three can only be fixed
by the key holder, who is the person least likely to be watching a terminal.
This closes that gap.

Run from cron. It posts to Slack when a year looks stopped, once per incident,
and once more when the year finishes.

Liveness needs two signals, not one. A stale heartbeat alone is not enough: on
resume the order loop lists the delivery prefix before its first heartbeat,
which can take the best part of an hour on a populated year — alerting on that
would cry wolf on every restart. So a year is "stopped" only when its heartbeat
is stale *and* no ordering process for it is running.

Usage:
    python planetscope-download/alert_if_stopped.py            # post if needed
    python planetscope-download/alert_if_stopped.py --dry-run  # print instead
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(os.environ.get("PSD_WORK", "/mnt/outputs/planetscope-download"))
STATUS_DIR = WORK / "status"
LOG_DIR = WORK / "logs"
# Comfortably past the 900s stall watchdog plus the supervisor's restart delay,
# so a self-healing restart never trips this.
STALE_AFTER_S = 1800
WEBHOOK_FILE = WORK / "slack_webhook"


def webhook_url() -> str | None:
    """Read the Slack webhook from env or the (secret, 600) file beside the run."""
    url = os.environ.get("PSD_SLACK_WEBHOOK")
    if url:
        return url.strip()
    if WEBHOOK_FILE.is_file():
        return WEBHOOK_FILE.read_text().strip() or None
    return None


def ordering_process_alive(year: int) -> bool:
    """True if an order_basemaps process for `year` is running."""
    r = subprocess.run(["pgrep", "-f", f"order_basemaps.py --year {year}"],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def log_tail(year: int, n: int = 12) -> str:
    p = LOG_DIR / f"orders_{year}.log"
    if not p.is_file():
        return "(no order log)"
    lines = [ln for ln in p.read_text(errors="replace").splitlines()
             if "FutureWarning" not in ln and "warnings.warn" not in ln]
    return "\n".join(lines[-n:])


def post(url: str, text: str, dry_run: bool) -> None:
    if dry_run or not url:
        print(text)
        return
    req = urllib.request.Request(
        url, data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be posted instead of posting")
    p.add_argument("--status-dir", type=Path, default=STATUS_DIR)
    args = p.parse_args()

    url = webhook_url()
    # Collect first, post second: with nothing to report this exits silently even
    # when unconfigured, so the cron log stays empty until something is actually
    # wrong. A checker that chatters every 10 minutes trains you to ignore it.
    messages: list[str] = []

    for f in sorted(args.status_dir.glob("[0-9][0-9][0-9][0-9].json")):
        try:
            s = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue                      # mid-write; next tick will catch it
        year = s["year"]
        marker = args.status_dir / f".alerted_{year}"
        done = s["n_done"] >= s["n_total"]
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(s["heartbeat_at"])).total_seconds()

        if done:
            if not (args.status_dir / f".done_{year}").exists():
                messages.append(
                    f":white_check_mark: *PlanetScope {year} acquisition complete* — "
                    f"{s['n_ordered']:,} ordered, {s['n_failed']:,} failed.\n"
                    f"Next: build the quad index with `--expect-quads {s['n_total']}`.")
                (args.status_dir / f".done_{year}").touch()
            marker.unlink(missing_ok=True)
            continue

        stopped = age > STALE_AFTER_S and not ordering_process_alive(year)
        if stopped and not marker.exists():
            messages.append(
                 f":rotating_light: *PlanetScope {year} acquisition has stopped* "
                 f"and needs a human.\n"
                 f"Progress: {s['n_done']:,}/{s['n_total']:,} ({s['pct_done']}%), "
                 f"last quad `{s['last_quad_id']}`, quiet for {age / 60:.0f} min.\n"
                 f"Most likely an expired Planet key (only Heidi can restart it) or a "
                 f"VM reboot. To resume:\n"
                 f"```\ntmux new -s planet\n"
                 f"cd /home/ext_yyang_woodwellclimate_org/RTSmappingDL\n"
                 f"./planetscope-download/run_year.sh {year}\n```\n"
                 f"Delivered quads are skipped, so it picks up where it stopped.\n"
                 f"Last log lines:\n```\n{log_tail(year)}\n```")
            marker.touch()
        elif not stopped:
            marker.unlink(missing_ok=True)   # recovered; re-arm for next time

    if not messages:
        return 0
    if not url and not args.dry_run:
        print(f"Something needs attention but no webhook is configured: write it to "
              f"{WEBHOOK_FILE} (chmod 600) or set PSD_SLACK_WEBHOOK.", file=sys.stderr)
        for m in messages:
            print(m, file=sys.stderr)
        return 2
    for m in messages:
        post(url, m, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
