"""Slack posting for interannual-inference alerts.

Deliberately a private copy of the few lines in
``planetscope-download/alert_if_stopped.py`` rather than a shared import. That
script is LIVE in cron watching Heidi's acquisition; refactoring it to import from
here would risk a running alert path to save ten lines. The webhook file itself is
shared, so there is still only one secret.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def webhook_url(cfg: dict) -> str | None:
    """Read the Slack webhook from the environment or the (mode 600) file."""
    url = os.environ.get("RTS_SLACK_WEBHOOK")
    if url:
        return url.strip()
    p = Path(cfg["alerts"]["webhook_file"])
    if p.is_file():
        return p.read_text().strip() or None
    return None


def post(url: str, text: str, dry_run: bool = False) -> bool:
    """Post `text` to the Slack webhook. Returns True if it was sent."""
    if dry_run:
        print(f"--- would post ---\n{text}\n------------------")
        return False
    req = urllib.request.Request(
        url, data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception as exc:  # noqa: BLE001 - a failed alert must not kill the alerter
        logger.error("Slack post failed: %s", exc)
        return False
