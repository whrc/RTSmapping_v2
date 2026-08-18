"""Step 3 — order a year's quads from Planet, delivered straight into GCS.

Port of `3_order_basemaps.qmd` from HRodenhizer/circumpolar_planet_basemaps
@ initial-download, with the changes Heidi asked for in her PR #61 review
(2026-08-17):

* **Bounded retries** (her change 4). A 401 fails fast — auth is the one error
  retrying cannot fix. Transient statuses back off exponentially, five attempts;
  on exhaustion the quad is recorded in `failed_orders_<year>.csv` and the loop
  *continues*, because abandoning a five-day run over one quad is worse than
  finishing with a short retry list. `--retry-failed` then re-orders just those,
  reading the CSV so it skips the bucket listing entirely.
* **Faster listing** (her change 3). The prior-delivered scan requests only
  object names instead of full metadata, which is where most of the wall time
  in a ~1.9M-object listing goes.
* **No rename pass** (her changes 2 and 3). The rename existed so the bucket
  read tidily; `inference/quad_index.py` matches on the basename and lists
  recursively, so it indexes the raw delivery — order-UUID directories and all —
  identically. Dropping it removes the crash-recovery problem and the second
  slow listing. `tidy_rename.py` keeps it available as cosmetic clean-up.

Delivery is server-side: Planet writes into the bucket, nothing transits this
machine. This process only places orders, so it is API-bound and single-core.

Usage (normally via run_year.sh, which supervises restarts):
    python planetscope-download/order_basemaps.py --year 2022 \
        --grids planetscope-download/data/circumpolar_south_planet_basemap_grids_2022.geojson
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.quad_index import _QUAD_NAME_RE  # SSoT for "what is a quad object"  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from utils.watchdog import start_stall_watchdog  # noqa: E402

logger = logging.getLogger(__name__)

# Runtime outputs live OUTSIDE the repo: the checkout is owned by whoever cloned
# it, and collaborators get their own OS Login accounts with no write access to
# it. /mnt/outputs is world-writable on the VM. Override with PSD_WORK.
DEFAULT_WORK = Path(os.environ.get("PSD_WORK", "/mnt/outputs/planetscope-download"))

ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"
TIMEOUT = (10, 120)          # (connect, read) — never leave a socket unbounded
OK_STATUS = 202
RETRY_STATUS = {400, 409, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 5
BACKOFF_S = [30, 60, 120, 240]   # between the 5 attempts
STATUS_EVERY_S = 60


class Progress:
    """Thread-safe counters plus the status file the supervisor and Heidi read."""

    def __init__(self, year: int, n_total: int, status_path: Path):
        self.year, self.n_total, self.status_path = year, n_total, status_path
        self.ordered = self.skipped = self.failed = 0
        self.last_quad_id = ""
        self.started = time.time()
        self.last_active = [time.time()]   # bumped for the stall watchdog
        self._lock = threading.Lock()
        self._last_write = 0.0

    def record(self, kind: str, quad_id: str) -> None:
        """Count one finished quad and refresh the status file if due."""
        with self._lock:
            setattr(self, kind, getattr(self, kind) + 1)
            self.last_quad_id = quad_id
            self.last_active[0] = time.time()
            due = time.time() - self._last_write > STATUS_EVERY_S
            if due:
                self._last_write = time.time()
        if due:
            self.write()

    def snapshot(self) -> dict:
        done = self.ordered + self.skipped + self.failed
        elapsed = max(time.time() - self.started, 1e-6)
        rate = self.ordered / (elapsed / 60)
        remaining = self.n_total - done
        return {
            "year": self.year,
            "started_at": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
            "n_total": self.n_total, "n_done": done,
            "n_ordered": self.ordered, "n_skipped": self.skipped, "n_failed": self.failed,
            "pct_done": round(100 * done / self.n_total, 2) if self.n_total else 0.0,
            "last_quad_id": self.last_quad_id,
            "orders_per_min": round(rate, 1),
            "eta_hours": round(remaining / rate / 60, 1) if rate > 0 and remaining > 0 else 0.0,
        }

    def write(self) -> None:
        """Write the status JSON. Never fatal — monitoring must not kill the run."""
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.write_text(json.dumps(self.snapshot(), indent=1))
        except OSError as e:
            logger.warning("Could not write status file: %s", e)


def list_delivered(bucket_name: str, prefix: str, use_match_glob: bool) -> set[str]:
    """Return the quad ids already delivered under `prefix`.

    Requests only object names: the default listing returns full metadata for
    every object, which dominates wall time on a ~1.9M-object prefix. This runs
    once per invocation, so its cost is a slow resume, not a slow loop.

    Args:
        bucket_name: Delivery bucket, e.g. "pdg-planet-data".
        prefix: e.g. "global_quarterly/2022/q3/".
        use_match_glob: Also filter server-side to quad tifs. Unvalidated —
            GCS documents matchGlob as applying when delimiter is "/", which
            interacts with recursive listing, so it falls back on any error.

    Returns:
        Set of "<col>-<row>" ids.
    """
    from google.cloud import storage

    client = storage.Client()
    kwargs = {"prefix": prefix, "fields": "items(name),nextPageToken"}
    if use_match_glob:
        kwargs["match_glob"] = "**/*quad*.tif"

    t0 = time.time()
    try:
        blobs = list(client.list_blobs(bucket_name, **kwargs))
    except Exception as e:  # noqa: BLE001 - any API rejection falls back
        if not use_match_glob:
            raise
        logger.warning("match_glob rejected (%s) — falling back to a name-only listing", e)
        kwargs.pop("match_glob")
        blobs = list(client.list_blobs(bucket_name, **kwargs))

    ids = set()
    for blob in blobs:
        m = _QUAD_NAME_RE.search(blob.name.rsplit("/", 1)[-1])
        if m:
            ids.add(f"{m.group(1)}-{m.group(2)}")
    logger.info("Prior delivery: %d quads already present under %s (listing took %.0fs)",
                len(ids), prefix, time.time() - t0)
    return ids


def place_order(session: requests.Session, row, bucket_name: str,
                gcs_key: str) -> tuple[str, str]:
    """Place one quad's order, retrying transient failures with backoff.

    Args:
        session: Session carrying the Planet API key as basic auth.
        row: Row from the step-2 GeoDataFrame.
        bucket_name: Delivery bucket.
        gcs_key: GCS delivery credential. Sent to Planet by design so their
            servers can write into the bucket.

    Returns:
        (outcome, detail) where outcome is "ordered" or "failed".

    Raises:
        PermissionError: On 401. Auth cannot be retried into working, and under
            the typed-per-session key model the fix is a restart with a fresh
            key — which resumes where this stopped, since delivered quads are
            skipped.
    """
    quad_id = row["id"]
    order = {
        "name": quad_id,
        "source_type": "basemaps",
        "products": [{"mosaic_name": row["basemap_name"], "quad_ids": [quad_id]}],
        "delivery": {"google_cloud_storage": {
            "bucket": bucket_name, "credentials": gcs_key,
            "path_prefix": row["delivery_location"]}},
        "tools": [{"file_format": {"format": "COG"}}],
    }

    last = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            res = session.post(ORDERS_URL, json=order, timeout=TIMEOUT)
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if res.status_code == OK_STATUS:
                return "ordered", ""
            if res.status_code == 401:
                raise PermissionError(
                    f"401 Unauthorized on quad {quad_id}. The Planet API key is "
                    f"invalid or expired — restart with a fresh key; already-delivered "
                    f"quads are skipped, so the run resumes where it stopped.")
            last = f"HTTP {res.status_code}: {res.text[:200]}"
            if res.status_code not in RETRY_STATUS:
                return "failed", last
        if attempt < MAX_ATTEMPTS - 1:
            delay = BACKOFF_S[attempt]
            logger.warning("quad %s attempt %d/%d failed (%s) — retrying in %ds",
                           quad_id, attempt + 1, MAX_ATTEMPTS, last, delay)
            time.sleep(delay)

    return "failed", last


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--grids", type=Path, required=True, help="step 2 output")
    p.add_argument("--bucket", default="pdg-planet-data")
    p.add_argument("--status-dir", type=Path, default=DEFAULT_WORK / "status")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent order placements. 1 = serial (default). Raise only "
                        "for the threading experiment; Planet may be the ceiling.")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N quads (threading experiment / smoke runs).")
    p.add_argument("--stall-timeout-s", type=float, default=900.0,
                   help="Exit 3 if no order completes in this long, for a supervised "
                        "restart. 0 disables.")
    p.add_argument("--use-match-glob", action="store_true",
                   help="Server-side glob on the prior-delivery listing (unvalidated).")
    p.add_argument("--retry-failed", type=Path, default=None,
                   help="Re-order only the quads in this failed_orders CSV, skipping "
                        "the bucket listing.")
    args = p.parse_args()
    setup_logging()

    api_key = os.environ.get("PL_BM_API_KEY")
    gcs_key = os.environ.get("PDG_PL_ORDERS_KEY")
    if not api_key or not gcs_key:
        logger.error("PL_BM_API_KEY and PDG_PL_ORDERS_KEY must both be set — "
                     "run through run_year.sh, which prompts for them.")
        return 2

    grids = gpd.read_file(args.grids)
    prefix = f"global_quarterly/{args.year}/q3/"

    if args.retry_failed:
        want = {r["quad_id"] for r in csv.DictReader(args.retry_failed.open())}
        grids = grids[grids["id"].isin(want)]
        delivered: set[str] = set()
        logger.info("Retry mode: %d previously failed quads", len(grids))
    else:
        delivered = list_delivered(args.bucket, prefix, args.use_match_glob)

    if args.limit:
        grids = grids.head(args.limit)

    failed_path = args.status_dir / f"failed_orders_{args.year}.csv"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    progress = Progress(args.year, len(grids), args.status_dir / f"{args.year}.json")
    stop_watchdog = start_stall_watchdog(
        progress.last_active, args.stall_timeout_s, f"orders {args.year}")

    # One session per thread: requests.Session is not safe to share across
    # concurrent requests, which only bites in the --workers experiment.
    local = threading.local()

    def session() -> requests.Session:
        if not hasattr(local, "s"):
            local.s = requests.Session()
            local.s.auth = (api_key, "")
        return local.s

    failures: list[dict] = []
    failures_lock = threading.Lock()

    def handle(row) -> None:
        if row["id"] in delivered:
            progress.record("skipped", row["id"])
            return
        outcome, detail = place_order(session(), row, args.bucket, gcs_key)
        if outcome == "failed":
            with failures_lock:
                failures.append({"quad_id": row["id"], "detail": detail})
        progress.record(outcome, row["id"])

    logger.info("Ordering %d quads for %dq3 -> gs://%s/%s (workers=%d)",
                len(grids), args.year, args.bucket, prefix, args.workers)
    rows = [r for _, r in grids.iterrows()]
    t0 = time.time()
    try:
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(handle, rows))
        else:
            for row in rows:
                handle(row)
    except PermissionError as e:
        logger.error("%s", e)
        progress.write()
        return 2
    finally:
        stop_watchdog()

    if failures:
        with failed_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["quad_id", "detail"])
            w.writeheader()
            w.writerows(failures)
        logger.warning("%d quads failed after %d attempts each — recorded in %s. "
                       "Re-run with --retry-failed %s to sweep them up.",
                       len(failures), MAX_ATTEMPTS, failed_path, failed_path)

    progress.write()
    elapsed = time.time() - t0
    logger.info("Done in %.1f h: %d ordered, %d already present, %d failed (%.1f orders/min)",
                elapsed / 3600, progress.ordered, progress.skipped, progress.failed,
                progress.ordered / max(elapsed / 60, 1e-6))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
