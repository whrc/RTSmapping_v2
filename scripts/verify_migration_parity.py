"""Verify PDG -> abruptthawmapping parity, prefix by prefix.

Gate row 1 of `computing/pdg_migration.md` §5: object count and total bytes, source
versus destination, per prefix. This is the evidence that gates deletion, so it counts
what is actually in the buckets rather than trusting the Storage Transfer counters or
the §1 inventory table -- that table was found incomplete on 2026-08-28 (it omitted
`ee_staging/` and `interannual_inference/`).

Counting is done with the JSON API's `fields` projection, so only names and sizes come
back, never object bodies. A 41.7 M-object prefix is ~42 k pages and takes a while;
that is the cost of actually knowing.

**Acquisition prefixes must be measured against a FROZEN source.** `S2_RGB/` and
`global_quarterly/` are still being written by the export driver and Heidi's order loop.
A mismatch there means "the source moved", not "the copy is wrong" -- rerun after the
producer is stopped, which is what §5 row 1 means by the frozen Phase-C measurement.

Usage:
    python scripts/verify_migration_parity.py                 # every pair
    python scripts/verify_migration_parity.py --pair inference
    python scripts/verify_migration_parity.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from typing import Iterator

import google.auth
import google.auth.transport.requests
import requests

logger = logging.getLogger(__name__)

API = "https://storage.googleapis.com/storage/v1/b"

# (label, source bucket, source prefix, dest bucket, dest prefix, frozen_required)
PAIRS: list[tuple[str, str, str, str, str, bool]] = [
    ("planet",      "pdg-planet-data",     "global_quarterly/",
                    "rts-arctic-usw1",     "global_quarterly/",         True),
    ("s2",          "rts-mapping-v2-usw1", "S2_RGB/",
                    "rts-arctic-usw1",     "S2_RGB/",                   True),
    ("inference",   "rts-mapping-v2-usw1", "inference/",
                    "rts-arctic-usw1",     "inference/",                False),
    ("ee_staging",  "rts-mapping-v2-usw1", "ee_staging/",
                    "rts-arctic-usw1",     "ee_staging/",               False),
    ("interannual", "rts-mapping-v2-usw1", "interannual_inference/",
                    "rts-arctic-usw1",     "interannual_inference/",    False),
    ("experiments", "rts-mapping-v2",      "",
                    "rts-arctic-us",       "",                          False),
    ("ee_mirror",   "rts-mapping-v2-usc1", "ee_mirror/",
                    "rts-arctic-usc1",     "ee_mirror/",                False),
]


@dataclass
class Tally:
    """Object count and byte total for one prefix."""

    objects: int
    bytes: int


def _session() -> requests.Session:
    """Return a session carrying ADC bearer credentials.

    Uses ADC (not the gcloud CLI credential) because on a GCE VM the CLI defaults to
    the attached service account, which has no standing on the PDG buckets -- see
    `pdg_migration.md` §5c.
    """
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/devstorage.read_only"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {creds.token}"
    return s


def _pages(session: requests.Session, bucket: str, prefix: str) -> Iterator[dict]:
    """Yield each listing page for `bucket`/`prefix`, following pagination."""
    token = None
    while True:
        params = {
            "prefix": prefix,
            "maxResults": 1000,
            "fields": "items(size),nextPageToken",
        }
        if token:
            params["pageToken"] = token
        r = session.get(f"{API}/{bucket}/o", params=params, timeout=120)
        r.raise_for_status()
        page = r.json()
        yield page
        token = page.get("nextPageToken")
        if not token:
            return


def tally(session: requests.Session, bucket: str, prefix: str) -> Tally:
    """Count objects and sum bytes under a prefix."""
    objects = 0
    total = 0
    for n, page in enumerate(_pages(session, bucket, prefix), 1):
        items = page.get("items", [])
        objects += len(items)
        total += sum(int(i.get("size", 0)) for i in items)
        if n % 100 == 0:
            logger.info("  gs://%s/%s ... %d objects so far", bucket, prefix, objects)
    return Tally(objects=objects, bytes=total)


def main() -> int:
    """Compare every configured prefix pair and report parity."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", help="only this label")
    p.add_argument("--json", type=argparse.FileType("w"), help="write results here")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    session = _session()

    pairs = [x for x in PAIRS if not args.pair or x[0] == args.pair]
    if not pairs:
        logger.error("no pair named %r; known: %s", args.pair, ", ".join(x[0] for x in PAIRS))
        return 2

    results, failures = {}, 0
    print(f"\n{'pair':13s} {'src objects':>13s} {'dst objects':>13s} {'src bytes':>16s} {'dst bytes':>16s}  verdict")
    print("-" * 96)
    for label, sb, sp, db, dp, frozen in pairs:
        src = tally(session, sb, sp)
        dst = tally(session, db, dp)
        ok = src.objects == dst.objects and src.bytes == dst.bytes
        verdict = "MATCH" if ok else "MISMATCH"
        if not ok:
            failures += 1
            if frozen:
                verdict += " (live source - refreeze and rerun)"
        results[label] = {"source": f"gs://{sb}/{sp}", "dest": f"gs://{db}/{dp}",
                          "src": asdict(src), "dst": asdict(dst), "match": ok,
                          "frozen_required": frozen}
        print(f"{label:13s} {src.objects:13,d} {dst.objects:13,d} "
              f"{src.bytes:16,d} {dst.bytes:16,d}  {verdict}")

    if args.json:
        json.dump(results, args.json, indent=2)
        logger.info("wrote %s", args.json.name)

    print()
    if failures:
        logger.error("%d of %d pairs did not match", failures, len(pairs))
    else:
        logger.info("all %d pairs match", len(pairs))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
