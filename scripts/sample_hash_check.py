"""Spot-check migrated objects by MD5 — gate row 2 of `computing/pdg_migration.md` §5.

Parity (row 1) proves the object *counts* and byte *totals* agree. That is not the same as
the bytes being right: a truncated or swapped object can preserve both. This samples
objects at random per leg and compares the MD5 GCS stores for each side.

Random, not first-N: listings are lexicographic, so the first N objects of a 42 M-object
prefix all come from one corner of the keyspace and would miss a fault anywhere else.
Reservoir sampling over the full listing gives a uniform sample in one pass without
holding 42 M keys in memory.

Usage:
    python scripts/sample_hash_check.py                  # 200 per leg
    python scripts/sample_hash_check.py --n 50 --pair inference
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Iterator

import google.auth
import google.auth.transport.requests

sys.path.insert(0, str(Path(__file__).resolve().parent))  # run from anywhere, not just scripts/
from verify_migration_parity import PAIRS, _session, API  # noqa: E402

logger = logging.getLogger(__name__)

SEED = 42  # CLAUDE.md: reproducible by default


def _iter_keys(session, bucket: str, prefix: str) -> Iterator[tuple[str, str]]:
    """Yield (name, md5Hash) for every object under a prefix."""
    token = None
    while True:
        params = {"prefix": prefix, "maxResults": 1000,
                  "fields": "items(name,md5Hash),nextPageToken"}
        if token:
            params["pageToken"] = token
        r = session.get(f"{API}/{bucket}/o", params=params, timeout=120)
        r.raise_for_status()
        page = r.json()
        for item in page.get("items", []):
            yield item["name"], item.get("md5Hash", "")
        token = page.get("nextPageToken")
        if not token:
            return


def sample(session, bucket: str, prefix: str, n: int) -> dict[str, str]:
    """Reservoir-sample `n` objects uniformly from a prefix, returning {name: md5}."""
    rng = random.Random(SEED)
    res: list[tuple[str, str]] = []
    for i, item in enumerate(_iter_keys(session, bucket, prefix)):
        if i < n:
            res.append(item)
        else:
            j = rng.randint(0, i)
            if j < n:
                res[j] = item
        if i and i % 500_000 == 0:
            logger.info("  ... scanned %d objects", i)
    return dict(res)


def md5_of(session, bucket: str, key: str) -> str | None:
    """Fetch one object's stored MD5, or None if it is absent."""
    from urllib.parse import quote
    r = session.get(f"{API}/{bucket}/o/{quote(key, safe='')}",
                    params={"fields": "md5Hash"}, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("md5Hash")


def main() -> int:
    """Sample each leg and compare MD5s across the copy."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="objects per leg (default 200)")
    p.add_argument("--pair", help="only this label")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    session = _session()

    pairs = [x for x in PAIRS if not args.pair or x[0] == args.pair]
    bad = 0
    for label, sb, sp, db, dp, _frozen in pairs:
        logger.info("sampling %s from gs://%s/%s", label, sb, sp)
        picked = sample(session, sb, sp, args.n)
        if not picked:
            print(f"  {label:13s} EMPTY — nothing to sample")
            continue
        mism = missing = 0
        for name, src_md5 in picked.items():
            dst_key = name if sp == dp else dp + name[len(sp):]
            dst_md5 = md5_of(session, db, dst_key)
            if dst_md5 is None:
                missing += 1
                logger.error("  MISSING at destination: %s", dst_key)
            elif dst_md5 != src_md5:
                mism += 1
                logger.error("  MD5 DIFFERS: %s  src=%s dst=%s", name, src_md5, dst_md5)
        ok = len(picked) - mism - missing
        verdict = "PASS" if (mism == 0 and missing == 0) else "FAIL"
        bad += mism + missing
        print(f"  {label:13s} {len(picked):4d} sampled  {ok:4d} match  "
              f"{mism:3d} differ  {missing:3d} missing  {verdict}")

    print()
    if bad:
        logger.error("%d object(s) failed the hash check", bad)
    else:
        logger.info("every sampled object matched")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
