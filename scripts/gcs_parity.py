"""Compare a GCS source prefix against its migrated copy (computing/pdg_migration.md §5).

This is the evidence that gates deletion, so it checks *every* object rather than a
sample: GCS lists lexicographically by name, and a correct copy has the same names
below its prefix, so the two listings can be walked in lockstep and compared
tuple-by-tuple. That is full coverage in constant memory — the largest leg is
~41.7M objects, which no sampling scheme could meaningfully cover and no in-memory
diff could hold.

Usage:
    python scripts/gcs_parity.py \
        --src  gs://rts-mapping-v2-usw1/inference/2025q3_south/probs \
        --dst  gs://rts-arctic-usw1/inference/2025q3_south/probs

Exit code is 0 only when the two sides hold exactly the same object names, sizes
and MD5s. Needs GOOGLE_CLOUD_PROJECT set when running on bare ADC (the storage
client requires a quota project).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google.cloud import storage  # noqa: E402

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

MAX_REPORTED = 20


class Entry(NamedTuple):
    """One object, named relative to the prefix being compared."""

    name: str
    size: int
    md5: str


class Result(NamedTuple):
    """Outcome of a lockstep comparison."""

    src_count: int
    src_bytes: int
    dst_count: int
    dst_bytes: int
    missing: list[str]     # in source, absent from destination
    extra: list[str]       # in destination, absent from source
    differing: list[str]   # present in both, different size or MD5

    @property
    def ok(self) -> bool:
        return not (self.missing or self.extra or self.differing)


def split_uri(uri: str) -> tuple[str, str]:
    """Split a ``gs://bucket/prefix`` URI.

    Args:
        uri: A ``gs://`` URI. A trailing slash is ignored.

    Returns:
        ``(bucket, prefix)``, where prefix may be an empty string.

    Raises:
        ValueError: If the URI does not start with ``gs://``.
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"not a gs:// URI: {uri}")
    bucket, _, prefix = uri[len("gs://"):].partition("/")
    return bucket, prefix.rstrip("/")


def entries(client: storage.Client, uri: str) -> Iterator[Entry]:
    """Stream a prefix as relative-named entries, in listing (lexicographic) order.

    Args:
        client: An authenticated storage client.
        uri: ``gs://bucket/prefix`` to walk.

    Yields:
        One :class:`Entry` per object, its name relative to the prefix so that two
        prefixes at different absolute paths are directly comparable. Directory
        placeholder objects are skipped.
    """
    bucket, prefix = split_uri(uri)
    # name/size/md5Hash only: the default projection triples the listing cost.
    blobs = client.list_blobs(
        bucket, prefix=f"{prefix}/" if prefix else None,
        fields="items(name,size,md5Hash),nextPageToken",
    )
    cut = len(prefix) + 1 if prefix else 0
    for blob in blobs:
        if blob.name.endswith("/"):  # directory placeholder, not an object
            continue
        yield Entry(blob.name[cut:], blob.size or 0, blob.md5_hash or "")


def compare(src: Iterator[Entry], dst: Iterator[Entry]) -> Result:
    """Walk two sorted entry streams in lockstep and diff them.

    Both GCS listings are lexicographic by name, so a single pass with one lookahead
    each is enough — no side is ever held in memory.

    Args:
        src: Entries from the original prefix.
        dst: Entries from the copy.

    Returns:
        A :class:`Result`. Divergence lists are capped at :data:`MAX_REPORTED` names
        for legibility; the counts still reflect everything seen.
    """
    missing: list[str] = []
    extra: list[str] = []
    differing: list[str] = []
    src_count = src_bytes = dst_count = dst_bytes = 0

    a = next(src, None)
    b = next(dst, None)
    while a is not None or b is not None:
        if b is None or (a is not None and a.name < b.name):
            src_count += 1; src_bytes += a.size
            if len(missing) < MAX_REPORTED:
                missing.append(a.name)
            a = next(src, None)
        elif a is None or b.name < a.name:
            dst_count += 1; dst_bytes += b.size
            if len(extra) < MAX_REPORTED:
                extra.append(b.name)
            b = next(dst, None)
        else:
            src_count += 1; src_bytes += a.size
            dst_count += 1; dst_bytes += b.size
            if (a.size, a.md5) != (b.size, b.md5) and len(differing) < MAX_REPORTED:
                differing.append(a.name)
            a = next(src, None)
            b = next(dst, None)

    return Result(src_count, src_bytes, dst_count, dst_bytes,
                  missing, extra, differing)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="gs:// URI of the original")
    p.add_argument("--dst", required=True, help="gs:// URI of the copy")
    args = p.parse_args()
    setup_logging()

    client = storage.Client()
    logger.info("Comparing %s -> %s (every object)", args.src, args.dst)
    result = compare(entries(client, args.src), entries(client, args.dst))

    logger.info("%-12s %15s %20s", "", "objects", "bytes")
    logger.info("%-12s %15d %20d", "source", result.src_count, result.src_bytes)
    logger.info("%-12s %15d %20d", "destination", result.dst_count, result.dst_bytes)

    for label, names in (("MISSING at destination", result.missing),
                         ("EXTRA at destination", result.extra),
                         ("DIFFERING size/MD5", result.differing)):
        if names:
            logger.error("%s: %d shown%s", label, len(names),
                         " (capped)" if len(names) == MAX_REPORTED else "")
            for name in names:
                logger.error("    %s", name)

    logger.info("PARITY %s", "PASS" if result.ok else "FAIL")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
