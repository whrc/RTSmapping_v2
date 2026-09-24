"""GCS-as-queue shard claiming for the dual-fleet inference run (plan Phase 1).

The pan-Arctic tile list is split into many spatially-contiguous shards
(`scripts/shard_tiles.py`). Every worker (one per GPU, across the heterogeneous
A100 + L4 fleets) repeatedly claims the next free shard, processes it, and marks
it done. Fast GPUs naturally take more shards, slow ones fewer — the queue
self-balances with no central scheduler and no extra service.

The queue is just objects under a base prefix:

    <base>/claims/<shard_id>   one per in-progress shard (atomic create-if-absent)
    <base>/done/<shard_id>     one per finished shard

A claim is won with an `if_generation_match=0` upload (create-only): GCS lets
exactly one writer create the object, so two workers can never own one shard.
A crashed worker leaves a stale claim; a claim whose heartbeat is older than
`stale_after_s` (and that has no `done` marker) is reclaimable — any worker may
delete it and re-claim. `done` markers are the source of truth on restart.

This module holds no GPU/torch deps and isolates all GCS calls behind a small
`bucket` object so it is unit-testable with a fake bucket (no network).
"""

from __future__ import annotations

import json
import logging
import socket
import time
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)


def default_worker_id() -> str:
    """Stable-ish id for this worker process: <host>:<pid>:<gpu-or-cpu>."""
    import os

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
    return f"{socket.gethostname()}:{os.getpid()}:gpu{gpu}"


class ClaimStore:
    """Atomic shard queue backed by objects under ``<base_prefix>/`` in a bucket.

    Args:
        bucket: a ``google.cloud.storage.Bucket`` (or a test double exposing
            ``blob(name)`` and ``list_blobs(prefix=...)``).
        base_prefix: key prefix under the bucket, e.g.
            ``"inference/2025q3_south"`` (no leading/trailing slash needed).
        worker_id: identifier written into each claim; defaults to host:pid:gpu.
        now_fn: clock injection for tests (seconds since epoch).
    """

    def __init__(self, bucket, base_prefix: str, worker_id: Optional[str] = None,
                 now_fn: Callable[[], float] = time.time) -> None:
        self.bucket = bucket
        self.base = base_prefix.strip("/")
        self.worker_id = worker_id or default_worker_id()
        self._now = now_fn

    # --- key helpers ------------------------------------------------------
    def _claim_key(self, shard_id: str) -> str:
        return f"{self.base}/claims/{shard_id}"

    def _done_key(self, shard_id: str) -> str:
        return f"{self.base}/done/{shard_id}"

    # --- primitives -------------------------------------------------------
    def try_claim(self, shard_id: str) -> bool:
        """Attempt to atomically claim ``shard_id``. Returns True iff won.

        Uses create-if-absent (``if_generation_match=0``); a PreconditionFailed
        means another worker already holds the claim.
        """
        from google.api_core.exceptions import PreconditionFailed

        payload = json.dumps({
            "worker_id": self.worker_id,
            "claimed_at": self._now(),
            "heartbeat_at": self._now(),
        })
        blob = self.bucket.blob(self._claim_key(shard_id))
        try:
            blob.upload_from_string(payload, if_generation_match=0)
            return True
        except PreconditionFailed:
            return False

    def heartbeat(self, shard_id: str) -> None:
        """Refresh this worker's claim so a long shard is not seen as stale.

        Overwrites the claim (unconditional) only if we still own it; a no-op if
        another worker has reclaimed it (we lost the race and should stop).
        """
        blob = self.bucket.blob(self._claim_key(shard_id))
        owner, _ = self._read_claim(blob)
        if owner != self.worker_id:
            return
        blob.upload_from_string(json.dumps({
            "worker_id": self.worker_id,
            "heartbeat_at": self._now(),
        }))

    def mark_done(self, shard_id: str) -> None:
        """Write the ``done`` marker (source of truth) and drop the claim."""
        self.bucket.blob(self._done_key(shard_id)).upload_from_string(
            json.dumps({"worker_id": self.worker_id, "done_at": self._now()}))
        try:
            self.bucket.blob(self._claim_key(shard_id)).delete()
        except Exception:  # noqa: BLE001 - claim cleanup is best-effort
            pass

    # --- listings ---------------------------------------------------------
    def _list_ids(self, kind: str) -> set[str]:
        prefix = f"{self.base}/{kind}/"
        out: set[str] = set()
        for blob in self.bucket.list_blobs(prefix=prefix):
            name = blob.name[len(prefix):]
            if name:  # skip the directory placeholder, if any
                out.add(name)
        return out

    def done_ids(self) -> set[str]:
        return self._list_ids("done")

    def claimed_ids(self) -> set[str]:
        return self._list_ids("claims")

    # --- stale-claim reclaim ---------------------------------------------
    def _read_claim(self, blob) -> tuple[Optional[str], Optional[float]]:
        """Return (worker_id, heartbeat_at) for a claim blob, or (None, None)."""
        try:
            data = json.loads(blob.download_as_text())
        except Exception:  # noqa: BLE001 - missing/corrupt claim
            return None, None
        return data.get("worker_id"), data.get("heartbeat_at", data.get("claimed_at"))

    def reclaim_if_stale(self, shard_id: str, stale_after_s: float) -> bool:
        """Delete ``shard_id``'s claim iff its heartbeat is older than the TTL.

        Returns True if a stale claim was deleted (so the caller may retry the
        claim). False if the claim is fresh, absent, or unreadable.

        Only the lost-the-race case is swallowed. Anything else — above all a
        403 from a runtime identity without ``storage.objects.delete`` — must
        surface: a blanket ``except`` here let the review campaign run seven
        weeks with its claim TTL silently dead (`review_campaign.md` §6.2).
        """
        from google.api_core.exceptions import NotFound

        blob = self.bucket.blob(self._claim_key(shard_id))
        _, heartbeat = self._read_claim(blob)
        if heartbeat is None:
            return False
        if self._now() - heartbeat <= stale_after_s:
            return False
        try:
            blob.delete()
            logger.warning("reclaimed stale shard %s (heartbeat %.0fs old)",
                           shard_id, self._now() - heartbeat)
            return True
        except NotFound:  # another worker deleted it first
            return False

    # --- the worker entry point ------------------------------------------
    def claim_next(self, shard_ids: Iterable[str], stale_after_s: float = 1800.0
                   ) -> Optional[str]:
        """Return the first shard this worker can claim, or None if all are done.

        Skips shards already ``done``; tries to claim each remaining shard; if a
        shard is held by someone else, reclaims it only when its heartbeat is
        stale and then retries the claim once.
        """
        done = self.done_ids()
        for sid in shard_ids:
            if sid in done:
                continue
            if self.try_claim(sid):
                return sid
            if self.reclaim_if_stale(sid, stale_after_s) and self.try_claim(sid):
                return sid
        return None
