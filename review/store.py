"""Campaign state: the queue, the claims, the verdicts.

A thin layer over `inference.claim.ClaimStore` (reused unchanged — same
`if_generation_match=0` atomicity and heartbeat that ran the pan-Arctic
inference queue, with a far longer claim TTL: see :data:`STALE_AFTER_S`) plus
the manifest and the verdict objects. Holds
**no web dependencies**, so the whole protocol is unit-testable against the
fake bucket in `tests/test_claim.py`.

Spec: `post-inference/review_campaign.md` §5–§6, §9.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Iterable, Optional

import pandas as pd

from inference.claim import ClaimStore

logger = logging.getLogger(__name__)

VERDICTS = ("rts", "false", "unsure")

# How long a claimed batch stays claimed, in SECONDS, measured from the last
# heartbeat (`inference.claim.ClaimStore.reclaim_if_stale`).
#
# The inference queue used 30 min because a crashed worker's shard was pure
# loss. A reviewer is not a crashed worker: their part-rated batch sits in the
# browser's localStorage and is still good the next morning, so a short TTL
# would hand those 200 polygons to a second reviewer and have them rated twice.
# One week is long enough to cover a weekend and a holiday, while still letting
# a truly abandoned batch return to the pool on its own instead of needing the
# manual release in `review_campaign.md` §6.1.
#
# Changing this needs a redeploy, not just an edit — the app runs from a
# container image (`scripts/deploy_review_vm.sh`).
STALE_AFTER_S = 7 * 24 * 3600.0  # one week


class ReviewStore:
    """Queue + verdict store for one review campaign.

    Args:
        bucket: a ``google.cloud.storage.Bucket`` (or a test double exposing
            ``blob(name)`` and ``list_blobs(prefix=...)``).
        manifest: the campaign manifest from
            `scripts/build_review_manifest.py`.
        base_prefix: key prefix for campaign state, e.g.
            ``"inference/2025q3_south/review"``.
        crop_prefix: key prefix of the rendered crop archive.
    """

    def __init__(self, bucket, manifest: pd.DataFrame, base_prefix: str,
                 crop_prefix: str) -> None:
        self.bucket = bucket
        self.manifest = manifest
        self.base = base_prefix.strip("/")
        self.crop_prefix = crop_prefix.strip("/")
        self._by_batch = {bid: grp.sort_values("seq")
                          for bid, grp in manifest.groupby("batch_id")}
        # Ascending batch id == descending max_prob (§3).
        self.batch_ids = sorted(self._by_batch)

    # --- claiming ---------------------------------------------------------
    def _claims(self, reviewer: str) -> ClaimStore:
        return ClaimStore(self.bucket, self.base, worker_id=reviewer)

    def done_ids(self) -> set[str]:
        """Batch ids with a ``done`` marker — the source of truth on restart."""
        return self._claims("read-only").done_ids()

    def claim_next(self, reviewer: str) -> Optional[str]:
        """Claim the first unfinished batch in queue order, or None if done."""
        bid = self._claims(reviewer).claim_next(self.batch_ids, STALE_AFTER_S)
        if bid:
            logger.info("reviewer %s claimed %s", reviewer, bid)
        return bid

    def heartbeat(self, reviewer: str, batch_id: str) -> None:
        """Record that this batch is still being worked on.

        This pushes the batch's expiry out by another :data:`STALE_AFTER_S`,
        but at a one-week TTL that hardly matters — an open tab beats every
        60 s, and no rating session runs for a week. Its real value is the
        timestamp: it is the only record of *when* a held batch was last
        touched, which is what you look at before releasing one by hand.
        """
        self._claims(reviewer).heartbeat(batch_id)

    # --- items ------------------------------------------------------------
    def has_batch(self, batch_id: str) -> bool:
        return batch_id in self._by_batch

    def batch_items(self, batch_id: str) -> list[dict]:
        """The batch's items in presentation order, with crop object keys."""
        grp = self._by_batch[batch_id]
        return [{
            "rts_id": int(r.rts_id),
            "max_prob": float(r.max_prob),
            "conf_class": str(r.conf_class),
            "area_m2": float(r.area_m2),
            "lat": float(r.centroid_lat),
            "lon": float(r.centroid_lon),
            "tight_key": f"{self.crop_prefix}/{int(r.rts_id)}_t.jpg",
            "wide_key": f"{self.crop_prefix}/{int(r.rts_id)}_w.jpg",
            # Same imagery without the burned-in outline, for the UI's toggle.
            "tight_plain_key": f"{self.crop_prefix}/{int(r.rts_id)}_t_plain.jpg",
            "wide_plain_key": f"{self.crop_prefix}/{int(r.rts_id)}_w_plain.jpg",
        } for r in grp.itertuples()]

    # --- verdicts ---------------------------------------------------------
    def _verdict_key(self, batch_id: str) -> str:
        return f"{self.base}/verdicts/{batch_id}.jsonl"

    def submit(self, reviewer: str, batch_id: str,
               verdicts: dict[int, str]) -> bool:
        """Write a completed batch's verdicts and release the claim.

        Idempotent: submitting a batch that is already ``done`` is accepted and
        ignored, so a duplicate click or a retried request cannot corrupt the
        record.

        Args:
            reviewer: who rated them.
            batch_id: the claimed batch.
            verdicts: ``{rts_id: verdict}``; every id must belong to the batch
                and every verdict must be in :data:`VERDICTS`.

        Returns:
            True if this call wrote the batch, False if it was already done.

        Raises:
            ValueError: on an unknown batch, an unknown id, or a bad verdict.
        """
        if batch_id not in self._by_batch:
            raise ValueError(f"unknown batch {batch_id}")
        claims = self._claims(reviewer)
        if batch_id in claims.done_ids():
            logger.warning("batch %s already submitted — ignoring", batch_id)
            return False

        valid = set(self._by_batch[batch_id]["rts_id"].astype(int))
        unknown = set(verdicts) - valid
        if unknown:
            raise ValueError(f"ids not in batch {batch_id}: {sorted(unknown)[:5]}")
        # A partial submit would mark the batch done, so its unrated polygons
        # would never be re-served while `progress` counted them as reviewed —
        # coverage loss that nothing downstream could detect.
        incomplete = valid - set(verdicts)
        if incomplete:
            raise ValueError(f"batch {batch_id} is incomplete: "
                             f"{len(incomplete)} of {len(valid)} items unrated")
        bad = {v for v in verdicts.values() if v not in VERDICTS}
        if bad:
            raise ValueError(f"invalid verdicts: {sorted(bad)}")

        injected = set(self._by_batch[batch_id]
                       .loc[lambda d: d["injected"], "rts_id"].astype(int))
        now = time.time()
        lines = [json.dumps({
            "rts_id": int(i), "verdict": v, "reviewer": reviewer,
            "batch_id": batch_id, "injected": int(i) in injected,
            "reviewed_at": now,
        }) for i, v in sorted(verdicts.items())]
        self.bucket.blob(self._verdict_key(batch_id)).upload_from_string(
            "\n".join(lines))
        claims.mark_done(batch_id)
        logger.info("reviewer %s submitted %s (%d verdicts)", reviewer,
                    batch_id, len(verdicts))
        return True

    def read_verdicts(self, batch_ids: Optional[Iterable[str]] = None
                      ) -> pd.DataFrame:
        """All submitted verdicts as a DataFrame (empty if none yet)."""
        prefix = f"{self.base}/verdicts/"
        wanted = set(batch_ids) if batch_ids is not None else None
        rows = []
        for blob in self.bucket.list_blobs(prefix=prefix):
            bid = blob.name[len(prefix):].removesuffix(".jsonl")
            if not bid or (wanted is not None and bid not in wanted):
                continue
            for line in blob.download_as_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        if not rows:
            return pd.DataFrame(columns=["rts_id", "verdict", "reviewer",
                                         "batch_id", "injected", "reviewed_at"])
        return pd.DataFrame(rows)

    # --- progress ---------------------------------------------------------
    def progress(self) -> dict:
        """Campaign state for the UI and the merge report (§9).

        ``headline_max_prob`` is the honest form of the campaign's claim: the
        lowest ``max_prob`` such that *every* polygon at or above it has been
        reviewed. It walks the batch order and stops at the first gap, so
        batches finished out of order do not inflate it.
        """
        done = self.done_ids()
        prefix_batches = []
        for bid in self.batch_ids:
            if bid not in done:
                break
            prefix_batches.append(bid)

        cov = self.manifest[~self.manifest["injected"]]
        done_cov = cov[cov["batch_id"].isin(done)]
        headline = (float(cov[cov["batch_id"].isin(prefix_batches)]
                          ["max_prob"].min())
                    if prefix_batches else None)
        return {
            "batches_total": len(self.batch_ids),
            "batches_done": len(done),
            "batches_contiguous": len(prefix_batches),
            "headline_max_prob": headline,
            "items_total": int(len(cov)),
            "items_done": int(len(done_cov)),
            "area_km2_total": float(cov["area_m2"].sum() / 1e6),
            "area_km2_done": float(done_cov["area_m2"].sum() / 1e6),
        }
