"""Unit tests for review.store.ReviewStore — the campaign's queue correctness.

Network-free: reuses the FakeBucket from `tests/test_claim.py`, which emulates
GCS create-if-absent atomicity. Covers the invariants a multi-reviewer campaign
depends on: two reviewers never get one batch, a claim outlives any plausible
rating session but not a week, a submitted batch is never re-served, submission
is idempotent, and the headline progress claim only counts a contiguous prefix.

Spec: `post-inference/review_campaign.md` §6, §9.
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from review.store import STALE_AFTER_S, ReviewStore
from tests.test_claim import _FakeBucket


def _manifest(n_batches: int = 3, per_batch: int = 4) -> pd.DataFrame:
    """A tiny manifest: max_prob descends across batches, one injected item."""
    rows = []
    rts_id = 1
    for b in range(n_batches):
        for s in range(per_batch):
            rows.append({
                "rts_id": rts_id, "batch_id": f"b{b:05d}", "seq": s,
                "injected": False, "max_prob": 1.0 - 0.1 * b,
                "conf_class": "high", "area_m2": 1000.0,
                "centroid_lat": 70.0, "centroid_lon": -120.0,
            })
            rts_id += 1
    # One replicate of item 1, injected into the last batch.
    rows.append({"rts_id": 1, "batch_id": f"b{n_batches - 1:05d}",
                 "seq": per_batch, "injected": True, "max_prob": 1.0,
                 "conf_class": "high", "area_m2": 1000.0,
                 "centroid_lat": 70.0, "centroid_lon": -120.0})
    return pd.DataFrame(rows)


@pytest.fixture
def store() -> ReviewStore:
    return ReviewStore(_FakeBucket(), _manifest(), "campaign/review",
                       "campaign/crops")


# --- claiming -------------------------------------------------------------
def test_two_reviewers_never_get_the_same_batch(store):
    a = store.claim_next("ann")
    b = store.claim_next("bob")
    assert a == "b00000" and b == "b00001"


def test_queue_is_served_in_probability_order(store):
    assert [store.claim_next(f"r{i}") for i in range(3)] == \
        ["b00000", "b00001", "b00002"]


def test_exhausted_queue_returns_none(store):
    for i in range(3):
        store.claim_next(f"r{i}")
    assert store.claim_next("late") is None


def test_a_claim_outlives_any_rating_session(store):
    """A batch left overnight is still its reviewer's in the morning.

    The inference queue reclaimed after 30 min; a reviewer must not have their
    part-rated batch handed to someone else, because their verdicts live in the
    browser and would then be rated twice. Two days is far beyond any plausible
    session and well inside the one-week TTL.
    """
    store.claim_next("ann")
    two_days_ago = time.time() - 2 * 24 * 3600
    store.bucket.blob("campaign/review/claims/b00000").upload_from_string(
        json.dumps({"worker_id": "ann", "heartbeat_at": two_days_ago}))
    assert store.claim_next("bob") == "b00001"


def test_a_claim_older_than_the_ttl_returns_to_the_pool(store):
    """Past one week the batch is presumed abandoned and re-served."""
    store.claim_next("ann")
    expired = time.time() - STALE_AFTER_S - 60
    store.bucket.blob("campaign/review/claims/b00000").upload_from_string(
        json.dumps({"worker_id": "ann", "heartbeat_at": expired}))
    assert store.claim_next("bob") == "b00000"


def test_the_ttl_is_one_week_in_seconds(store):
    """The unit is seconds — a wrong unit here silently steals live batches."""
    assert STALE_AFTER_S == 604800.0


def test_a_released_claim_is_re_servable(store):
    """The manual escape hatch: deleting the claim object frees the batch."""
    store.claim_next("ann")
    store.bucket.blob("campaign/review/claims/b00000").delete()
    assert store.claim_next("bob") == "b00000"


def test_fresh_claim_is_not_stolen(store):
    store.claim_next("ann")
    assert store.claim_next("bob") == "b00001"


# --- submission -----------------------------------------------------------
def test_submitted_batch_is_never_re_served(store):
    store.claim_next("ann")
    ids = [i["rts_id"] for i in store.batch_items("b00000")]
    store.submit("ann", "b00000", {i: "rts" for i in ids})
    assert store.claim_next("bob") == "b00001"


def test_submit_is_idempotent(store):
    ids = [i["rts_id"] for i in store.batch_items("b00000")]
    assert store.submit("ann", "b00000", {i: "rts" for i in ids}) is True
    assert store.submit("ann", "b00000", {i: "false" for i in ids}) is False
    verdicts = store.read_verdicts()
    assert len(verdicts) == len(ids)
    assert set(verdicts["verdict"]) == {"rts"}  # the retry did not overwrite


def test_submit_rejects_an_id_from_another_batch(store):
    with pytest.raises(ValueError, match="not in batch"):
        store.submit("ann", "b00000", {999: "rts"})


def test_submit_rejects_an_unknown_verdict(store):
    ids = [i["rts_id"] for i in store.batch_items("b00000")]
    payload = {i: "rts" for i in ids}
    payload[ids[0]] = "maybe"
    with pytest.raises(ValueError, match="invalid verdicts"):
        store.submit("ann", "b00000", payload)


def test_submit_rejects_an_unknown_batch(store):
    with pytest.raises(ValueError, match="unknown batch"):
        store.submit("ann", "b99999", {})


def test_injected_items_are_flagged_in_the_record(store):
    items = store.batch_items("b00002")
    store.submit("ann", "b00002", {i["rts_id"]: "rts" for i in items})
    v = store.read_verdicts()
    # rts_id 1 appears in b00002 only as the injected replicate.
    assert bool(v.loc[v["rts_id"] == 1, "injected"].iloc[0]) is True
    assert not v.loc[v["rts_id"] != 1, "injected"].any()


# --- items ----------------------------------------------------------------
def test_batch_items_are_in_presentation_order_with_crop_keys(store):
    items = store.batch_items("b00000")
    assert [i["rts_id"] for i in items] == [1, 2, 3, 4]
    assert items[0]["tight_key"] == "campaign/crops/1_t.jpg"
    assert items[0]["wide_key"] == "campaign/crops/1_w.jpg"
    # The outline toggle's second copy of the same imagery.
    assert items[0]["tight_plain_key"] == "campaign/crops/1_t_plain.jpg"
    assert items[0]["wide_plain_key"] == "campaign/crops/1_w_plain.jpg"


# --- progress -------------------------------------------------------------
def test_headline_counts_only_a_contiguous_prefix(store):
    """Finishing a later batch must not claim the probabilities in between."""
    later = [i["rts_id"] for i in store.batch_items("b00001")]
    store.submit("ann", "b00001", {i: "rts" for i in later})
    p = store.progress()
    assert p["batches_done"] == 1
    assert p["batches_contiguous"] == 0
    assert p["headline_max_prob"] is None

    first = [i["rts_id"] for i in store.batch_items("b00000")]
    store.submit("ann", "b00000", {i: "rts" for i in first})
    p = store.progress()
    assert p["batches_contiguous"] == 2
    assert p["headline_max_prob"] == pytest.approx(0.9)


def test_progress_counts_coverage_items_not_injected_ones(store):
    items = store.batch_items("b00002")
    store.submit("ann", "b00002", {i["rts_id"]: "rts" for i in items})
    p = store.progress()
    assert p["items_total"] == 12          # 3 batches × 4, injected excluded
    assert p["items_done"] == 4            # the 5th item in b00002 is injected
    assert p["area_km2_total"] == pytest.approx(12 * 1000.0 / 1e6)


def test_submit_rejects_an_incomplete_batch(store):
    """A partial submit marks the batch done, so its unrated polygons would
    never be re-served while progress counted them as reviewed."""
    ids = [i["rts_id"] for i in store.batch_items("b00000")]
    with pytest.raises(ValueError, match="incomplete"):
        store.submit("ann", "b00000", {ids[0]: "rts"})
    assert store.progress()["batches_done"] == 0
