"""Unit tests for inference.claim.ClaimStore (plan Phase 1 queue correctness).

GPU-free, network-free: a FakeBucket emulates GCS's create-if-absent
(`if_generation_match=0`) atomicity, listing, download, and delete. Covers the
invariants the dual-fleet run depends on: two workers never win one shard,
done-skip on restart, and stale-claim reclaim.
"""

from __future__ import annotations

import json

import pytest
from google.api_core.exceptions import (Forbidden, NotFound,
                                        PreconditionFailed)

from inference.claim import ClaimStore


# --------------------------------------------------------------------------
# In-memory fake GCS bucket
# --------------------------------------------------------------------------
class _FakeBlob:
    def __init__(self, store: dict, name: str) -> None:
        self._store = store
        self.name = name

    def upload_from_string(self, data, if_generation_match=None) -> None:
        if if_generation_match == 0 and self.name in self._store:
            raise PreconditionFailed(f"object exists: {self.name}")
        gen = self._store.get(self.name, (0, None))[0] + 1
        self._store[self.name] = (gen, data if isinstance(data, bytes) else data.encode())

    def download_as_text(self) -> str:
        if self.name not in self._store:
            raise NotFound(self.name)
        return self._store[self.name][1].decode()

    def delete(self, if_generation_match=None) -> None:
        if self.name not in self._store:
            raise NotFound(self.name)
        del self._store[self.name]


class _FakeBucket:
    def __init__(self) -> None:
        self._store: dict[str, tuple[int, bytes]] = {}

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self._store, name)

    def list_blobs(self, prefix: str = ""):
        return [_FakeBlob(self._store, n) for n in sorted(self._store) if n.startswith(prefix)]


class _CreateOnlyBlob(_FakeBlob):
    """A blob under GCS's ``roles/storage.objectCreator``.

    That role grants create-if-absent and nothing else: overwriting an existing
    object and deleting one both need ``storage.objects.delete``, so both 403.
    The plain `_FakeBucket` models atomicity but not IAM, which is why the live
    review campaign ran from 2026-08-04 to 2026-09-23 with dead heartbeats and a
    dead stale-claim TTL and no test could have caught it.
    """

    def upload_from_string(self, data, if_generation_match=None) -> None:
        if self.name in self._store:
            raise (PreconditionFailed(f"object exists: {self.name}")
                   if if_generation_match == 0
                   else Forbidden("storage.objects.delete denied (overwrite)"))
        super().upload_from_string(data, if_generation_match)

    def delete(self, if_generation_match=None) -> None:
        if self.name not in self._store:
            raise NotFound(self.name)
        raise Forbidden("storage.objects.delete denied")


class _CreateOnlyBucket(_FakeBucket):
    def blob(self, name: str) -> _CreateOnlyBlob:
        return _CreateOnlyBlob(self._store, name)


@pytest.fixture
def bucket() -> _FakeBucket:
    return _FakeBucket()


@pytest.fixture
def create_only_bucket() -> _CreateOnlyBucket:
    return _CreateOnlyBucket()


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_two_workers_never_win_one_shard(bucket):
    """Atomic create-if-absent: exactly one of two contenders claims a shard."""
    a = ClaimStore(bucket, "inf/run", worker_id="A")
    b = ClaimStore(bucket, "inf/run", worker_id="B")
    won_a = a.try_claim("s0001")
    won_b = b.try_claim("s0001")
    assert won_a is True
    assert won_b is False
    # The claim records the actual winner.
    owner = json.loads(bucket.blob("inf/run/claims/s0001").download_as_text())["worker_id"]
    assert owner == "A"


def test_claim_next_skips_done_and_returns_first_free(bucket):
    a = ClaimStore(bucket, "inf/run", worker_id="A")
    a.mark_done("s0001")
    a.mark_done("s0002")
    got = a.claim_next(["s0001", "s0002", "s0003", "s0004"])
    assert got == "s0003"  # first not-done, not-claimed


def test_done_skip_on_restart(bucket):
    """A restarted worker never reprocesses a shard with a done marker."""
    a = ClaimStore(bucket, "inf/run", worker_id="A")
    a.try_claim("s0003")
    a.mark_done("s0003")
    assert "s0003" in a.done_ids()
    # mark_done dropped the claim, and claim_next skips done shards.
    assert a.claim_next(["s0003"]) is None


def test_mark_done_clears_claim(bucket):
    a = ClaimStore(bucket, "inf/run", worker_id="A")
    a.try_claim("s0007")
    assert "s0007" in a.claimed_ids()
    a.mark_done("s0007")
    assert "s0007" not in a.claimed_ids()
    assert "s0007" in a.done_ids()


def test_fresh_claim_is_not_reclaimed(bucket):
    clock = [1000.0]
    a = ClaimStore(bucket, "inf/run", worker_id="A", now_fn=lambda: clock[0])
    a.try_claim("s0001")
    clock[0] += 100  # younger than the 1800s TTL
    b = ClaimStore(bucket, "inf/run", worker_id="B", now_fn=lambda: clock[0])
    assert b.reclaim_if_stale("s0001", stale_after_s=1800) is False
    assert b.claim_next(["s0001"]) is None  # still held by A


def test_stale_claim_is_reclaimed_and_reassigned(bucket):
    clock = [1000.0]
    a = ClaimStore(bucket, "inf/run", worker_id="A", now_fn=lambda: clock[0])
    a.try_claim("s0001")              # A claims at t=1000
    clock[0] += 2000                  # A "crashes"; 2000s > 1800s TTL
    b = ClaimStore(bucket, "inf/run", worker_id="B", now_fn=lambda: clock[0])
    got = b.claim_next(["s0001"], stale_after_s=1800)
    assert got == "s0001"
    owner = json.loads(bucket.blob("inf/run/claims/s0001").download_as_text())["worker_id"]
    assert owner == "B"


def test_heartbeat_keeps_claim_fresh(bucket):
    clock = [1000.0]
    a = ClaimStore(bucket, "inf/run", worker_id="A", now_fn=lambda: clock[0])
    a.try_claim("s0001")
    clock[0] += 1700
    a.heartbeat("s0001")              # refresh just before the 1800s TTL
    clock[0] += 200                   # now 1900s since claim, but 200s since heartbeat
    b = ClaimStore(bucket, "inf/run", worker_id="B", now_fn=lambda: clock[0])
    assert b.reclaim_if_stale("s0001", stale_after_s=1800) is False


def test_heartbeat_does_not_steal_others_claim(bucket):
    a = ClaimStore(bucket, "inf/run", worker_id="A")
    a.try_claim("s0001")
    b = ClaimStore(bucket, "inf/run", worker_id="B")
    b.heartbeat("s0001")  # B does not own it -> no-op
    owner = json.loads(bucket.blob("inf/run/claims/s0001").download_as_text())["worker_id"]
    assert owner == "A"


def test_reclaim_absent_claim_is_false(bucket):
    a = ClaimStore(bucket, "inf/run", worker_id="A")
    assert a.reclaim_if_stale("nope", stale_after_s=10) is False


# --------------------------------------------------------------------------
# Create-only identity: the failure mode that ran silently for seven weeks
# --------------------------------------------------------------------------
def test_claiming_still_works_without_delete_permission(create_only_bucket):
    """The half that never broke: create-if-absent needs only objectCreator."""
    a = ClaimStore(create_only_bucket, "inf/run", worker_id="A")
    assert a.try_claim("s0001") is True
    b = ClaimStore(create_only_bucket, "inf/run", worker_id="B")
    assert b.try_claim("s0001") is False


def test_heartbeat_surfaces_a_missing_delete_permission(create_only_bucket):
    """Refreshing a claim is an overwrite, which objectCreator cannot do.

    It must raise. The review app's browser beat has no ``.catch``, so a
    swallowed failure here is invisible: every claim in the live bucket still
    read ``claimed_at == heartbeat_at`` to the microsecond after seven weeks.
    """
    a = ClaimStore(create_only_bucket, "inf/run", worker_id="A")
    a.try_claim("s0001")
    with pytest.raises(Forbidden):
        a.heartbeat("s0001")


def test_stale_reclaim_surfaces_a_missing_delete_permission(create_only_bucket):
    """A 403 on reclaim must not be swallowed as 'someone else got there'."""
    clock = [1000.0]
    a = ClaimStore(create_only_bucket, "inf/run", worker_id="A",
                   now_fn=lambda: clock[0])
    a.try_claim("s0001")
    clock[0] += 2000  # well past the TTL
    b = ClaimStore(create_only_bucket, "inf/run", worker_id="B",
                   now_fn=lambda: clock[0])
    with pytest.raises(Forbidden):
        b.reclaim_if_stale("s0001", stale_after_s=1800)


class _RacingBlob(_FakeBlob):
    """Another worker always wins the delete: the object is gone, so 404."""

    def delete(self, if_generation_match=None) -> None:
        self._store.pop(self.name, None)
        raise NotFound(self.name)


class _RacingBucket(_FakeBucket):
    def blob(self, name: str) -> _RacingBlob:
        return _RacingBlob(self._store, name)


def test_lost_reclaim_race_is_still_swallowed():
    """Narrowing the except must not break the case it was written for:
    another worker deleting the claim first is normal, not an error."""
    bucket = _RacingBucket()
    clock = [1000.0]
    a = ClaimStore(bucket, "inf/run", worker_id="A", now_fn=lambda: clock[0])
    a.try_claim("s0001")
    clock[0] += 2000
    b = ClaimStore(bucket, "inf/run", worker_id="B", now_fn=lambda: clock[0])
    assert b.reclaim_if_stale("s0001", stale_after_s=1800) is False
