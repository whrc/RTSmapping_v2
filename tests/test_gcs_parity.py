"""Unit tests for scripts/gcs_parity.py — the check that gates deletion.

GPU-free, network-free: a FakeClient serves canned listings. The tool compares
every object rather than sampling, so these tests pin the ways a copy can be
wrong — missing, extra, truncated, corrupted — and confirm each one fails loudly.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gcs_parity", Path(__file__).resolve().parent.parent / "scripts" / "gcs_parity.py"
)
gcs_parity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gcs_parity)

Entry = gcs_parity.Entry


class _FakeBlob:
    def __init__(self, name: str, size: int, md5: str, crc32c: str = "crc") -> None:
        self.name = name
        self.size = size
        self.md5_hash = md5
        self.crc32c = crc32c


class _FakeClient:
    """Serves listings keyed by ``bucket/prefix-with-trailing-slash``."""

    def __init__(self, listings: dict[str, list[_FakeBlob]]) -> None:
        self._listings = listings

    def list_blobs(self, bucket, prefix=None, fields=None):
        return iter(self._listings.get(f"{bucket}/{prefix or ''}", []))


def _blobs(prefix: str, n: int, size: int = 100) -> list[_FakeBlob]:
    """n objects in lexicographic order, as GCS would list them."""
    return [_FakeBlob(f"{prefix}obj{i:04d}", size, f"md5-{i}", f"crc-{i}") for i in range(n)]


def _entries(n: int, size: int = 100) -> list[Entry]:
    return [Entry(f"obj{i:04d}", size, f"md5-{i}", f"crc-{i}") for i in range(n)]


# ---------------------------------------------------------------- split_uri
@pytest.mark.parametrize(
    "uri,expected",
    [
        ("gs://b/p/q", ("b", "p/q")),
        ("gs://b/p/q/", ("b", "p/q")),
        ("gs://b", ("b", "")),
        ("gs://b/", ("b", "")),
    ],
)
def test_split_uri(uri, expected):
    assert gcs_parity.split_uri(uri) == expected


def test_split_uri_rejects_non_gs():
    with pytest.raises(ValueError):
        gcs_parity.split_uri("/mnt/outputs/thing")


# ----------------------------------------------------------------- entries
def test_entries_are_named_relative_to_the_prefix():
    """The two sides sit at different absolute paths; comparison needs relative names."""
    client = _FakeClient({
        "b/old/": _blobs("old/", 3),
        "c/brand/new/": _blobs("brand/new/", 3),
    })
    src = list(gcs_parity.entries(client, "gs://b/old"))
    dst = list(gcs_parity.entries(client, "gs://c/brand/new"))
    assert [e.name for e in src] == ["obj0000", "obj0001", "obj0002"]
    assert src == dst


def test_entries_skips_directory_placeholders():
    client = _FakeClient({"b/old/": _blobs("old/", 3) + [_FakeBlob("old/sub/", 0, "")]})
    assert len(list(gcs_parity.entries(client, "gs://b/old"))) == 3


def test_entries_tolerates_null_size_and_checksums():
    client = _FakeClient({"b/old/": [_FakeBlob("old/a", None, None, None)]})
    assert list(gcs_parity.entries(client, "gs://b/old")) == [Entry("a", 0, "", "")]


# ----------------------------------------------------------------- compare
def test_identical_listings_match():
    result = gcs_parity.compare(iter(_entries(20)), iter(_entries(20)))
    assert result.ok
    assert result.src_count == result.dst_count == 20
    assert result.src_bytes == result.dst_bytes == 2000


def test_both_empty_is_a_pass():
    """A prefix that was legitimately empty is not a failure."""
    assert gcs_parity.compare(iter([]), iter([])).ok


def test_missing_object_is_reported_by_name():
    src = _entries(20)
    dst = [e for e in src if e.name != "obj0007"]
    result = gcs_parity.compare(iter(src), iter(dst))
    assert not result.ok
    assert result.missing == ["obj0007"]
    assert result.src_count == 20 and result.dst_count == 19


def test_missing_object_at_the_end_is_caught():
    """The lockstep walk must drain the longer side, not stop at the shorter one."""
    result = gcs_parity.compare(iter(_entries(20)), iter(_entries(19)))
    assert result.missing == ["obj0019"]


def test_missing_object_at_the_start_is_caught():
    result = gcs_parity.compare(iter(_entries(20)), iter(_entries(20)[1:]))
    assert result.missing == ["obj0000"]


def test_extra_object_at_destination_is_reported():
    dst = _entries(20) + [Entry("obj9999", 100, "md5-x", "crc-x")]
    result = gcs_parity.compare(iter(_entries(20)), iter(dst))
    assert not result.ok
    assert result.extra == ["obj9999"]


def test_truncated_object_is_caught_by_size():
    """Same count, wrong bytes — the classic partial write."""
    dst = _entries(20)
    dst[3] = Entry("obj0003", 1, "md5-3", "crc-3")
    result = gcs_parity.compare(iter(_entries(20)), iter(dst))
    assert not result.ok
    assert result.differing == ["obj0003"]


def test_corrupt_object_is_caught_by_md5():
    """Same count AND same bytes — only the MD5 gives it away. Sampling would miss it."""
    dst = _entries(20)
    dst[3] = Entry("obj0003", 100, "CORRUPT", "crc-3")
    result = gcs_parity.compare(iter(_entries(20)), iter(dst))
    assert not result.ok
    assert result.differing == ["obj0003"]


def test_every_corruption_is_caught_not_just_a_sample():
    """The point of lockstep: 500 objects, all corrupt, none missed."""
    src = _entries(500)
    dst = [Entry(e.name, e.size, "CORRUPT", e.crc32c) for e in src]
    result = gcs_parity.compare(iter(src), iter(dst))
    assert not result.ok
    assert len(result.differing) == gcs_parity.MAX_REPORTED  # reporting is capped…
    assert result.src_count == result.dst_count == 500       # …but counting is not


def test_reported_names_are_capped_but_the_walk_completes():
    src = _entries(100)
    result = gcs_parity.compare(iter(src), iter([]))
    assert len(result.missing) == gcs_parity.MAX_REPORTED
    assert result.src_count == 100 and result.dst_count == 0


# -------------------------------------------------------------------- main
def _run_main(monkeypatch, listings):
    monkeypatch.setattr(gcs_parity.storage, "Client", lambda: _FakeClient(listings))
    monkeypatch.setattr(
        gcs_parity.sys, "argv",
        ["gcs_parity.py", "--src", "gs://b/old", "--dst", "gs://c/new"],
    )
    return gcs_parity.main()


def test_main_exits_zero_on_a_clean_copy(monkeypatch):
    assert _run_main(monkeypatch, {
        "b/old/": _blobs("old/", 20),
        "c/new/": _blobs("new/", 20),
    }) == 0


def test_main_exits_nonzero_on_a_bad_copy(monkeypatch):
    bad = _blobs("new/", 20)
    bad[3] = _FakeBlob("new/obj0003", 100, "CORRUPT")
    assert _run_main(monkeypatch, {"b/old/": _blobs("old/", 20), "c/new/": bad}) == 1


def test_main_exits_nonzero_on_an_empty_destination(monkeypatch):
    assert _run_main(monkeypatch, {"b/old/": _blobs("old/", 20), "c/new/": []}) == 1


# ---------------------------------------------------------------- composite objects


def test_composite_objects_are_compared_on_crc32c_not_waved_through():
    """GCS stores no md5Hash for composite objects; size alone must not pass them.

    This is the 2026-08-28 finding: three hand-uploaded files in the t65 build (the
    largest 3.6 GB) carried no MD5, so an MD5-only comparison saw "" == "" and would
    have called a corrupt copy identical.
    """
    src = [Entry("big", 3_605_613_560, "", "zMHaaQ==")]
    dst = [Entry("big", 3_605_613_560, "", "DIFFERENT")]
    result = gcs_parity.compare(iter(src), iter(dst))
    assert result.differing == ["big"]
    assert not result.ok


def test_composite_objects_matching_on_crc32c_pass():
    src = [Entry("big", 3_605_613_560, "", "zMHaaQ==")]
    dst = [Entry("big", 3_605_613_560, "", "zMHaaQ==")]
    assert gcs_parity.compare(iter(src), iter(dst)).ok


def test_no_shared_checksum_is_reported_not_assumed_equal():
    """Absence of evidence is not agreement."""
    src = [Entry("x", 10, "", "")]
    dst = [Entry("x", 10, "", "")]
    assert gcs_parity.compare(iter(src), iter(dst)).differing == ["x"]


def test_md5_still_wins_when_both_sides_have_it():
    src = [Entry("x", 10, "same", "crc-a")]
    dst = [Entry("x", 10, "same", "crc-b")]
    assert gcs_parity.compare(iter(src), iter(dst)).ok
