"""Unit tests for scripts/build_review_manifest.py — queue construction.

The campaign's headline claim ("every polygon with max_prob >= p is reviewed")
is only true if batches are cut in probability order and every polygon lands in
exactly one of them. These tests pin that, plus the within-batch shuffle that
defends against rater response bias, and the determinism the manifest needs to
be rebuildable.

Spec: `post-inference/review_campaign.md` §3, §5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_review_manifest import REPLICATE_OFFSET, build_manifest


@pytest.fixture
def attributes(tmp_path):
    """1,000 polygons with distinct, descending-sortable probabilities."""
    rng = np.random.default_rng(0)
    n = 1000
    df = pd.DataFrame({
        "rts_id": np.arange(1, n + 1),
        "max_prob": rng.uniform(0.3, 1.0, n),
        "conf_class": np.where(rng.uniform(size=n) > 0.5, "high", "low"),
        "area_m2": rng.uniform(50, 20000, n),
        "centroid_lat": rng.uniform(60, 76, n),
        "centroid_lon": rng.uniform(-160, 160, n),
    })
    path = tmp_path / "attrs.parquet"
    df.to_parquet(path, index=False)
    return str(path)


# --- coverage -------------------------------------------------------------
def test_every_polygon_appears_exactly_once_as_coverage(attributes):
    man = build_manifest(attributes, batch_size=100, n_replicates=20)
    cov = man[~man["injected"]]
    assert len(cov) == 1000
    assert cov["rts_id"].is_unique
    assert set(cov["rts_id"]) == set(range(1, 1001))


def test_batches_are_full_except_the_last(attributes):
    man = build_manifest(attributes, batch_size=300, n_replicates=0)
    sizes = man.groupby("batch_id").size().to_list()
    assert sizes == [300, 300, 300, 100]


def test_batch_and_item_ids_never_collide_within_a_batch(attributes):
    """(batch_id, rts_id) must be unique or a verdict is ambiguous."""
    man = build_manifest(attributes, batch_size=100, n_replicates=50)
    assert not man.duplicated(["batch_id", "rts_id"]).any()


# --- ordering -------------------------------------------------------------
def test_batches_are_cut_in_descending_probability_order(attributes):
    man = build_manifest(attributes, batch_size=100, n_replicates=0)
    g = man.groupby("batch_id")["max_prob"]
    lo, hi = g.min().to_numpy(), g.max().to_numpy()
    # Every batch's floor is at or above the next batch's ceiling.
    assert (lo[:-1] >= hi[1:]).all()


def test_item_order_within_a_batch_is_shuffled(attributes):
    """Defends against the response bias a sorted run would induce."""
    man = build_manifest(attributes, batch_size=100, n_replicates=0)
    batch = man[man["batch_id"] == "b00000"]
    assert not batch["max_prob"].is_monotonic_decreasing
    assert list(batch["seq"]) == list(range(len(batch)))


# --- replicates -----------------------------------------------------------
def test_replicates_are_injected_into_later_batches(attributes):
    man = build_manifest(attributes, batch_size=100, n_replicates=20)
    inj = man[man["injected"]]
    assert len(inj) == 20
    for rts_id, batch_id in zip(inj["rts_id"], inj["batch_id"]):
        source = man[(man["rts_id"] == rts_id) & (~man["injected"])]
        assert batch_id > source["batch_id"].iloc[0]
        assert int(batch_id[1:]) - int(source["batch_id"].iloc[0][1:]) == \
            REPLICATE_OFFSET


def test_replicates_are_spread_through_the_campaign(attributes):
    """Clustered replicates would measure agreement on one slice only."""
    man = build_manifest(attributes, batch_size=100, n_replicates=10)
    batches = sorted(int(b[1:]) for b in man.loc[man["injected"], "batch_id"])
    assert len(set(batches)) > 5
    assert max(batches) - min(batches) >= 5


def test_zero_replicates_is_allowed(attributes):
    man = build_manifest(attributes, batch_size=100, n_replicates=0)
    assert not man["injected"].any()
    assert len(man) == 1000


# --- determinism ----------------------------------------------------------
def test_rebuild_is_identical(attributes):
    a = build_manifest(attributes, batch_size=100, n_replicates=20)
    b = build_manifest(attributes, batch_size=100, n_replicates=20)
    pd.testing.assert_frame_equal(a, b)


def test_duplicate_input_ids_are_rejected(attributes, tmp_path):
    df = pd.read_parquet(attributes)
    dup = pd.concat([df, df.head(1)], ignore_index=True)
    path = tmp_path / "dup.parquet"
    dup.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate rts_id"):
        build_manifest(str(path), batch_size=100, n_replicates=0)
