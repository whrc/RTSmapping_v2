"""Unit tests for data/sampler.py — curriculum schedule + per-batch pos:neg ratio."""

from __future__ import annotations

import pandas as pd
import pytest

from data.sampler import BalancedBatchSampler, parse_curriculum_schedule, ratio_for_epoch


def test_parse_schedule_sorted():
    s = parse_curriculum_schedule({"11-30": 5, "1-10": 1, "31-50": 10})
    assert s == [(1, 10, 1), (11, 30, 5), (31, 50, 10)]


def test_ratio_for_epoch():
    s = parse_curriculum_schedule({"1-10": 1, "11-30": 5, "31-50": 10})
    assert ratio_for_epoch(s, 1) == 1
    assert ratio_for_epoch(s, 10) == 1
    assert ratio_for_epoch(s, 15) == 5
    assert ratio_for_epoch(s, 40) == 10
    assert ratio_for_epoch(s, 1000) == 10  # clamped


def _make_metadata(n_pos: int, n_neg: int) -> tuple[list[str], pd.DataFrame]:
    rows = []
    ids = []
    for i in range(n_pos):
        tid = f"p{i:04d}"
        ids.append(tid)
        rows.append({"Tile_id": tid, "TrainClass": "Positive", "RegionName": "r", "UIDs": "x",
                     "centroid_lat": 0, "centroid_lon": 0})
    for i in range(n_neg):
        tid = f"n{i:04d}"
        ids.append(tid)
        rows.append({"Tile_id": tid, "TrainClass": "Negative", "RegionName": "r", "UIDs": "",
                     "centroid_lat": 0, "centroid_lon": 0})
    return ids, pd.DataFrame(rows)


def test_sampler_ratio_1_1():
    ids, df = _make_metadata(n_pos=32, n_neg=32)
    sampler = BalancedBatchSampler(
        tile_ids=ids, metadata=df, batch_size=8,
        schedule={"1-10": 1}, seed=42, epoch=1,
    )
    pos_ids = {df.set_index("Tile_id").index.get_loc(t) for t in df.loc[df.TrainClass == "Positive", "Tile_id"]}
    for batch in sampler:
        pos_count = sum(1 for i in batch if i in pos_ids)
        # batch_size=8, ratio 1:1 → 4 pos, 4 neg
        assert pos_count == 4


def test_sampler_ratio_1_5_shifts_distribution():
    ids, df = _make_metadata(n_pos=32, n_neg=200)
    sampler = BalancedBatchSampler(
        tile_ids=ids, metadata=df, batch_size=12,
        schedule={"1-10": 5}, seed=42, epoch=1,
    )
    pos_ids = {i for i, tid in enumerate(ids) if df.iloc[i]["TrainClass"] == "Positive"}
    batches = list(sampler)
    for batch in batches:
        pos_count = sum(1 for i in batch if i in pos_ids)
        # batch_size=12, ratio 1:5 → n_pos = 12 // 6 = 2, n_neg = 10
        assert pos_count == 2
        assert len(batch) == 12


def test_sampler_determinism_across_epochs():
    ids, df = _make_metadata(n_pos=32, n_neg=200)
    s1 = BalancedBatchSampler(ids, df, batch_size=8, schedule={"1-10": 1}, seed=42, epoch=1)
    s2 = BalancedBatchSampler(ids, df, batch_size=8, schedule={"1-10": 1}, seed=42, epoch=1)
    assert list(s1) == list(s2)

    s3 = BalancedBatchSampler(ids, df, batch_size=8, schedule={"1-10": 1}, seed=42, epoch=2)
    # Different epoch must differ (seed+epoch is the RNG key).
    assert list(s3) != list(s1)


def test_sampler_requires_both_classes():
    ids, df = _make_metadata(n_pos=16, n_neg=0)
    with pytest.raises(ValueError, match="both classes"):
        BalancedBatchSampler(ids, df, batch_size=4, schedule={"1-10": 1}, seed=42)


# --------------------------------------------------------------------------
# train_positive_subset_pct (consumed by scripts/train.py:_filter_train_positive_subset)
# --------------------------------------------------------------------------


def test_filter_train_positive_subset_keeps_negatives_intact():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from train import _filter_train_positive_subset

    ids, df = _make_metadata(n_pos=20, n_neg=80)
    out = _filter_train_positive_subset(ids, df, subset_pct=25)
    class_by_id = df.set_index("Tile_id")["TrainClass"].to_dict()
    pos_kept = [t for t in out if class_by_id[t] == "Positive"]
    neg_kept = [t for t in out if class_by_id[t] == "Negative"]
    # 25% of 20 positives = 5; all 80 negatives untouched.
    assert len(pos_kept) == 5
    assert len(neg_kept) == 80


def test_filter_train_positive_subset_is_deterministic():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from train import _filter_train_positive_subset

    ids, df = _make_metadata(n_pos=20, n_neg=80)
    a = _filter_train_positive_subset(ids, df, subset_pct=25)
    b = _filter_train_positive_subset(ids, df, subset_pct=25)
    assert a == b


def test_filter_train_positive_subset_full_pct_no_op():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from train import _filter_train_positive_subset

    ids, df = _make_metadata(n_pos=20, n_neg=80)
    out = _filter_train_positive_subset(ids, df, subset_pct=100)
    assert len(out) == 100
