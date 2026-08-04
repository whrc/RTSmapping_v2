"""Unit tests for scripts/merge_review_verdicts.py — pooling and agreement.

The merge is what turns a pile of batch JSONLs into the verified inventory, so
these pin the rules that decide what the product says: coverage verdicts win
over replicates, a stray id is an error rather than a silent drop, kappa is the
real statistic, and a partial campaign yields a partial-but-honest product.

Spec: `post-inference/review_campaign.md` §7–§8.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.merge_review_verdicts import (agreement_report, cohens_kappa,
                                           merge_verdicts, read_verdicts,
                                           replicate_pairs)


def _manifest() -> pd.DataFrame:
    rows = [{"rts_id": i, "batch_id": "b00000", "seq": i - 1,
             "injected": False, "max_prob": 0.9, "conf_class": "high",
             "area_m2": 100.0, "centroid_lat": 70.0, "centroid_lon": -120.0}
            for i in range(1, 5)]
    rows.append({**rows[0], "batch_id": "b00003", "injected": True})
    return pd.DataFrame(rows)


def _verdict(rts_id, verdict, reviewer, batch="b00000", injected=False, ts=1.0):
    return {"rts_id": rts_id, "verdict": verdict, "reviewer": reviewer,
            "batch_id": batch, "injected": injected, "reviewed_at": ts}


# --- kappa ----------------------------------------------------------------
def test_kappa_matches_a_hand_computed_table():
    """Classic 2x2: po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5 → kappa = 0.4."""
    a = pd.Series(["rts"] * 5 + ["false"] * 5)
    b = pd.Series(["rts"] * 4 + ["false"] + ["rts"] * 2 + ["false"] * 3)
    assert cohens_kappa(a, b) == pytest.approx(0.4)


def test_kappa_is_one_for_perfect_agreement():
    a = pd.Series(["rts", "false", "unsure", "rts"])
    assert cohens_kappa(a, a.copy()) == pytest.approx(1.0)


def test_kappa_is_nan_when_one_label_is_universal():
    """No variance to agree about — undefined, and must not read as perfect."""
    a = pd.Series(["rts"] * 4)
    assert pd.isna(cohens_kappa(a, a.copy()))


def test_kappa_of_an_empty_sample_is_nan():
    assert pd.isna(cohens_kappa(pd.Series(dtype=str), pd.Series(dtype=str)))


# --- pairing --------------------------------------------------------------
def test_replicate_pairs_join_a_replicate_to_its_coverage_verdict():
    v = pd.DataFrame([_verdict(1, "rts", "ann"),
                      _verdict(1, "false", "bob", "b00003", injected=True)])
    pairs = replicate_pairs(v)
    assert len(pairs) == 1
    assert pairs.iloc[0]["verdict_replicate"] == "false"
    assert pairs.iloc[0]["verdict_coverage"] == "rts"
    assert pairs.iloc[0]["reviewer_replicate"] == "bob"
    assert pairs.iloc[0]["reviewer_coverage"] == "ann"


def test_a_replicate_without_its_coverage_verdict_is_dropped():
    """Its source batch is not rated yet — nothing to pair against."""
    v = pd.DataFrame([_verdict(1, "rts", "bob", "b00003", injected=True)])
    assert len(replicate_pairs(v)) == 0


# --- merging --------------------------------------------------------------
def test_coverage_verdict_wins_over_the_replicate():
    v = pd.DataFrame([_verdict(1, "rts", "ann"),
                      _verdict(1, "false", "bob", "b00003", injected=True)])
    merged = merge_verdicts(v, _manifest())
    row = merged[merged["rts_id"] == 1].iloc[0]
    assert row["qc_verdict"] == "rts"
    assert row["n_reviews"] == 2
    assert row["reviewers"] == "ann,bob"
    assert row["agreement"] is False or row["agreement"] == False  # noqa: E712


def test_agreement_is_true_when_both_reviewers_match():
    v = pd.DataFrame([_verdict(1, "rts", "ann"),
                      _verdict(1, "rts", "bob", "b00003", injected=True)])
    assert merge_verdicts(v, _manifest()).iloc[0]["agreement"] == True  # noqa: E712


def test_unreplicated_polygons_have_no_agreement_value():
    v = pd.DataFrame([_verdict(2, "false", "ann")])
    assert pd.isna(merge_verdicts(v, _manifest()).iloc[0]["agreement"])


def test_a_stray_id_is_an_error_not_a_silent_drop():
    v = pd.DataFrame([_verdict(999, "rts", "ann")])
    with pytest.raises(ValueError, match="not in the manifest"):
        merge_verdicts(v, _manifest())


def test_duplicate_coverage_verdicts_keep_the_latest():
    v = pd.DataFrame([_verdict(1, "rts", "ann", ts=1.0),
                      _verdict(1, "false", "ann", ts=2.0)])
    assert merge_verdicts(v, _manifest()).iloc[0]["qc_verdict"] == "false"


def test_a_partial_campaign_merges_only_what_was_rated():
    v = pd.DataFrame([_verdict(1, "rts", "ann"), _verdict(2, "false", "ann")])
    merged = merge_verdicts(v, _manifest())
    assert len(merged) == 2
    assert set(merged["rts_id"]) == {1, 2}


def test_an_empty_campaign_merges_to_an_empty_frame():
    merged = merge_verdicts(pd.DataFrame(), _manifest())
    assert merged.empty
    assert "qc_verdict" in merged.columns


# --- report ---------------------------------------------------------------
def test_report_counts_coverage_and_excludes_injected_from_the_total():
    v = pd.DataFrame([_verdict(1, "rts", "ann"), _verdict(2, "false", "ann")])
    rep = agreement_report(v, merge_verdicts(v, _manifest()), _manifest())
    assert rep["polygons_total"] == 4        # the injected row is not coverage
    assert rep["polygons_reviewed"] == 2
    assert rep["fraction_reviewed"] == 0.5
    assert rep["verdict_counts"] == {"rts": 1, "false": 1}
    assert rep["per_reviewer"] == {"ann": 2}


def test_kappa_is_only_computed_across_different_reviewers():
    """A reviewer agreeing with themselves is not inter-rater agreement."""
    same = pd.DataFrame([_verdict(1, "rts", "ann"),
                         _verdict(1, "rts", "ann", "b00003", injected=True)])
    rep = agreement_report(same, merge_verdicts(same, _manifest()), _manifest())
    assert rep["replicate_pairs"] == 1
    assert rep["replicate_pairs_cross_reviewer"] == 0
    assert "kappa_cross_reviewer" not in rep


def test_report_compares_against_the_2026_07_pass():
    v = pd.DataFrame([_verdict(1, "rts", "ann"), _verdict(2, "false", "ann")])
    prior = pd.DataFrame({"rts_id": [1, 2], "qc_verdict": ["rts", "rts"]})
    rep = agreement_report(v, merge_verdicts(v, _manifest()), _manifest(), prior)
    assert rep["vs_2026_07"]["n"] == 2
    assert rep["vs_2026_07"]["raw_agreement"] == 0.5


# --- reading --------------------------------------------------------------
def test_read_verdicts_reads_every_batch_file(tmp_path):
    (tmp_path / "b00000.jsonl").write_text(
        "\n".join(json.dumps(_verdict(i, "rts", "ann")) for i in (1, 2)))
    (tmp_path / "b00001.jsonl").write_text(json.dumps(_verdict(3, "false", "bob")))
    df = read_verdicts(str(tmp_path))
    assert len(df) == 3
    assert set(df["rts_id"]) == {1, 2, 3}


def test_read_verdicts_of_an_empty_campaign_is_an_empty_frame(tmp_path):
    df = read_verdicts(str(tmp_path))
    assert df.empty
    assert "verdict" in df.columns


def test_duplicate_coverage_verdicts_do_not_inflate_the_pair_count():
    """Joining on a non-unique id would multiply the agreement sample."""
    v = pd.DataFrame([_verdict(1, "rts", "ann", ts=1.0),
                      _verdict(1, "false", "ann", ts=2.0),
                      _verdict(1, "rts", "bob", "b00003", injected=True)])
    pairs = replicate_pairs(v)
    assert len(pairs) == 1
    assert pairs.iloc[0]["verdict_coverage"] == "false"  # the latest one
