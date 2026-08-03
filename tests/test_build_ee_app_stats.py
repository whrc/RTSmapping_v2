"""GEE app stats generator (scripts/build_ee_app_stats.py): the precomputed MMU
retention ladders the app reads instead of aggregating server-side. GPU-free.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from scripts.build_ee_app_stats import (
    LADDER_M2, build_stats, min_blob_m2, retention, write_js,
)

RES_M = 4.777314267158508


@pytest.fixture
def inventories():
    """A 9-polygon candidate inventory across all three tiers + 3 t65 cores."""
    candidates = pd.DataFrame({
        "area_m2": [5.0, 50.0, 90.0, 800.0, 3000.0, 12000.0, 25.0, 400.0, 9000.0],
        "conf_class": ["high", "high", "high", "medium", "medium", "medium",
                       "low", "low", "low"],
        "centroid_lat": [60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0, 74.0, 76.0],
    })
    t65 = pd.DataFrame({"area_m2": [40.0, 600.0, 7000.0]})
    return candidates, t65


def test_retention_starts_whole_and_decreases(inventories):
    candidates, _ = inventories
    areas = candidates["area_m2"].values
    r = retention(areas, LADDER_M2)

    assert r["n"][0] == len(areas)                        # MMU 0 keeps everything
    assert r["km2"][0] == pytest.approx(areas.sum() / 1e6)
    assert all(a >= b for a, b in zip(r["n"], r["n"][1:]))
    assert all(a >= b for a, b in zip(r["km2"], r["km2"][1:]))


def test_retention_filter_is_inclusive_at_the_threshold():
    # `area_m2 >= mmu` — the app's ee.Filter.gte and vectorize_region's geodesic
    # filter both keep a polygon sitting exactly on the floor.
    r = retention(np.array([79.0, 78.0]), [79])
    assert r["n"] == [1]


def test_tier_series_sum_to_the_whole_inventory(inventories):
    candidates, t65 = inventories
    stats = build_stats(candidates, t65, min_blob_px=2000, resolution_m=RES_M)

    for i in range(len(stats["ladder_m2"])):
        pooled = sum(stats["series"][t]["n"][i] for t in ("high", "medium", "low"))
        assert pooled == retention(candidates["area_m2"].values,
                                   stats["ladder_m2"])["n"][i]
    assert stats["series"]["t65"]["n"][0] == len(t65)
    assert stats["totals"]["candidates_n"] == len(candidates)


def test_non_exhaustive_conf_class_is_rejected(inventories):
    candidates, t65 = inventories
    candidates.loc[0, "conf_class"] = "unclassified"
    with pytest.raises(ValueError, match="not exhaustive"):
        build_stats(candidates, t65, min_blob_px=2000, resolution_m=RES_M)


def test_min_blob_ground_area_shrinks_with_latitude():
    # A pixel MMU is not a constant area: 3857 pixel ground area is res²·cos²(lat).
    at60, at76 = (min_blob_m2(2000, RES_M, la) for la in (60, 76))
    assert at60 == pytest.approx(2000 * RES_M ** 2 * 0.25, rel=1e-6)
    assert at60 > at76 * 4                       # ~7x across the domain
    assert min_blob_m2(2000, RES_M, 0.0) == pytest.approx(2000 * RES_M ** 2)


def test_ladder_carries_arts_p1_and_the_representative_min_blob(inventories):
    candidates, t65 = inventories
    stats = build_stats(candidates, t65, min_blob_px=2000, resolution_m=RES_M)

    assert 79 in stats["ladder_m2"]                       # ARTS P1
    assert stats["min_blob"]["representative_m2"] in stats["ladder_m2"]
    assert stats["ladder_m2"] == sorted(set(stats["ladder_m2"]))
    assert stats["min_blob"]["median_lat"] == pytest.approx(68.0)


def test_write_js_emits_a_parseable_literal(tmp_path, inventories):
    candidates, t65 = inventories
    stats = build_stats(candidates, t65, min_blob_px=2000, resolution_m=RES_M)
    out = tmp_path / "app_stats.js"
    write_js(stats, out, ["a.parquet", "b.gpkg"])

    js = out.read_text()
    m = re.search(r"var APP_STATS = (\{.*\});", js, re.S)
    assert m, "no APP_STATS block"
    assert json.loads(m.group(1)) == stats
    assert "a.parquet" in js and "do not hand-edit" in js
