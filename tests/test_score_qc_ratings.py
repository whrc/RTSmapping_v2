"""QC-rating scorer (scripts/score_qc_ratings.py): verdicts → precision per
(conf tier × size band) with Wilson CIs → the A(p) acceptance grid at a
precision floor. GPU-free; synthetic ratings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.score_qc_ratings import (SIZE_BANDS, export_false_polygons,
                                      precision_grid)


def _ratings() -> pd.DataFrame:
    rows = []
    # high tier: 9/10 real in the small band, 10/10 in a large band
    rows += [dict(conf_class="high", area_m2=300, qc_verdict="rts")] * 9
    rows += [dict(conf_class="high", area_m2=300, qc_verdict="false")]
    rows += [dict(conf_class="high", area_m2=30000, qc_verdict="rts")] * 10
    # low tier small: 2/10 real (fails any sane floor); 1 unsure excluded
    rows += [dict(conf_class="low", area_m2=300, qc_verdict="rts")] * 2
    rows += [dict(conf_class="low", area_m2=300, qc_verdict="false")] * 8
    rows += [dict(conf_class="low", area_m2=300, qc_verdict="unsure")]
    return pd.DataFrame(rows)


def test_precision_grid_counts_and_wilson():
    g = precision_grid(_ratings(), floor=0.5)
    hi_small = g[(g.conf_class == "high") & (g.band == "<500")].iloc[0]
    assert hi_small.n_rated == 10 and hi_small.n_rts == 9
    assert abs(hi_small.precision - 0.9) < 1e-9
    assert 0.55 < hi_small.wilson_lo < 0.9 < hi_small.wilson_hi <= 1.0
    assert bool(hi_small.accept)
    lo_small = g[(g.conf_class == "low") & (g.band == "<500")].iloc[0]
    assert lo_small.n_rated == 10          # unsure excluded
    assert lo_small.n_unsure == 1
    assert not bool(lo_small.accept)       # 0.2 < floor


def test_empty_cells_are_reported_not_dropped():
    g = precision_grid(_ratings(), floor=0.5)
    # full grid: every tier × band combination present, unrated cells n=0
    assert len(g) == 3 * len(SIZE_BANDS)
    empty = g[(g.conf_class == "medium")]
    assert (empty.n_rated == 0).all()
    assert empty.precision.isna().all()
    assert (~empty.accept.astype(bool)).all()  # never accept unmeasured cells


def test_export_false_polygons_hard_negative_seed(tmp_path):
    """Rated-false polygons export with geometry + verdict + tier/size — the
    v3 hard-negative seed set."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    sample = gpd.GeoDataFrame(
        {"rts_id": [1, 2, 3], "conf_class": ["high", "low", "low"],
         "area_m2": [300.0, 400.0, 500.0]},
        geometry=[Polygon([(i, 0), (i + 1, 0), (i + 1, 1)]) for i in range(3)],
        crs="EPSG:3857")
    sample.to_file(tmp_path / "s.gpkg", driver="GPKG")
    ratings = pd.DataFrame({"rts_id": [1, 2, 3],
                            "qc_verdict": ["rts", "false", "unsure"]})
    ratings.to_csv(tmp_path / "r.csv", index=False)

    export_false_polygons(str(tmp_path / "r.csv"), str(tmp_path / "s.gpkg"),
                          str(tmp_path / "f.gpkg"))
    out = gpd.read_file(tmp_path / "f.gpkg")
    assert list(out["rts_id"]) == [2]          # only rated false
    assert out.iloc[0]["qc_verdict"] == "false"
    assert {"conf_class", "area_m2"} <= set(out.columns)
    assert out.crs.to_epsg() == 3857
