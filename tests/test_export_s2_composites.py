"""Unit tests for scripts/export_s2_composites.py — grid + domain-cell logic.

Earth Engine and GCS are not exercised; these cover the pure footprint geometry:
the lat/lon grid, deterministic cell ids, and clipping cells to a domain polygon.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import box

from scripts.export_s2_composites import cell_id, domain_cells, latlon_grid


def test_latlon_grid_aligns_and_covers():
    cells = latlon_grid((-10.0, 60.0, 5.0, 64.0), dlat=1.0, dlon=3.0)
    # lon spans multiples of 3 from -12..6 (6 cols), lat 60..64 (4 rows)
    assert len(cells) == 6 * 4
    # every cell is dlon x dlat and origin-aligned
    for lon0, lat0, lon1, lat1 in cells:
        assert abs((lon1 - lon0) - 3.0) < 1e-9
        assert abs((lat1 - lat0) - 1.0) < 1e-9
        assert abs(lon0 % 3.0) < 1e-9 and abs(lat0 % 1.0) < 1e-9
    # the bbox corners are covered
    assert any(c[0] <= -10.0 < c[2] and c[1] <= 60.0 < c[3] for c in cells)
    assert any(c[0] <= 5.0 - 1e-9 < c[2] and c[1] <= 64.0 - 1e-9 < c[3] for c in cells)


def test_cell_id_deterministic_and_sign_safe():
    assert cell_id(3.0, 70.0) == cell_id(3.0, 70.0)
    assert cell_id(-150.0, 74.0) == "W1500_N0740"
    assert cell_id(0.0, -2.5) == "E0000_S0025"
    # distinct corners -> distinct ids
    assert cell_id(3.0, 70.0) != cell_id(6.0, 70.0)


def test_domain_cells_keeps_only_intersecting():
    # a small domain box in WGS84; reproject path exercised via crs set
    domain = gpd.GeoDataFrame(geometry=[box(-10.0, 60.0, 5.0, 64.0)], crs="EPSG:4326")
    cells = domain_cells(domain, dlat=1.0, dlon=3.0)
    assert len(cells) > 0
    ids = {c[0] for c in cells}
    assert len(ids) == len(cells)                       # unique ids
    for cid, bbox, clip in cells:
        assert not clip.is_empty
        # clip is contained in the domain box and in its own cell
        assert clip.bounds[0] >= -10.0 - 1e-6 and clip.bounds[2] <= 5.0 + 1e-6
        lon0, lat0, lon1, lat1 = bbox
        assert clip.bounds[0] >= lon0 - 1e-6 and clip.bounds[2] <= lon1 + 1e-6
    # a cell far from the domain is excluded
    assert "E0900_N0700" not in ids                     # lon 90E, lat 70N


def test_queue_full_error_is_recognised():
    """The per-user ceiling arrives as a message, not a typed exception."""
    from scripts.export_s2_composites import is_queue_full_error
    assert is_queue_full_error(
        Exception("Too many tasks already in the queue (3002, limit 3000)."))
    assert not is_queue_full_error(Exception("Image.divide: ... Got 0 and 1."))


def test_submit_with_backoff_waits_out_a_full_queue(monkeypatch):
    """A full queue is a WAIT, not a failure.

    Launching three more years while 1,799 tasks were queued died at 3,002 server-side
    (2026-08-25) because the per-project slot check cannot see the per-user ceiling.
    A multi-day export must not abort partway just because the queue was briefly full.
    """
    import scripts.export_s2_composites as X
    monkeypatch.setattr(X.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def submit():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("Too many tasks already in the queue (3002, limit 3000).")
        return "task"

    assert X.submit_with_backoff(None, submit, poll=1) == "task"
    assert calls["n"] == 3


def test_submit_with_backoff_reraises_other_errors(monkeypatch):
    """A real export bug must surface immediately, not be retried for a day."""
    import scripts.export_s2_composites as X
    monkeypatch.setattr(X.time, "sleep", lambda s: None)

    def submit():
        raise ValueError("Image.divide: If one image has no bands ...")

    with pytest.raises(ValueError):
        X.submit_with_backoff(None, submit, poll=1)


def test_submit_with_backoff_gives_up_eventually(monkeypatch):
    """Waiting forever would hide a queue that is never going to drain."""
    import scripts.export_s2_composites as X
    monkeypatch.setattr(X.time, "sleep", lambda s: None)

    def submit():
        raise Exception("Too many tasks already in the queue (3002, limit 3000).")

    with pytest.raises(Exception, match="Too many tasks"):
        X.submit_with_backoff(None, submit, poll=1, max_wait_s=3)
