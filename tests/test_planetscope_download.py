"""Acquisition scripts (planetscope-download/).

GCS-free and network-free: the Planet API is a fake session, so the retry
policy, the domain clip and the flatten mapping are all exercised on canned
input. Covers the four changes Heidi asked for in the PR #61 review.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely as shp

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load(stem: str):
    """Import a module from the hyphenated package dir (not a valid identifier)."""
    spec = importlib.util.spec_from_file_location(
        f"psd_{stem}", REPO / "planetscope-download" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


order_basemaps = _load("order_basemaps")
filter_to_domain = _load("filter_to_domain")
tidy_rename = _load("tidy_rename")
check_status = _load("check_status")
alert_if_stopped = _load("alert_if_stopped")


# ---------------------------------------------------------------------------
# retry policy (Heidi's change 4)
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code, self.text = status_code, text


class FakeSession:
    """Replays a scripted list of statuses, recording how many calls it saw."""

    def __init__(self, statuses):
        self.statuses, self.calls = list(statuses), 0

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        code = self.statuses[min(self.calls - 1, len(self.statuses) - 1)]
        if isinstance(code, Exception):
            raise code
        return FakeResponse(code, text=f"body {code}")


ROW = pd.Series({"id": "338-1474", "basemap_name": "global_quarterly_2022q3_mosaic",
                 "delivery_location": "global_quarterly/2022/q3/338/1474/"})


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Back-off is the point of the test, not something to actually wait for."""
    monkeypatch.setattr(order_basemaps.time, "sleep", lambda s: None)


def test_202_succeeds_first_try():
    s = FakeSession([202])
    assert order_basemaps.place_order(s, ROW, "b", "k") == ("ordered", "")
    assert s.calls == 1


def test_401_fails_fast_without_retrying():
    """Auth cannot be retried into working; burning 5 attempts only delays the fix."""
    s = FakeSession([401])
    with pytest.raises(PermissionError, match="restart with a fresh key"):
        order_basemaps.place_order(s, ROW, "b", "k")
    assert s.calls == 1


def test_transient_status_retries_then_succeeds():
    s = FakeSession([500, 409, 202])
    assert order_basemaps.place_order(s, ROW, "b", "k")[0] == "ordered"
    assert s.calls == 3


def test_retries_are_bounded_then_recorded_as_failed():
    """The run must not abort: exhaustion returns 'failed' so the loop continues."""
    s = FakeSession([503])
    outcome, detail = order_basemaps.place_order(s, ROW, "b", "k")
    assert outcome == "failed"
    assert s.calls == order_basemaps.MAX_ATTEMPTS
    assert "503" in detail


def test_non_retryable_status_gives_up_immediately():
    s = FakeSession([404])
    assert order_basemaps.place_order(s, ROW, "b", "k")[0] == "failed"
    assert s.calls == 1


def test_connection_errors_are_retried():
    import requests
    s = FakeSession([requests.ConnectionError("reset"), 202])
    assert order_basemaps.place_order(s, ROW, "b", "k")[0] == "ordered"
    assert s.calls == 2


def test_order_payload_matches_the_2025_delivery_shape():
    """The COG file_format tool is what makes filenames match the 2025 delivery."""
    captured = {}

    class Capturing(FakeSession):
        def post(self, url, json=None, timeout=None):
            captured.update(json)
            captured["_url"], captured["_timeout"] = url, timeout
            return FakeResponse(202)

    order_basemaps.place_order(Capturing([202]), ROW, "pdg-planet-data", "secret")
    assert captured["source_type"] == "basemaps"
    assert captured["products"][0]["quad_ids"] == ["338-1474"]
    assert captured["tools"] == [{"file_format": {"format": "COG"}}]
    gcs = captured["delivery"]["google_cloud_storage"]
    assert (gcs["bucket"], gcs["credentials"]) == ("pdg-planet-data", "secret")
    assert gcs["path_prefix"] == "global_quarterly/2022/q3/338/1474/"
    assert captured["_timeout"] is not None, "every call must be time-bounded"


# ---------------------------------------------------------------------------
# progress / status
# ---------------------------------------------------------------------------

def test_progress_counts_and_writes_status(tmp_path):
    p = order_basemaps.Progress(2022, 4, tmp_path / "2022.json")
    p.record("ordered", "1-1")
    p.record("skipped", "1-2")
    p.record("failed", "1-3")
    p.write()
    snap = p.snapshot()
    assert (snap["n_ordered"], snap["n_skipped"], snap["n_failed"]) == (1, 1, 1)
    assert snap["n_done"] == 3 and snap["pct_done"] == 75.0
    assert (tmp_path / "2022.json").exists()


def test_status_write_failure_does_not_kill_the_run(tmp_path):
    """Monitoring is not allowed to take down a five-day job."""
    bad = tmp_path / "nope"
    bad.write_text("i am a file, not a directory")
    p = order_basemaps.Progress(2022, 1, bad / "2022.json")
    p.write()  # must not raise


def test_watchdog_timestamp_advances_on_progress():
    p = order_basemaps.Progress(2022, 2, Path("/dev/null"))
    before = p.last_active[0]
    p.record("ordered", "1-1")
    assert p.last_active[0] >= before


# ---------------------------------------------------------------------------
# domain clip (R -> geopandas port)
# ---------------------------------------------------------------------------

def test_filter_to_domain_clips_and_derives_columns():
    grids = gpd.GeoDataFrame(
        {"name": ["global_quarterly_2022q3_mosaic"] * 2,
         "id": ["10-20", "900-800"], "link": ["u1", "u2"], "percent_covered": [100, 100]},
        geometry=[shp.geometry.box(0, 0, 1, 1), shp.geometry.box(50, 50, 51, 51)],
        crs="EPSG:4326")
    domain = gpd.GeoDataFrame(geometry=[shp.geometry.box(-1, -1, 2, 2)], crs="EPSG:4326")

    out = filter_to_domain.filter_to_domain(grids, domain, 2022)

    assert list(out["id"]) == ["10-20"]                       # the far quad is dropped
    assert (out["grid_column"][0], out["grid_row"][0]) == (10, 20)
    assert out["basemap_name"][0] == "global_quarterly_2022q3_mosaic"
    assert out["delivery_location"][0] == "global_quarterly/2022/q3/10/20/"


def test_filter_to_domain_survives_an_empty_clip():
    """A grid file that misses the domain entirely must report cleanly, not
    KeyError deep in the column derivation (found by the end-to-end smoke)."""
    grids = gpd.GeoDataFrame({"id": ["10-20"], "link": ["u"]},
                             geometry=[shp.geometry.box(0, 0, 1, 1)], crs="EPSG:4326")
    far = gpd.GeoDataFrame(geometry=[shp.geometry.box(50, 50, 51, 51)], crs="EPSG:4326")
    out = filter_to_domain.filter_to_domain(grids, far, 2022)
    assert out.empty
    assert "delivery_location" in out.columns   # schema intact for the caller


def test_filter_to_domain_sorts_by_column_then_row():
    boxes = [shp.geometry.box(0, 0, 1, 1)] * 3
    grids = gpd.GeoDataFrame({"id": ["5-9", "2-3", "5-1"], "link": ["a", "b", "c"]},
                             geometry=boxes, crs="EPSG:4326")
    domain = gpd.GeoDataFrame(geometry=[shp.geometry.box(-1, -1, 2, 2)], crs="EPSG:4326")
    out = filter_to_domain.filter_to_domain(grids, domain, 2022)
    assert list(out["id"]) == ["2-3", "5-1", "5-9"]


# ---------------------------------------------------------------------------
# tidy_rename mapping
# ---------------------------------------------------------------------------

def test_flatten_name_strips_uuid_and_folds_mosaic_dir():
    raw = ("global_quarterly/2022/q3/338/1474/"
           "2a2ddc73-1111-2222-3333-444455556666/"
           "global_quarterly_2022q3_mosaic/338-1474_quad_file_format.tif")
    assert tidy_rename.flatten_name(raw) == (
        "global_quarterly/2022/q3/338/1474/"
        "global_quarterly_2022q3_mosaic_338-1474_quad_file_format.tif")


def test_flatten_name_is_idempotent():
    """Re-running the tidy-up must be a no-op, not a second mangling."""
    flat = ("global_quarterly/2022/q3/338/1474/"
            "global_quarterly_2022q3_mosaic_338-1474_quad_file_format.tif")
    assert tidy_rename.flatten_name(flat) == flat


def test_flattened_names_still_index():
    """Whatever the tidy-up produces must remain matchable by the quad indexer."""
    from inference.quad_index import _QUAD_NAME_RE
    raw = ("global_quarterly/2022/q3/338/1474/2a2ddc73-1111-2222-3333-444455556666/"
           "global_quarterly_2022q3_mosaic/338-1474_quad.tif")
    for name in (raw, tidy_rename.flatten_name(raw)):
        m = _QUAD_NAME_RE.search(name.rsplit("/", 1)[-1])
        assert m and (m.group(1), m.group(2)) == ("338", "1474")


# ---------------------------------------------------------------------------
# check_status rendering
# ---------------------------------------------------------------------------

def _status(tmp_path, year, done, total, heartbeat_age_s):
    import json
    from datetime import datetime, timedelta, timezone
    hb = datetime.now(timezone.utc) - timedelta(seconds=heartbeat_age_s)
    (tmp_path / f"{year}.json").write_text(json.dumps({
        "year": year, "started_at": hb.isoformat(), "heartbeat_at": hb.isoformat(),
        "n_total": total, "n_done": done, "n_ordered": done, "n_skipped": 0,
        "n_failed": 0, "pct_done": 100 * done / total, "last_quad_id": "1-1",
        "orders_per_min": 39.0, "eta_hours": 0.0}))


def test_finished_year_is_not_reported_stale(tmp_path, capsys, monkeypatch):
    """A completed run stops heartbeating by design. Flagging it STALE would send
    the operator chasing a process that actually succeeded."""
    _status(tmp_path, 2019, done=100, total=100, heartbeat_age_s=7200)
    monkeypatch.setattr("sys.argv", ["check_status.py", "--status-dir", str(tmp_path)])
    check_status.main()
    out = capsys.readouterr().out
    assert "complete" in out and "STALE" not in out


def test_incomplete_and_quiet_is_reported_stale(tmp_path, capsys, monkeypatch):
    _status(tmp_path, 2022, done=10, total=100, heartbeat_age_s=7200)
    monkeypatch.setattr("sys.argv", ["check_status.py", "--status-dir", str(tmp_path)])
    check_status.main()
    assert "STALE" in capsys.readouterr().out


def test_live_run_is_flagged_neither(tmp_path, capsys, monkeypatch):
    _status(tmp_path, 2022, done=10, total=100, heartbeat_age_s=30)
    monkeypatch.setattr("sys.argv", ["check_status.py", "--status-dir", str(tmp_path)])
    check_status.main()
    out = capsys.readouterr().out
    assert "STALE" not in out and "complete" not in out


# ---------------------------------------------------------------------------
# runtime output location
# ---------------------------------------------------------------------------

def test_runtime_outputs_default_outside_the_repo():
    """The checkout is shared and read-only to collaborators: writing runtime
    output into it fails for everyone but its owner (Heidi hit exactly this on
    her first run, 2026-08-18)."""
    repo = str(REPO)
    for default in (order_basemaps.DEFAULT_WORK, check_status.DEFAULT_STATUS_DIR):
        assert not str(default).startswith(repo), f"{default} is inside the repo"


def test_psd_work_env_var_overrides_the_default(monkeypatch):
    """PSD_WORK is the documented escape hatch when /mnt/outputs is unavailable."""
    monkeypatch.setenv("PSD_WORK", "/tmp/psd-elsewhere")
    reloaded = _load("order_basemaps")
    assert str(reloaded.DEFAULT_WORK) == "/tmp/psd-elsewhere"


# ---------------------------------------------------------------------------
# stopped-run alerting
# ---------------------------------------------------------------------------

def _status_file(tmp_path, year, done, total, heartbeat_age_s):
    import json
    from datetime import datetime, timedelta, timezone
    hb = (datetime.now(timezone.utc) - timedelta(seconds=heartbeat_age_s)).isoformat()
    (tmp_path / f"{year}.json").write_text(json.dumps({
        "year": year, "started_at": hb, "heartbeat_at": hb, "n_total": total,
        "n_done": done, "n_ordered": done, "n_skipped": 0, "n_failed": 0,
        "pct_done": 100 * done / total, "last_quad_id": "1-1",
        "orders_per_min": 39.0, "eta_hours": 1.0}))


def _run_alert(tmp_path, monkeypatch, alive: bool):
    posted = []
    monkeypatch.setattr(alert_if_stopped, "ordering_process_alive", lambda y: alive)
    monkeypatch.setattr(alert_if_stopped, "webhook_url", lambda: "https://hook.test")
    monkeypatch.setattr(alert_if_stopped, "post",
                        lambda url, text, dry: posted.append(text))
    monkeypatch.setattr("sys.argv",
                        ["alert_if_stopped.py", "--status-dir", str(tmp_path)])
    alert_if_stopped.main()
    return posted


def test_silent_while_the_run_is_healthy(tmp_path, monkeypatch):
    _status_file(tmp_path, 2022, done=10, total=100, heartbeat_age_s=30)
    assert _run_alert(tmp_path, monkeypatch, alive=True) == []


def test_silent_when_stale_but_process_still_alive(tmp_path, monkeypatch):
    """On resume the loop lists the delivery prefix before its first heartbeat,
    which can take the best part of an hour. Alerting on that would cry wolf on
    every restart, so a live process suppresses the alarm."""
    _status_file(tmp_path, 2022, done=10, total=100, heartbeat_age_s=7200)
    assert _run_alert(tmp_path, monkeypatch, alive=True) == []


def test_alerts_when_stale_and_no_process(tmp_path, monkeypatch):
    _status_file(tmp_path, 2022, done=10, total=100, heartbeat_age_s=7200)
    posted = _run_alert(tmp_path, monkeypatch, alive=False)
    assert len(posted) == 1
    assert "has stopped" in posted[0] and "run_year.sh 2022" in posted[0]


def test_alert_is_not_repeated_every_tick(tmp_path, monkeypatch):
    _status_file(tmp_path, 2022, done=10, total=100, heartbeat_age_s=7200)
    assert len(_run_alert(tmp_path, monkeypatch, alive=False)) == 1
    assert _run_alert(tmp_path, monkeypatch, alive=False) == []


def test_recovery_rearms_the_alert(tmp_path, monkeypatch):
    _status_file(tmp_path, 2022, done=10, total=100, heartbeat_age_s=7200)
    assert len(_run_alert(tmp_path, monkeypatch, alive=False)) == 1
    _status_file(tmp_path, 2022, done=20, total=100, heartbeat_age_s=10)   # resumed
    assert _run_alert(tmp_path, monkeypatch, alive=True) == []
    _status_file(tmp_path, 2022, done=20, total=100, heartbeat_age_s=7200)  # died again
    assert len(_run_alert(tmp_path, monkeypatch, alive=False)) == 1


def test_completion_alerts_once(tmp_path, monkeypatch):
    _status_file(tmp_path, 2022, done=100, total=100, heartbeat_age_s=7200)
    posted = _run_alert(tmp_path, monkeypatch, alive=False)
    assert len(posted) == 1 and "complete" in posted[0]
    assert "--expect-quads 100" in posted[0]
    assert _run_alert(tmp_path, monkeypatch, alive=False) == []
