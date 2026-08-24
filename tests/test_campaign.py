"""Tests for the interannual campaign coordinator (campaign/).

The coordinator's whole job is to be trustworthy about what has and has not run
across a months-long, two-person campaign. So these tests concentrate on the ways
it could lie: losing state on a crashed write, re-running finished work, running a
stage whose inputs are not ready, walking past a human gate, or alerting either not
at all or every ten minutes forever.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from campaign import alert, stages as S, state as ST  # noqa: E402
from campaign import run_stage, status  # noqa: E402


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    """A campaign config pointing entirely inside tmp_path."""
    return {
        "years": [2022],
        "paths": {"work": str(tmp_path / "work"),
                  "local_inference": str(tmp_path / "inf"),
                  "acquisition_status": str(tmp_path / "status_{year}.json"),
                  "quad_baseline": str(tmp_path / "inf" / "baseline.csv"),
                  "s2_bucket": "test-bucket",
                  "s2_prefix": "S2_RGB/{year}_south",
                  "state_mirror": None},
        "expect": {"n_quads": 309100, "quad_tolerance": 0.01, "n_s2_cells": 1799,
                   "n_tiles": 41567572, "shard_size": 20000, "drift_sample_quads": 300},
        "docker": {"project": "gcs-billing-proj", "ee_project": "ee-compute-proj",
                   "train_image": "img", "dataprep_image": "dimg"},
        "alerts": {"stale_after_s": 1800, "no_progress_after_s": 21600,
                   "webhook_file": str(tmp_path / "hook")},
    }


@pytest.fixture
def work(cfg: dict) -> Path:
    return Path(cfg["paths"]["work"])


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def test_state_round_trip(work: Path):
    st = ST.load(work, 2022, S.NAMES)
    assert all(e["status"] == "pending" for e in st["stages"].values())
    ST.save(work, 2022, st)
    assert json.loads(ST.state_path(work, 2022).read_text())["year"] == 2022


def test_save_is_atomic_no_tmp_left_behind(work: Path):
    ST.save(work, 2022, ST.load(work, 2022, S.NAMES))
    assert not list(ST.state_path(work, 2022).parent.glob("*.tmp"))


def test_new_stage_added_later_reads_as_pending(work: Path):
    ST.save(work, 2022, ST.skeleton(2022, ["acquire"]))
    st = ST.load(work, 2022, S.NAMES)
    assert st["stages"]["infer"]["status"] == "pending"
    assert st["stages"]["acquire"]["status"] == "pending"


def test_set_stage_rejects_unknown_status(work: Path):
    with pytest.raises(ValueError, match="unknown status"):
        ST.set_stage(work, 2022, "acquire", S.NAMES, "finished-ish")


def test_running_sets_heartbeat_done_sets_finished(work: Path):
    ST.set_stage(work, 2022, "acquire", S.NAMES, "running")
    e = ST.load(work, 2022, S.NAMES)["stages"]["acquire"]
    assert e["heartbeat_at"] and "finished_at" not in e
    ST.set_stage(work, 2022, "acquire", S.NAMES, "done", evidence={"n_ordered": 7})
    e = ST.load(work, 2022, S.NAMES)["stages"]["acquire"]
    assert e["finished_at"] and e["evidence"]["n_ordered"] == 7


# --------------------------------------------------------------------------
# prerequisites and idempotency — the things that protect $30/hr compute
# --------------------------------------------------------------------------
def _stub(name: str, argv: list[str], **kw) -> S.Stage:
    """Replace a real stage's command with a harmless one."""
    return dataclasses.replace(S.get(name), cmd=lambda y, c: argv, evidence=None, **kw)


@pytest.fixture
def stub_table(monkeypatch):
    """Swap in stub commands so tests never invoke docker."""
    def _apply(**overrides: S.Stage):
        table = dict(S.BY_NAME)
        table.update(overrides)
        monkeypatch.setattr(S, "BY_NAME", table)
    return _apply


def test_refuses_stage_with_unmet_prereq(cfg, stub_table, caplog):
    stub_table(quad_index=_stub("quad_index", ["true"]))
    rc = run_stage.run(2022, "quad_index", cfg)
    assert rc == 2
    assert "acquire" in caplog.text  # names what is missing


def test_runs_once_prereq_is_done(cfg, work, stub_table):
    stub_table(quad_index=_stub("quad_index", ["true"]))
    ST.set_stage(work, 2022, "acquire", S.NAMES, "done")
    assert run_stage.run(2022, "quad_index", cfg) == 0
    assert ST.load(work, 2022, S.NAMES)["stages"]["quad_index"]["status"] == "done"


def test_done_stage_is_a_noop_without_force(cfg, work, stub_table, tmp_path):
    marker = tmp_path / "ran"
    stub_table(quad_index=_stub("quad_index", ["touch", str(marker)]))
    ST.set_stage(work, 2022, "acquire", S.NAMES, "done")
    ST.set_stage(work, 2022, "quad_index", S.NAMES, "done")
    assert run_stage.run(2022, "quad_index", cfg) == 0
    assert not marker.exists(), "a done stage must not re-run"


def test_force_reruns_a_done_stage(cfg, work, stub_table, tmp_path):
    marker = tmp_path / "ran"
    stub_table(quad_index=_stub("quad_index", ["touch", str(marker)]))
    ST.set_stage(work, 2022, "acquire", S.NAMES, "done")
    ST.set_stage(work, 2022, "quad_index", S.NAMES, "done")
    assert run_stage.run(2022, "quad_index", cfg, force=True) == 0
    assert marker.exists()


def test_failure_records_exit_code_and_log(cfg, work, stub_table):
    stub_table(quad_index=_stub("quad_index", ["false"]))
    ST.set_stage(work, 2022, "acquire", S.NAMES, "done")
    assert run_stage.run(2022, "quad_index", cfg) != 0
    e = ST.load(work, 2022, S.NAMES)["stages"]["quad_index"]
    assert e["status"] == "failed" and e["exit_code"] != 0
    assert Path(e["log"]).is_file()


def test_external_stage_is_not_run_here(cfg):
    rc = run_stage.run(2022, "acquire", cfg)
    assert rc == 2, "acquire is Heidi's to run; the driver must refuse it"


def test_dry_run_executes_nothing(cfg, work, stub_table, tmp_path, capsys):
    marker = tmp_path / "ran"
    stub_table(quad_index=_stub("quad_index", ["touch", str(marker)]))
    ST.set_stage(work, 2022, "acquire", S.NAMES, "done")
    assert run_stage.run(2022, "quad_index", cfg, dry_run=True) == 0
    assert not marker.exists()
    assert "touch" in capsys.readouterr().out


# --------------------------------------------------------------------------
# gates — the reason the driver exists rather than a shell script
# --------------------------------------------------------------------------
def test_gate_stage_ends_blocked_not_done(cfg, work, stub_table):
    stub_table(drift_check=_stub("drift_check", ["true"]))
    ST.set_stage(work, 2022, "acquire", S.NAMES, "done")
    ST.set_stage(work, 2022, "quad_index", S.NAMES, "done")
    run_stage.run(2022, "drift_check", cfg)
    assert ST.load(work, 2022, S.NAMES)["stages"]["drift_check"]["status"] == "blocked"


def test_blocked_gate_blocks_its_dependents(cfg, work, stub_table):
    stub_table(infer=_stub("infer", ["true"]))
    for n in ("acquire", "quad_index", "s2_export", "s2_index", "tile_grid", "shard"):
        ST.set_stage(work, 2022, n, S.NAMES, "done")
    ST.set_stage(work, 2022, "drift_check", S.NAMES, "blocked")
    assert run_stage.run(2022, "infer", cfg) == 2, "inference must not start behind a gate"


def test_sign_off_clears_the_gate(cfg, work):
    ST.set_stage(work, 2022, "drift_check", S.NAMES, "blocked")
    run_stage.mark(2022, "drift_check", cfg, "done")
    assert ST.load(work, 2022, S.NAMES)["stages"]["drift_check"]["status"] == "done"


def test_detached_stage_stays_running_after_its_launcher_exits(cfg, work, stub_table):
    """The GEE launcher exits 0 immediately; that is not the work finishing."""
    stub_table(s2_export=_stub("s2_export", ["true"]))
    run_stage.run(2022, "s2_export", cfg)
    assert ST.load(work, 2022, S.NAMES)["stages"]["s2_export"]["status"] == "running"


# --------------------------------------------------------------------------
# evidence parsers, against real file shapes
# --------------------------------------------------------------------------
def test_quad_evidence_counts_rows_not_lines(cfg, tmp_path):
    d = Path(cfg["paths"]["local_inference"])
    d.mkdir(parents=True)
    (d / "quad_index_2022q3.csv").write_text("quad_id,x,y\na,1,2\nb,3,4\n")
    assert S._quad_evidence(2022, cfg) == {"n_quads": 2}


def test_acquire_evidence_reads_the_order_loop_status(cfg, tmp_path):
    Path(str(tmp_path / "status_2022.json")).write_text(json.dumps(
        {"n_ordered": 309109, "n_total": 309109, "n_failed": 0, "n_done": 309109}))
    assert S._acquire_evidence(2022, cfg)["n_ordered"] == 309109
    assert S._acquire_probe(2022, cfg)["pct"] == 100.0


def test_drift_evidence_compares_against_the_quad_baseline(cfg):
    """2022 vs the 2025 QUAD baseline must read as small, unlike vs training tiles."""
    d = Path(cfg["paths"]["local_inference"])
    d.mkdir(parents=True)
    hdr = "channel,train_mean,train_std,sample_mean,sample_std\n"
    (d / "baseline.csv").write_text(hdr + "R,54,36,51.13,38.35\nB,40,30,33.95,35.03\n")
    (d / "drift_report_2022q3.csv").write_text(hdr + "R,54,36,54.79,39.83\nB,40,30,35.29,36.12\n")
    ev = S._drift_evidence(2022, cfg)
    assert ev["worst_mean_drift_sigma"] < 0.5
    assert ev["worst_std_ratio"] < 0.25
    assert ev["baseline"].endswith("baseline.csv")


def test_s2_export_passes_the_EE_project_not_the_gcs_one(cfg):
    """The two projects are different things and conflating them broke the first launch.

    --project is Earth Engine batch-compute quota; GOOGLE_CLOUD_PROJECT is GCS billing
    for the bucket listing. pdg-project-406720 cannot run batch exports at all
    (restricted mode: PENDING 51 min, 0 EECU), so the EE project must be the override.
    """
    argv = S._s2_cmd(2022, cfg)
    assert argv[argv.index("--project") + 1] == "ee-compute-proj"
    assert "GOOGLE_CLOUD_PROJECT=gcs-billing-proj" in argv
    assert "ee-compute-proj" != cfg["docker"]["project"]


def test_unset_ee_project_refuses_loudly(cfg):
    """No authorised EE project must fail fast, never silently borrow another team's.

    Our IAM happens to permit pdg-wg-* working-group projects, but those belong to
    other teams and a year of exports is >=54M EECU-seconds of their quota. Failing
    here is correct; stalling or substituting is not.
    """
    cfg["docker"]["ee_project"] = None
    with pytest.raises(ValueError, match="no Earth Engine project"):
        S._s2_cmd(2022, cfg)


def test_ee_project_guard_names_both_real_options(cfg):
    cfg["docker"]["ee_project"] = None
    with pytest.raises(ValueError) as e:
        S._s2_cmd(2022, cfg)
    msg = str(e.value)
    assert "abruptthawmapping" in msg and "serviceUsageConsumer" in msg
    assert "pdg-wg-" in msg  # explicitly warns them off


def test_docker_wrapper_sets_the_gcs_billing_project(cfg):
    """Omitting it fails with 'Project was not passed and could not be determined'."""
    argv = S.docker("img", ["x.py"], cfg)
    assert "GOOGLE_CLOUD_PROJECT=gcs-billing-proj" in argv


def test_evidence_failure_is_recorded_not_raised(cfg, work, stub_table):
    boom = dataclasses.replace(S.get("quad_index"), cmd=lambda y, c: ["true"],
                               evidence=lambda y, c: 1 / 0)
    stub_table(quad_index=boom)
    ST.set_stage(work, 2022, "acquire", S.NAMES, "done")
    assert run_stage.run(2022, "quad_index", cfg) == 0
    e = ST.load(work, 2022, S.NAMES)["stages"]["quad_index"]
    assert e["status"] == "done" and "evidence_error" in e["evidence"]


# --------------------------------------------------------------------------
# status rendering
# --------------------------------------------------------------------------
def test_matrix_shows_one_row_per_year_and_marks_each_stage(cfg, work):
    ST.set_stage(work, 2022, "acquire", S.NAMES, "done")
    ST.set_stage(work, 2022, "s2_export", S.NAMES, "running")
    snap = status.snapshot(cfg, [2022], live=False)
    out = status.render_matrix(snap)
    assert "2022" in out and "acquire" in out
    assert status._cell(snap["years"][2022]["acquire"]) == "✓"
    assert status._cell(snap["years"][2022]["infer"]) == "·"


def test_cell_shows_percentage_when_a_probe_reported(cfg):
    assert status._cell({"status": "running", "progress": {"pct": 23.4}}) == "▶23%"


def test_year_detail_names_the_gate(cfg, work):
    ST.set_stage(work, 2022, "drift_check", S.NAMES, "blocked", evidence={"worst": 0.1})
    snap = status.snapshot(cfg, [2022], live=False)
    assert "GATE" in status.render_year(2022, snap["years"][2022])


# --------------------------------------------------------------------------
# alerting — must fire, and must not repeat
# --------------------------------------------------------------------------
def test_failure_alerts_once_not_every_tick(cfg, work):
    ST.set_stage(work, 2022, "acquire", S.NAMES, "done")
    ST.set_stage(work, 2022, "quad_index", S.NAMES, "failed", exit_code=1, log="/x.log")
    first, seen = alert.collect(cfg, [2022])
    assert len(first) == 1 and "FAILED" in first[0][1]
    for k, _ in first:
        seen[k] = 1.0
    alert.save_seen(cfg, seen)
    again, _ = alert.collect(cfg, [2022])
    assert again == []


def test_gate_alert_names_the_stage(cfg, work):
    ST.set_stage(work, 2022, "drift_check", S.NAMES, "blocked", evidence={"worst": 0.1})
    incidents, _ = alert.collect(cfg, [2022])
    assert any("drift_check" in m and "sign-off" in m for _, m in incidents)


def test_year_complete_alerts(cfg, work):
    for n in S.NAMES:
        ST.set_stage(work, 2022, n, S.NAMES, "done")
    incidents, _ = alert.collect(cfg, [2022])
    assert any("complete" in m for _, m in incidents)


def test_stale_heartbeat_alone_is_not_stuck(cfg):
    """The false-alarm guard: a detached stage that is still advancing is fine."""
    entry = {"heartbeat_at": "2000-01-01T00:00:00+00:00"}
    prev = {"done": 10, "at": 0.0}
    assert alert.stuck(entry, {"done": 50}, prev, cfg) is False


def test_stale_heartbeat_plus_no_progress_is_stuck(cfg):
    entry = {"heartbeat_at": "2000-01-01T00:00:00+00:00"}
    prev = {"done": 50, "at": 0.0}   # epoch 0 -> long past no_progress_after_s
    assert alert.stuck(entry, {"done": 50}, prev, cfg) is True


def test_first_sighting_never_alerts(cfg):
    """Nothing to compare against yet — must not cry wolf on the first tick."""
    entry = {"heartbeat_at": "2000-01-01T00:00:00+00:00"}
    assert alert.stuck(entry, {"done": 5}, None, cfg) is False


def test_fresh_heartbeat_is_never_stuck(cfg):
    from datetime import datetime, timezone
    entry = {"heartbeat_at": datetime.now(timezone.utc).isoformat()}
    assert alert.stuck(entry, {"done": 0}, {"done": 0, "at": 0.0}, cfg) is False
