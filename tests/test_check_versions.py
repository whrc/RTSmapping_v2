"""Tests for scripts/check_versions.py — the guard that keeps versions.yaml the SSoT.

Synthetic repos under tmp_path; no GPU, no network. One test runs the real repo.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("check_versions", REPO / "scripts" / "check_versions.py")
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)

GOOD = {
    "retired": {"v1.0": "D1.0", "v1.1": "D1.1", "v2.1": "M3.0"},
    "data": [
        {"id": "D1.0", "status": "current", "gcs_root": "gs://b/training/v1.0"},
        {"id": "D1.1", "status": "rejected", "verdict": "no-improvement", "parent": "D1.0"},
    ],
    "models": [
        {"id": "M1.0", "status": "deployed", "data": "D1.0"},
        {"id": "M3.0", "status": "rejected", "verdict": "negative", "parent": "M1.0", "data": ["D1.0", "D1.1"]},
    ],
    "epochs": [{"year": 2025, "model": "M1.0"}],
}


def make_repo(tmp_path: Path, reg: dict | None = None, files: dict[str, str] | None = None) -> Path:
    """Write versions.yaml (+ extra files) into tmp_path and return it."""
    (tmp_path / "versions.yaml").write_text(yaml.safe_dump(reg or GOOD), encoding="utf-8")
    for rel, text in (files or {}).items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def mutated(fn) -> dict:
    """Deep-copy GOOD and apply `fn` to it."""
    reg = copy.deepcopy(GOOD)
    fn(reg)
    return reg


def test_real_repo_is_clean():
    """The checked-in registry, configs, docs and code agree (the guard that matters)."""
    assert cv.run(REPO) == []


def test_good_registry_is_clean(tmp_path):
    assert cv.run(make_repo(tmp_path)) == []


@pytest.mark.parametrize("fn, needle", [
    (lambda r: r["data"].append({"id": "D1.0", "status": "superseded"}), "duplicate id D1.0"),
    (lambda r: r["data"].append({"id": "1.2", "status": "superseded"}), "bad data id"),
    (lambda r: r["models"].append({"id": "D2.0", "status": "superseded", "data": "D1.0"}), "bad models id"),
    (lambda r: r["models"][1].pop("verdict"), "verdict"),
    (lambda r: r["models"][0].update(status="superseded"), "exactly one deployed"),
    (lambda r: r["data"][1].update(status="current", verdict=None), "exactly one current"),
    (lambda r: r["models"][0].update(data="D9.9"), "not a data id"),
    (lambda r: r["models"][1].update(parent="D1.0"), "parent D1.0 is not a models id"),
    (lambda r: r["epochs"][0].update(model="D1.0"), "not a model id"),
    (lambda r: r["retired"].update({"v9.9": "D9.9"}), "unknown id D9.9"),
    (lambda r: r["models"][0].update(status="bogus"), "invalid status"),
], ids=["dup", "bad-data-id", "bad-model-id", "no-verdict", "no-deployed", "two-current",
        "dangling-data", "cross-kind-parent", "epoch-model", "retired-target", "bad-status"])
def test_integrity_rejects(tmp_path, fn, needle):
    problems = cv.run(make_repo(tmp_path, mutated(fn)))
    assert any(needle in p for p in problems), problems


def test_doc_pointer_must_exist(tmp_path):
    reg = mutated(lambda r: r["data"][0].update(doc="docs/missing.md"))
    assert any("does not exist" in p for p in cv.run(make_repo(tmp_path, reg)))


def test_config_data_version_must_be_a_data_id(tmp_path):
    root = make_repo(tmp_path, files={"configs/a.yaml": "data:\n  version: '1.0'\n",
                                       "configs/b.yaml": "data:\n  version: 'D1.0'\n",
                                       "configs/c.yaml": "model: {name: x}\n"})
    problems = cv.run(root)
    assert len(problems) == 1 and "configs/a.yaml" in problems[0]


def test_baseline_roots_must_sit_under_current_data_root(tmp_path):
    base = {"data": {"data_root": "gs://b/training/v1.0",
                     "normalization_stats_path": "gs://other/training/v1.0/n.json"},
            "splits": {"regions_geojson": "gs://b/training/v1.0/r.geojson"}}
    problems = cv.run(make_repo(tmp_path, files={"configs/baseline.yaml": yaml.safe_dump(base)}))
    assert len(problems) == 1 and "normalization_stats_path" in problems[0]


def test_version_json_must_match_registry(tmp_path):
    ok = {"version": "D1.0", "gcs_root": "gs://b/training/v1.0"}
    assert cv.run(make_repo(tmp_path, files={"data/version.json": json.dumps(ok)})) == []
    bad = dict(ok, gcs_root="gs://old/training/v1.0")
    assert any("gcs_root" in p for p in cv.run(make_repo(tmp_path, files={"data/version.json": json.dumps(bad)})))
    old = dict(ok, version="1.0")
    assert any("not a data id" in p for p in cv.run(make_repo(tmp_path, files={"data/version.json": json.dumps(old)})))


def test_unknown_id_mentioned_in_prose_is_flagged(tmp_path):
    root = make_repo(tmp_path, files={"docs/a.md": "Trained on D1.0, scored vs M1.0. See also D7.7.\n"})
    problems = cv.run(root)
    assert len(problems) == 1 and "D7.7" in problems[0] and "docs/a.md:1" in problems[0]


def test_retired_name_flagged_as_bare_word(tmp_path):
    root = make_repo(tmp_path, files={"docs/a.md": "keep v1.0 deployed\n", "pkg/b.py": '"""the v2.1 program"""\n'})
    problems = cv.run(root)
    assert len(problems) == 2
    assert any("docs/a.md:1" in p and "D1.0" in p for p in problems)
    assert any("pkg/b.py:1" in p and "M3.0" in p for p in problems)


def test_legacy_names_inside_paths_and_filenames_are_left_alone(tmp_path):
    text = ("`gs://b/training/v1.0/x`, `docs/v1.0_qc.md`, `configs/v1_1_seed42.yaml`, `RTS_MODEL_V21`, "
            "branch `v2.1-pretraining`, `/mnt/outputs/v1.0`, version v1.0.3\n")
    assert cv.run(make_repo(tmp_path, files={"docs/a.md": text})) == []


def test_history_and_run_configs_are_not_scanned(tmp_path):
    root = make_repo(tmp_path, files={"docs/archive/old.md": "v1.0 v1.1 v2.1\n",
                                       "docs/superpowers/plan.md": "v2.1\n",
                                       "configs/run.yaml": "# trained on v1.0\n"})
    assert cv.run(root) == []
