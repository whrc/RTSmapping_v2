"""Validate `versions.yaml` and that the repo actually uses it.

`versions.yaml` (repo root) is the SSoT for what every data version (D<maj>.<min>),
model version (M<maj>.<min>) and inference epoch means. This script keeps the rest of the
repo honest about it. Three checks, no GPU or network needed:

1. Integrity  — ids well-formed and unique, references resolve, exactly one `deployed`
                model and one `current` data version, `rejected` entries carry a verdict.
2. Usage      — `data.version` in configs, `data/version.json`, and the data roots in
                `configs/baseline.yaml` agree with the registry; every D/M id mentioned in
                docs or code exists.
3. No relapse — retired legacy names (`v1.0`, `v1.1`, `v2.1`, ...) no longer appear as bare
                words in docs or code. Physical paths that embed them (`training/v1.0`,
                `RTS_MODEL_V21`) are not bare words and are deliberately left alone.

Usage:
    python scripts/check_versions.py            # exit 0 = clean, 1 = problems listed
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

STATUSES = {"deployed", "current", "superseded", "rejected"}
VERDICTS = {"no-improvement", "negative"}
ID_FORMAT = {"data": re.compile(r"D\d+\.\d+"), "models": re.compile(r"M\d+\.\d+")}
# Any D/M id as written in prose or code (not part of a longer word or number).
ID_MENTION = re.compile(r"(?<![\w.])[DM]\d+\.\d+(?![\w]|\.\d)")
# Where bare-word / id scanning looks, and what it skips (history, finished-run files, fixtures).
SCAN_SUFFIXES = {".md", ".py"}
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", "node_modules", "outputs", "plots", "configs"}
SKIP_PREFIXES = ("docs/archive/", "docs/superpowers/")
SKIP_FILES = {"versions.yaml", "scripts/check_versions.py", "tests/test_check_versions.py"}


def as_list(value: Any) -> list[str]:
    """Return `value` as a list (a registry field may be one id or a list of ids)."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def check_integrity(reg: dict, root: Path) -> list[str]:
    """Check registry structure: ids, references, uniqueness, status rules, doc paths."""
    problems: list[str] = []
    ids: dict[str, str] = {}  # id -> kind
    for kind in ("data", "models"):
        for e in reg.get(kind, []):
            i = e.get("id", "")
            if not ID_FORMAT[kind].fullmatch(i):
                problems.append(f"versions.yaml: bad {kind} id {i!r}")
            if i in ids:
                problems.append(f"versions.yaml: duplicate id {i}")
            ids[i] = kind
            if e.get("status") not in STATUSES:
                problems.append(f"versions.yaml: {i} has invalid status {e.get('status')!r}")
            if e.get("status") == "rejected" and e.get("verdict") not in VERDICTS:
                problems.append(f"versions.yaml: {i} is rejected but verdict is not one of {sorted(VERDICTS)}")
            if e.get("doc") and not (root / e["doc"]).exists():
                problems.append(f"versions.yaml: {i} doc {e['doc']} does not exist")
    for kind in ("data", "models"):
        for e in reg.get(kind, []):
            if e.get("parent") and ids.get(e["parent"]) != kind:
                problems.append(f"versions.yaml: {e['id']} parent {e['parent']} is not a {kind} id")
    for e in reg.get("models", []):
        refs = as_list(e.get("data"))
        if not refs:
            problems.append(f"versions.yaml: model {e['id']} names no data")
        problems += [f"versions.yaml: model {e['id']} data {r} is not a data id"
                     for r in refs if ids.get(r) != "data"]
    years: set[int] = set()
    for e in reg.get("epochs", []):
        if e.get("year") in years:
            problems.append(f"versions.yaml: duplicate epoch {e.get('year')}")
        years.add(e.get("year"))
        if ids.get(e.get("model")) != "models":
            problems.append(f"versions.yaml: epoch {e.get('year')} model {e.get('model')} is not a model id")
    for name, target in reg.get("retired", {}).items():
        if target not in ids:
            problems.append(f"versions.yaml: retired name {name!r} maps to unknown id {target}")
    for kind, status in (("models", "deployed"), ("data", "current")):
        n = sum(e.get("status") == status for e in reg.get(kind, []))
        if n != 1:
            problems.append(f"versions.yaml: expected exactly one {status} entry in {kind}, found {n}")
    return problems


def iter_files(root: Path):
    """Yield (relative posix path, absolute path) for every scanned doc/code file."""
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts[:-1])
        if (not path.is_file() or path.suffix not in SCAN_SUFFIXES or parts & SKIP_DIRS
                or rel.startswith(SKIP_PREFIXES) or rel in SKIP_FILES):
            continue
        yield rel, path


def current_data_root(reg: dict) -> str:
    """gcs_root of the `current` data version."""
    return next(e["gcs_root"] for e in reg["data"] if e["status"] == "current")


def check_usage(reg: dict, root: Path) -> list[str]:
    """Check that configs, data/version.json and prose agree with the registry."""
    problems: list[str] = []
    data_ids = {e["id"]: e for e in reg["data"]}
    all_ids = data_ids.keys() | {e["id"] for e in reg["models"]}

    for cfg in sorted((root / "configs").rglob("*.yaml")):
        try:
            version = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("data", {}).get("version")
        except (yaml.YAMLError, AttributeError, OSError):
            continue
        if version is not None and version not in data_ids:
            problems.append(f"{cfg.relative_to(root).as_posix()}: data.version {version!r} is not a data id")

    spec = root / "data" / "version.json"
    if spec.exists():
        info = json.loads(spec.read_text(encoding="utf-8"))
        entry = data_ids.get(info.get("version"))
        if entry is None:
            problems.append(f"data/version.json: version {info.get('version')!r} is not a data id")
        elif info.get("gcs_root") != entry.get("gcs_root"):
            problems.append(f"data/version.json: gcs_root {info.get('gcs_root')!r} != registry {entry.get('gcs_root')!r}")

    baseline = root / "configs" / "baseline.yaml"
    if baseline.exists():
        cfg = yaml.safe_load(baseline.read_text(encoding="utf-8"))
        want = current_data_root(reg)
        for section, key in (("data", "data_root"), ("data", "normalization_stats_path"), ("splits", "regions_geojson")):
            value = cfg.get(section, {}).get(key, "")
            if not value.startswith(want):
                problems.append(f"configs/baseline.yaml: {section}.{key} {value!r} is not under current data root {want}")

    for rel, path in iter_files(root):
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            problems += [f"{rel}:{n}: {m} is not in versions.yaml" for m in ID_MENTION.findall(line) if m not in all_ids]
    return problems


def retired_pattern(name: str) -> re.Pattern[str]:
    """Regex matching `name` only as a bare word (not inside a path, filename, branch name or longer token)."""
    return re.compile(r"(?<![/\w.\-])" + re.escape(name) + r"(?![\w/]|\.\d|-[A-Za-z])")


def check_retired(reg: dict, root: Path) -> list[str]:
    """Flag retired legacy names used as bare words in docs or code."""
    patterns = {name: retired_pattern(name) for name in reg.get("retired", {})}
    problems: list[str] = []
    for rel, path in iter_files(root):
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for name, pat in patterns.items():
                if pat.search(line):
                    problems.append(f"{rel}:{n}: retired name '{name}' -> use {reg['retired'][name]}")
    return problems


def run(root: Path) -> list[str]:
    """Run all three checks against the repo at `root`."""
    reg = yaml.safe_load((root / "versions.yaml").read_text(encoding="utf-8"))
    problems = check_integrity(reg, root)
    if problems:  # later checks assume a well-formed registry
        return problems
    return check_usage(reg, root) + check_retired(reg, root)


def main() -> int:
    """CLI entry point: log every problem, exit 1 if any."""
    problems = run(Path(__file__).resolve().parent.parent)
    for p in problems:
        logger.error(p)
    logger.info("versions OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
