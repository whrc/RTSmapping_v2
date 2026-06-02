"""MLflow run setup, param logging, and artifact logging.

Tracking backend is the GCS file store at `configs/baseline.yaml:mlflow.tracking_uri`
— no separate tracking server. View locally with
`mlflow ui --backend-store-uri <uri>` against that same URI.

Multi-seed runs execute sequentially (plan risk #14) — concurrent writes to
the GCS store are not atomic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import yaml

logger = logging.getLogger(__name__)


def config_sha(cfg: dict) -> str:
    """SHA256 of the canonicalised YAML dump — stable across key reorderings."""
    canonical = yaml.safe_dump(cfg, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_sha_and_dirty(repo_path: Path | None = None) -> tuple[str, bool]:
    """Return (HEAD SHA, is_dirty). ('unknown', False) when git is unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path) if repo_path else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        diff = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_path) if repo_path else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha, bool(diff)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def _flatten_params(cfg: Any, prefix: str = "") -> dict[str, str]:
    """Flatten a nested cfg dict into dot-notation keys for MLflow params.

    MLflow has a 500-char value limit per param; lists are stringified.
    """
    flat: dict[str, str] = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            flat.update(_flatten_params(v, key))
    elif isinstance(cfg, list):
        flat[prefix] = str(cfg)[:500]
    else:
        flat[prefix] = str(cfg)[:500]
    return flat


def setup_mlflow(cfg: dict) -> mlflow.ActiveRun:
    """Start an MLflow run and log boilerplate (config, git SHA).

    Returns the active run (caller must close via `mlflow.end_run()` or use as
    a context manager upstream). The config YAML is logged as an artifact and
    its SHA256 goes into both params and tags.
    """
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    run = mlflow.start_run(run_name=cfg["mlflow"].get("run_name"))

    sha, dirty = git_sha_and_dirty()
    cfg_sha = config_sha(cfg)
    mlflow.set_tags({
        "git_sha": sha,
        "git_dirty": str(dirty).lower(),
        "config_sha": cfg_sha,
    })

    params = _flatten_params(cfg)
    # MLflow has a 100-param batch limit on older versions; slice.
    items = list(params.items())
    for i in range(0, len(items), 100):
        mlflow.log_params(dict(items[i:i + 100]))

    logger.info(
        "MLflow run started: id=%s, experiment=%s, git_sha=%s%s, config_sha=%s",
        run.info.run_id, cfg["mlflow"]["experiment_name"], sha[:12],
        " (dirty)" if dirty else "", cfg_sha[:12],
    )
    return run


def log_config_artifact(cfg: dict, tmp_dir: Path) -> Path:
    """Write cfg to disk and log as an MLflow artifact. Returns the file path."""
    path = tmp_dir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
    mlflow.log_artifact(str(path))
    return path


def log_requirements_frozen(tmp_dir: Path) -> Path | None:
    """Capture `pip freeze` into requirements_frozen.txt and log as artifact.

    Returns None and emits a warning if pip isn't importable (e.g. stripped env).
    """
    try:
        import subprocess as sp
        out = sp.check_output(["pip", "freeze"], text=True)
        path = tmp_dir / "requirements_frozen.txt"
        path.write_text(out)
        mlflow.log_artifact(str(path))
        return path
    except (sp.CalledProcessError, FileNotFoundError, ImportError) as e:
        logger.warning("Skipping requirements_frozen.txt: %s", e)
        return None


def log_metrics_step(metrics: dict[str, float], step: int) -> None:
    """Log a metrics dict at a given step, skipping non-numeric entries."""
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    if numeric:
        mlflow.log_metrics(numeric, step=step)


def log_run_summary(
    cfg: dict,
    final_metrics: dict[str, float],
    training_duration_s: float,
    nan_events: list[dict],
    tmp_dir: Path,
) -> Path:
    """Write a human-readable run_summary.md and log it as an MLflow artifact."""
    path = tmp_dir / "run_summary.md"
    lines = [
        f"# Run summary — {cfg['mlflow'].get('run_name', 'unnamed')}",
        "",
        f"- Experiment: `{cfg['mlflow']['experiment_name']}`",
        f"- Seed: {cfg.get('seed')}",
        f"- Precision: {cfg['training'].get('precision')}",
        f"- Training duration: {training_duration_s:.1f} s ({training_duration_s / 3600:.2f} h)",
        "",
        "## Final metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in sorted(final_metrics.items()):
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.6f} |")
        else:
            lines.append(f"| {k} | {v} |")

    if nan_events:
        lines += ["", "## NaN / Inf events", ""]
        for ev in nan_events:
            lines.append(f"- {ev}")

    path.write_text("\n".join(lines))
    mlflow.log_artifact(str(path))
    return path
