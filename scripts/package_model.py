"""Bundle a trained MLflow run into a self-contained deployment package.

Per the methodology (training.md §4.3-§4.6, inference.md §2.2), inference
consumes a directory, not a raw checkpoint. This script assembles that
directory from an MLflow run's artifacts + the source repo's baseline config +
a hand-calibrated configs/deployment.yaml.

Output layout:
    {output_dir}/
        weights.pth                  # from best_deployment.pth (EMA state_dict)
        normalization_stats.json     # carries channel-name bindings (training.md §4.5)
        model_config.yaml            # derived from training config (model + channels blocks)
        deployment_config.yaml       # as-supplied (threshold/temperature must be set)
        run_metadata.json            # git SHA, mlflow run id, training date, seed
        requirements_frozen.txt      # from MLflow artifacts if present

Run:
    python scripts/package_model.py \\
        --run-id <mlflow_run_id> \\
        --deployment-config configs/deployment.yaml \\
        --output gs://abruptthawmapping/models/rts-v2-seed42

Refuses to write if deployment_config.yaml has null threshold or temperature
(calibration has not run yet).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logging import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


def _assert_calibration_complete(deployment_cfg: dict) -> None:
    """Refuse to package an un-calibrated deployment config."""
    missing = []
    if deployment_cfg.get("threshold") is None:
        missing.append("threshold")
    if deployment_cfg.get("temperature") is None:
        missing.append("temperature")
    if missing:
        raise ValueError(
            f"deployment_config.yaml has null {missing}; "
            f"run calibration on val (training.md §12) before packaging. "
            f"See configs/deployment.yaml header."
        )


def _extract_model_config(training_cfg: dict) -> dict:
    """Project the training config down to the keys inference needs."""
    return {
        "model": training_cfg["model"],
        "channels": training_cfg["channels"],
        "data": {
            "tile_size": training_cfg["data"]["tile_size"],
            "crs": training_cfg["data"]["crs"],
            "label_ignore_index": training_cfg["data"]["label_ignore_index"],
        },
        "loss": {
            "boundary_handling": training_cfg["loss"]["boundary_handling"],
            "boundary_ignore_width": training_cfg["loss"].get("boundary_ignore_width", 3),
        },
    }


def package_model(
    run_id: str,
    deployment_cfg_path: Path,
    output_dir: Path,
    *,
    best_deployment_artifact: str = "best_deployment.pth",
) -> None:
    """Assemble the deployment package directory from an MLflow run.

    Args:
        run_id: Source MLflow run. The run must be FINISHED and contain the
            best_deployment checkpoint + its normalization_stats.json.
        deployment_cfg_path: Local path to configs/deployment.yaml with
            calibrated threshold + temperature.
        output_dir: Target directory (local or gs:// URI). For gs://, the
            script writes locally to a staging directory and then uploads
            via the google-cloud-storage client — Phase 1 leaves the upload
            as a TODO and writes locally; Phase 2 wires GCS upload when
            deployment is live.
    """
    # Load and validate the deployment config first so we fail fast.
    deployment_cfg = yaml.safe_load(deployment_cfg_path.read_text())
    _assert_calibration_complete(deployment_cfg)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise RuntimeError(
            f"MLflow run {run_id} status={run.info.status}; only FINISHED runs "
            f"are packageable (plan Step 8 guard)."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pull artifacts from the MLflow run.
    staging = output_dir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Deployment checkpoint.
        ckpt_local = Path(client.download_artifacts(run_id, best_deployment_artifact, dst_path=str(staging)))
        import torch
        ckpt = torch.load(ckpt_local, map_location="cpu", weights_only=False)
        weights = ckpt["model_state_dict"]
        torch.save(weights, output_dir / "weights.pth")

        # 2. Normalization stats — carried alongside weights. Channel-name
        # binding (training.md §4.5) is the integrity guarantee at load time;
        # no separate hash is computed or verified here.
        stats_local = Path(client.download_artifacts(run_id, "normalization_stats.json", dst_path=str(staging)))
        shutil.copy2(stats_local, output_dir / "normalization_stats.json")

        # 3. Model config — derived from the source training config.
        # MLflow logs the full config.yaml as a run artifact.
        cfg_local = Path(client.download_artifacts(run_id, "config.yaml", dst_path=str(staging)))
        training_cfg = yaml.safe_load(cfg_local.read_text())
        model_cfg = _extract_model_config(training_cfg)
        (output_dir / "model_config.yaml").write_text(
            yaml.safe_dump(model_cfg, sort_keys=False, default_flow_style=False)
        )

        # 4. Deployment config — as provided, validated.
        shutil.copy2(deployment_cfg_path, output_dir / "deployment_config.yaml")

        # 5. Run metadata.
        meta = {
            "mlflow_run_id": run_id,
            "mlflow_experiment_id": run.info.experiment_id,
            "packaging_date": datetime.now(timezone.utc).isoformat(),
            "training_start_time": datetime.fromtimestamp(
                run.info.start_time / 1000, tz=timezone.utc).isoformat(),
            "git_sha": run.data.tags.get("git_sha", "unknown"),
            "git_dirty": run.data.tags.get("git_dirty", "unknown"),
            "config_sha": run.data.tags.get("config_sha", "unknown"),
            "seed": int(training_cfg.get("seed", 0)),
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_best_metric": ckpt.get("best_metric"),
            "trained_with": ckpt.get("trained_with", {}),
            "channel_names": ckpt.get("channel_names", []),
        }
        (output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

        # 6. requirements_frozen.txt (optional — log warning if absent).
        try:
            req_local = Path(client.download_artifacts(
                run_id, "requirements_frozen.txt", dst_path=str(staging)))
            shutil.copy2(req_local, output_dir / "requirements_frozen.txt")
        except Exception as e:  # noqa: BLE001
            logger.warning("No requirements_frozen.txt in run: %s", e)

    finally:
        shutil.rmtree(staging, ignore_errors=True)

    logger.info("Deployment package written to %s", output_dir)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-id", required=True, help="MLflow run ID")
    p.add_argument("--deployment-config", type=Path, required=True,
                   help="configs/deployment.yaml with calibrated threshold+temperature")
    p.add_argument("--output", type=Path, required=True,
                   help="Local output directory for the deployment package")
    p.add_argument("--tracking-uri", default=None,
                   help="Override MLflow tracking URI (else uses env/default)")
    args = p.parse_args()

    setup_logging(level="INFO")
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    package_model(args.run_id, args.deployment_config, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
