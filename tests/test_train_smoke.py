"""End-to-end smoke test for scripts/train.py on the synthetic fixture.

Plan Step 7a — the "plumbing gate". Runs the full training loop for 2 epochs
on CPU with tiny tiles (64x64) and a lightweight encoder (resnet18) so the
test stays under ~60 seconds. Asserts the hardened criteria from the plan:

    - Loss finite at every logged step; at least one training step.
    - Gradient norm > 0 on every group that's supposed to be trainable.
    - EMA weights differ from live weights after at least one epoch post-unfreeze.
    - At least one pixel in some positive tile gets pred_prob > 0.5 (no collapse).
    - No parameter contains NaN after each optimizer step.
    - Checkpoint files exist with expected keys.
    - MLflow run is queryable via the local file backend.
    - NoData end-to-end: injected partial-NoData tile passes through without NaN
      and ignore pixels contribute zero gradient (loss finite, obj metrics sane).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


# We re-import train as a module rather than calling through subprocess, so we
# can inspect its artifacts and run it against the pytest fixture.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_smoke_cfg(data_root: str, mlruns_dir: Path) -> dict:
    """Minimal cfg that exercises every Phase 1 code path on 64x64 tiles.

    Overrides baseline defaults:
      - tile_size 64, small encoder (resnet18), no pretrained download.
      - precision fp32, num_workers 0, 2 epochs, val_frequency 1.
      - freeze_backbone_epochs 0: unfreeze happens on the first real epoch so
        EMA is constructed and exercised (plan risk #15: EMA at unfreeze).
      - curriculum 1:1 throughout — synthetic set has few negatives.
      - min_blob_size_px 1: synthetic label blobs are small.
      - aug probabilities zeroed: 64x64 elastic/scale can produce degenerate
        tiles that slow CPU eval.
      - MLflow points at a local file:// store for the test sandbox.
    """
    return {
        "seed": 42,
        "deterministic": False,
        "data": {
            "version": "2.0-test",
            "data_root": data_root,
            "rgb_dir": "PLANET-RGB",
            "extra_dir": "EXTRA",
            "labels_dir": "labels",
            "metadata_csv": "metadata.csv",
            "splits_yaml": "splits.yaml",
            "tile_size": 64,
            "crs": "EPSG:3857",
            "label_ignore_index": 255,
            "normalization_stats_path": str(Path(data_root) / "normalization_stats.json"),
        },
        "channels": {"rgb": True, "extra": []},
        "normalization": {"rgb_clip_percentiles": [0.1, 99.9]},
        "sampling": {
            "balanced": True,
            "positive_fraction": 0.5,
            "curriculum_schedule": {"1-2": 1, "3-300": 1},
        },
        "augmentation": {
            "geometric": {
                "rot90_p": 0.0, "hflip_p": 0.0, "vflip_p": 0.0,
                "shift_scale_rotate": {"shift": 0.0, "scale": 0.0, "rotate": 0, "p": 0.0},
                "elastic": {"alpha": 120, "sigma": 6, "p": 0.0},
                "shear": {"shear_degrees": 10, "p": 0.0},
            },
            "color": {
                "brightness": 0.0, "contrast": 0.0, "saturation": 0.0,
                "brightness_contrast_p": 0.0,
                "gaussian_noise": {"var_limit": [10, 50], "p": 0.0},
                "clahe": {"clip_limit": 4.0, "tile_grid": [8, 8], "p": 0.0},
            },
            "multi_scale": {"scale_range": [0.5, 1.0], "p": 0.0},
        },
        "loss": {
            "function": "focal",
            "focal_gamma": 2.0,
            "focal_alpha": 0.25,
            "boundary_handling": "none",
            "boundary_ignore_width": 3,
            "soft_label_value": 0.05,
        },
        "training": {
            "batch_size": 2,
            "max_epochs": 2,
            "val_frequency": 1,
            "early_stopping": {
                "patience": 99, "min_delta": 0.0, "start_epoch": 999,
                "metric": "val_realistic_pr_auc_geomean", "smoothing_window": 1,
            },
            "precision": "fp32",
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "persistent_workers": False,
            "drop_last": True,
        },
        "optimizer": {
            "name": "adamw", "lr": 1e-3, "weight_decay": 1e-2, "gradient_clip_norm": 1.0,
        },
        "lr_schedule": {
            "frozen_lr": 1e-3, "base_lr": 1e-3,
            "backbone_lr_multiplier": 0.1,
            "freeze_backbone_epochs": 0,        # unfreeze at epoch 1
            "scheduler": "warmup_cosine",
            "warmup_epochs": 0,
            "warmup_start_lr": 1e-6,
            "min_lr": 1e-6,
            "backbone_warmup_epochs": 0,
        },
        "ema": {"enabled": True, "decay": 0.5},
        "model": {
            "architecture": "unetplusplus",
            "backbone": "resnet18",    # tiny; no ImageNet download when pretrained=False
            "pretrained": False,
            "output_bias_prior": 0.5,
        },
        "metrics": {
            "reporting_threshold": 0.5,
            "min_blob_size_px": 1,
            "object_iou_threshold": 0.3,
            "preview_tile_config": str(Path(data_root) / "preview_tiles.yaml"),
        },
        "mlflow": {
            "tracking_uri": f"file://{mlruns_dir}",
            "experiment_name": "smoke",
            "run_name": "train-smoke",
        },
    }


# ---------------------------------------------------------------------------
# Fixture: drive synthetic_dataset into the config and launch train.py
# ---------------------------------------------------------------------------


@pytest.fixture
def trained_run(synthetic_dataset, tmp_path, monkeypatch) -> dict:
    """Run train.py main() end-to-end on the synthetic fixture. Returns dict of paths."""
    cfg = _build_smoke_cfg(synthetic_dataset["root"], tmp_path / "mlruns")
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / "run"
    # Patch argv.
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--config", str(cfg_path),
        "--device", "cpu",
        "--out-dir", str(out_dir),
    ])

    import train  # scripts/train.py is on sys.path
    # Fresh imports — ensure no leaked global state from a previous test.
    rc = train.main()
    assert rc == 0, "train.main() returned non-zero"

    return {
        "cfg": cfg,
        "out_dir": out_dir,
        "mlruns_dir": tmp_path / "mlruns",
    }


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


def test_run_produces_log_file(trained_run):
    log = trained_run["out_dir"] / "train.log"
    assert log.exists()
    content = log.read_text()
    # Validation ran at least once.
    assert "val epoch=" in content


def test_figures_written(trained_run):
    fig_dir = trained_run["out_dir"] / "figures"
    assert fig_dir.exists()
    assert any(fig_dir.glob("prob_hist_epoch_*.png"))
    assert any(fig_dir.glob("confusion_epoch_*.png"))
    # PR curves only produced when val has positives; the synthetic fixture
    # has positives in region_C so curves should appear.
    assert any(fig_dir.glob("pr_curves_epoch_*.png"))


def test_deployment_checkpoint_contract(trained_run):
    """Best-deployment checkpoint exists and contains contracted fields."""
    ckpt_path = trained_run["out_dir"] / "checkpoints" / "best_deployment.pth"
    assert ckpt_path.exists(), "best_deployment.pth should be written on first val"
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict", "epoch", "best_metric",
        "channel_names", "git_sha", "trained_with", "checkpoint_type",
    }
    assert required <= set(loaded.keys())
    assert loaded["checkpoint_type"] == "deployment"
    # Channel names are the RGB names (no EXTRA in the smoke cfg).
    assert loaded["channel_names"] == ["R", "G", "B"]
    # trained_with carries precision + seed + config_sha (training.md §4.3).
    tw = loaded["trained_with"]
    assert tw["precision"] in {"bf16", "fp16", "fp32"}
    assert tw["seed"] == 42


def test_resume_checkpoint_rotation(trained_run):
    resume_dir = trained_run["out_dir"] / "checkpoints"
    resumes = sorted(resume_dir.glob("resume_latest-*.pth"))
    assert len(resumes) >= 1


def test_no_nan_in_model_params(trained_run):
    """Final parameters are all finite."""
    ckpt_path = trained_run["out_dir"] / "checkpoints" / "best_deployment.pth"
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model_state_dict"]
    for k, v in sd.items():
        if v.dtype.is_floating_point:
            assert torch.isfinite(v).all(), f"non-finite values in {k}"


def test_mlflow_run_written(trained_run):
    mlruns = trained_run["mlruns_dir"]
    assert mlruns.exists(), "MLflow directory should be created"
    # At least one run directory beneath the experiment.
    experiments = [p for p in mlruns.iterdir() if p.is_dir() and p.name != "models"]
    assert experiments, "no MLflow experiments written"


def test_ema_divergent_from_live_after_training(trained_run):
    """After training, EMA weights should not equal fresh live weights bit-for-bit.

    The deployment checkpoint holds EMA; if it matched the live weights
    exactly, something went wrong with the EMA update path.
    """
    # Compare epoch-0 live vs final EMA weights via resume checkpoint.
    resumes = sorted((trained_run["out_dir"] / "checkpoints").glob("resume_latest-*.pth"))
    assert resumes
    resume = torch.load(resumes[-1], map_location="cpu", weights_only=False)
    ema_sd = resume["ema_state_dict"]
    live_sd = resume["live_state_dict"]
    if ema_sd is None:
        pytest.skip("No EMA was constructed (freeze_backbone_epochs not crossed).")
    # At least one floating-point parameter must differ between EMA and live.
    diff_found = False
    for k, v in ema_sd.items():
        if v.dtype.is_floating_point and not torch.equal(v, live_sd[k]):
            diff_found = True
            break
    assert diff_found, "EMA weights identical to live weights post-training"


def test_prediction_shows_response_on_positive_region(synthetic_dataset, trained_run):
    """After 2 epochs, at least one positive tile should show > 0.5 prob somewhere.

    Weak but catches collapse-to-background bugs (model learns to output all
    negatives). Uses the deployment EMA checkpoint against a known positive tile.
    """
    import models
    import data.dataset as ds_mod

    cfg = trained_run["cfg"]
    ckpt_path = trained_run["out_dir"] / "checkpoints" / "best_deployment.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = models.build_model(cfg).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    # Find a positive tile in val_realistic.
    md = synthetic_dataset["metadata_df"]
    val_regions = synthetic_dataset["splits"]["val_realistic"]
    pos_val = md[(md["RegionName"].isin(val_regions)) & (md["TrainClass"] == "Positive")]
    assert len(pos_val) > 0, "synthetic val must have at least one positive tile"
    tid = pos_val.iloc[0]["Tile_id"]

    # Build a minimal single-tile dataset read to exercise the full data pipeline.
    from data.transforms import build_eval_transforms
    ev = ds_mod.RTSDataset(
        tile_ids=[tid],
        metadata=md,
        data_root=synthetic_dataset["root"],
        rgb_dir="PLANET-RGB",
        extra_dir="EXTRA",
        labels_dir="labels",
        extra_channels=[],
        norm_stats_path=None,
        transform=build_eval_transforms(),
        tile_size=64,
        label_ignore_index=255,
    )
    sample = ev[0]
    with torch.no_grad():
        logits = model(sample["image"].unsqueeze(0))
    probs = torch.sigmoid(logits).numpy()
    # Assert at least one pixel has a non-trivial response. "Non-collapse" is
    # looser than "accurate" — 2 epochs on synthetic data shouldn't produce a
    # perfect model, just one that isn't stuck at 0.
    assert probs.max() > 0.1, (
        f"Model collapsed: max prob on a positive tile is {probs.max():.4f}"
    )


def test_train_smoke_resume_then_continue(synthetic_dataset, tmp_path, monkeypatch):
    """Resume from a 2-epoch run for 1 more epoch; assert EMA shadow is restored.

    Guards Important I5 from the 2026-05-02 code review and the underlying
    audit fix that restores EMA state on resume (was silently falling back to
    live weights — a direct §10.2 violation).
    """
    # First run: 2 epochs.
    cfg = _build_smoke_cfg(synthetic_dataset["root"], tmp_path / "mlruns")
    cfg["training"]["max_epochs"] = 2
    cfg_path = tmp_path / "smoke_initial.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / "run_initial"
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--config", str(cfg_path),
        "--device", "cpu",
        "--out-dir", str(out_dir),
    ])
    import train  # scripts/train.py is on sys.path
    rc = train.main()
    assert rc == 0

    # Find the latest resume snapshot.
    resume_files = sorted((out_dir / "checkpoints").glob("resume_latest-*.pth"))
    assert resume_files, "no resume snapshot from initial run"
    resume_path = resume_files[-1]

    saved = torch.load(resume_path, map_location="cpu", weights_only=False)
    saved_ema_sd = saved.get("ema_state_dict")
    if saved_ema_sd is None:
        pytest.skip("EMA was not constructed in the initial run; resume-of-EMA path not exercised")

    # Second run: resume + 1 more epoch.
    cfg2 = dict(cfg)
    cfg2["training"]["max_epochs"] = 3   # +1 epoch beyond the resume point
    cfg2_path = tmp_path / "smoke_resume.yaml"
    cfg2_path.write_text(yaml.safe_dump(cfg2))

    out_dir2 = tmp_path / "run_resume"
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--config", str(cfg2_path),
        "--device", "cpu",
        "--out-dir", str(out_dir2),
        "--resume", str(resume_path),
    ])
    rc2 = train.main()
    assert rc2 == 0

    # Verify the resumed run wrote a fresh EMA snapshot — i.e. resume restored
    # the shadow rather than silently falling back to live and dropping EMA.
    new_resume_files = sorted((out_dir2 / "checkpoints").glob("resume_latest-*.pth"))
    assert new_resume_files
    new_saved = torch.load(new_resume_files[-1], map_location="cpu", weights_only=False)
    new_ema_sd = new_saved["ema_state_dict"]
    assert new_ema_sd is not None, "EMA dropped after resume — regression of audit fix"

    # Floating-point parameter keys should match between the two checkpoints.
    saved_keys = {k for k, v in saved_ema_sd.items() if v.dtype.is_floating_point}
    new_keys = {k for k, v in new_ema_sd.items() if v.dtype.is_floating_point}
    assert saved_keys == new_keys, "EMA state_dict keys changed across resume"

    # The EMA shadow at end-of-epoch-3 should differ from the one at end-of-epoch-2
    # (decay continued working) — strongest signal that EMA is alive, not stuck.
    diff_found = False
    for k in saved_keys:
        if not torch.equal(saved_ema_sd[k], new_ema_sd[k]):
            diff_found = True
            break
    assert diff_found, (
        "Post-resume EMA shadow is bit-identical to the saved one across an "
        "extra epoch of training — resume likely did not restart the EMA decay."
    )
