"""Unit tests for training.checkpoint.CheckpointManager."""

from __future__ import annotations

import torch
import torch.nn as nn

from training.checkpoint import CheckpointManager, TrainedWith


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))


def test_save_deployment_contains_contracted_fields(tmp_path):
    model = _tiny_model()
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
    mgr = CheckpointManager(tmp_path)

    mgr.save_deployment(
        ema_state_dict=ema_state,
        epoch=42,
        best_metric=0.75,
        channel_names=["R", "G", "B"],
        trained_with=TrainedWith(precision="bf16", seed=42, config_sha="abc123"),
    )
    target = tmp_path / "best_deployment.pth"
    assert target.exists()
    loaded = torch.load(target, weights_only=False)
    # Contracted keys per training.md §4.3.
    assert set(loaded.keys()) >= {
        "model_state_dict", "epoch", "best_metric",
        "channel_names", "git_sha", "trained_with", "checkpoint_type",
    }
    # Channel-name binding (training.md §4.5) is the integrity guarantee —
    # no separate file hash is kept on the checkpoint.
    assert "normalization_stats_hash" not in loaded
    assert loaded["checkpoint_type"] == "deployment"
    assert loaded["epoch"] == 42
    assert loaded["best_metric"] == 0.75
    assert loaded["channel_names"] == ["R", "G", "B"]
    assert loaded["trained_with"]["precision"] == "bf16"


def test_save_resume_contains_full_state(tmp_path):
    model = _tiny_model()
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path, keep_last_n=3)

    mgr.save_resume(
        model=model,
        ema_state_dict=ema_state,
        optimizer=optim,
        scheduler_state={"phase": 1},
        scaler_state={"scale": 65536.0},
        epoch=5,
        early_stopping_state={"best_epoch": 4},
        rng_states={"python": "..."},
    )
    target = tmp_path / "resume_latest-0005.pth"
    assert target.exists()
    loaded = torch.load(target, weights_only=False)
    assert loaded["checkpoint_type"] == "resume"
    assert loaded["epoch"] == 5
    assert "live_state_dict" in loaded
    assert "ema_state_dict" in loaded
    assert "optimizer_state_dict" in loaded
    assert loaded["scheduler_state"] == {"phase": 1}
    assert loaded["early_stopping_state"]["best_epoch"] == 4


def test_resume_rotation_keeps_last_n(tmp_path):
    model = _tiny_model()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr = CheckpointManager(tmp_path, keep_last_n=2)

    for ep in [1, 2, 3, 4]:
        mgr.save_resume(
            model=model,
            ema_state_dict=None,
            optimizer=optim,
            scheduler_state=None,
            scaler_state=None,
            epoch=ep,
            early_stopping_state={},
            rng_states={},
        )
    files = sorted(tmp_path.glob("resume_latest-*.pth"))
    assert len(files) == 2
    assert files[0].name == "resume_latest-0003.pth"
    assert files[1].name == "resume_latest-0004.pth"
    assert mgr.latest_resume().name == "resume_latest-0004.pth"


def test_update_best_tracks_smoothed_monotone():
    mgr = CheckpointManager(".", keep_last_n=1)  # no writes exercised
    assert mgr.update_best(0.1) is True
    assert mgr.update_best(0.2) is True
    assert mgr.update_best(0.15) is False
    assert mgr.best_smoothed == 0.2
