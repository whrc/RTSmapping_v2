"""Checkpoint management.

Two distinct checkpoint types (training.md §4.3):
    - best_deployment.pth — EMA weights under `model_state_dict`,
      plus channel names, git SHA, epoch, best_metric, and a `trained_with`
      block. Inference loads this file. Channel-name binding (training.md §4.5)
      is the integrity guarantee for normalization stats.
    - resume_latest.pth  — full state (live + EMA weights, optimizer,
      scheduler state, scaler state, RNG states). Training continuation
      only; rotated last-N.

"Best" selection uses the smoothed metric from EarlyStopping, not the raw
value (plan risk #6 — checkpoint-stopping metric consistency). The smoothed
value is retrieved via `EarlyStopping.smoothed_value()` after `update()` is
called on the current epoch's metrics; the train loop passes it in.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _git_sha(repo_path: Path | None = None) -> str:
    """Return the current HEAD SHA, or 'unknown' if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path) if repo_path else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class TrainedWith:
    """Immutable metadata about how this model was trained.

    Persisted in the deployment checkpoint; inference / evaluate_test asserts
    `precision` and `torch_compile` match the deployment config (training.md §4.6).
    """
    precision: str
    seed: int
    config_sha: str  # SHA256 of the serialised config YAML


class CheckpointManager:
    """Save / load deployment + resume checkpoints.

    Args:
        out_dir: Directory to write checkpoints into.
        keep_last_n: How many resume_latest rotations to keep (oldest rotated off).
    """

    def __init__(self, out_dir: str | Path, keep_last_n: int = 3):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = int(keep_last_n)
        self._best_smoothed: float = float("-inf")

    # -------------------------------------------------------------- deployment

    def save_deployment(
        self,
        *,
        ema_state_dict: dict[str, torch.Tensor],
        epoch: int,
        best_metric: float,
        channel_names: list[str],
        trained_with: TrainedWith,
    ) -> Path:
        """Write best_deployment.pth atomically.

        Contains EMA weights under `model_state_dict` — inference never needs
        to know about EMA (training.md §4.3). Channel-name binding
        (training.md §4.5) is the integrity guarantee — both training and
        inference assert names against config; no separate file hash is kept.
        """
        payload = {
            "model_state_dict": ema_state_dict,
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "channel_names": list(channel_names),
            "git_sha": _git_sha(),
            "trained_with": asdict(trained_with),
            "checkpoint_type": "deployment",
        }
        target = self.out_dir / "best_deployment.pth"
        tmp = target.with_suffix(".pth.tmp")
        torch.save(payload, tmp)
        tmp.replace(target)
        logger.info("Saved deployment checkpoint to %s (epoch=%d, metric=%.5f)",
                    target, epoch, best_metric)
        return target

    # ------------------------------------------------------------------ resume

    def save_resume(
        self,
        *,
        model: nn.Module,
        ema_state_dict: Optional[dict[str, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        scheduler_state: Optional[dict],
        scaler_state: Optional[dict],
        epoch: int,
        early_stopping_state: dict,
        rng_states: dict,
    ) -> Path:
        """Write resume_latest-<epoch>.pth and rotate old snapshots."""
        payload = {
            "live_state_dict": model.state_dict(),
            "ema_state_dict": ema_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state": scheduler_state,
            "scaler_state": scaler_state,
            "epoch": int(epoch),
            "early_stopping_state": early_stopping_state,
            "rng_states": rng_states,
            "checkpoint_type": "resume",
        }
        target = self.out_dir / f"resume_latest-{epoch:04d}.pth"
        tmp = target.with_suffix(".pth.tmp")
        torch.save(payload, tmp)
        tmp.replace(target)
        self._rotate_resume()
        logger.info("Saved resume checkpoint to %s", target)
        return target

    def _rotate_resume(self) -> None:
        resumes = sorted(self.out_dir.glob("resume_latest-*.pth"))
        while len(resumes) > self.keep_last_n:
            victim = resumes.pop(0)
            victim.unlink()
            logger.debug("Rotated off resume checkpoint %s", victim)

    def latest_resume(self) -> Path | None:
        resumes = sorted(self.out_dir.glob("resume_latest-*.pth"))
        return resumes[-1] if resumes else None

    # ------------------------------------------------------------------ best tracking

    def update_best(self, smoothed_metric: float) -> bool:
        """Track best-so-far by the smoothed metric. Returns True iff improved."""
        if smoothed_metric > self._best_smoothed:
            self._best_smoothed = smoothed_metric
            return True
        return False

    @property
    def best_smoothed(self) -> float:
        return self._best_smoothed
