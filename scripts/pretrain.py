"""MAE continue-pretraining of DINOv3-L on the arctic corpus (spec: pretraining/pretraining.md §3).

Single DDP job across all 8 GPUs of a100-8x-train:
  torchrun --nproc_per_node=8 scripts/pretrain.py --config configs/v21/pretrain_dinov3l_arctic.yaml

Runs single-process too (no torchrun) for the 1-GPU / CPU smoke. Writes the encoder-only
state_dict to <out_dir>/encoder_final.pt (consumed by model.encoder_init at fine-tune) plus
periodic checkpoints, and a run_summary.json. MLflow logging on rank 0 only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pretraining.mim_dataset import MIMCorpusDataset  # noqa: E402
from pretraining.mim_model import MaskedAutoencoderViT  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.logging import setup_logging  # noqa: E402
from utils.seed import seed_everything  # noqa: E402

logger = logging.getLogger(__name__)


def _ddp_setup() -> tuple[int, int, int]:
    """Init DDP from torchrun env; return (rank, world_size, local_rank). (0,1,0) if unset."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local)
    return rank, world, local


def _lr_at(step: int, total: int, warmup: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay (per-step)."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--corpus-dir", type=str, default=None,
                   help="override cfg.pretrain.corpus_dir (e.g. the pilot corpus for a smoke)")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke)")
    args = p.parse_args()

    rank, world, local = _ddp_setup()
    is_main = rank == 0
    setup_logging()
    cfg = load_config(args.config)
    pt = cfg["pretrain"]
    seed_everything(cfg.get("seed", 42) + rank, deterministic=False)

    corpus_dir = args.corpus_dir or pt["corpus_dir"]
    out_dir = Path(args.out_dir or pt["out_dir"])
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    model = MaskedAutoencoderViT(
        backbone=pt["backbone"], pretrained=pt.get("pretrained", True),
        in_channels=4, patch_px=pt.get("patch_px", 16),
        grad_checkpointing=pt.get("grad_checkpointing", False),
    ).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local])

    ds = MIMCorpusDataset(corpus_dir, patch_px=pt.get("patch_px", 16),
                          mask_ratio=pt.get("mask_ratio", 0.75), seed=cfg.get("seed", 42))
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(ds, batch_size=pt["batch_size_per_gpu"], sampler=sampler,
                        shuffle=(sampler is None), num_workers=pt.get("num_workers", 8),
                        pin_memory=True, drop_last=True, persistent_workers=pt.get("num_workers", 8) > 0)

    global_bs = pt["batch_size_per_gpu"] * world
    base_lr = pt["base_lr_per_256"] * global_bs / 256
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            weight_decay=pt.get("weight_decay", 0.05),
                            betas=tuple(pt.get("betas", (0.9, 0.95))))
    epochs = pt["epochs"]
    steps_per_epoch = len(loader)
    total_steps = epochs * steps_per_epoch
    warmup = pt.get("warmup_epochs", 10) * steps_per_epoch
    min_lr = pt.get("min_lr", 1e-6)

    mlflow = None
    if is_main:
        import mlflow as _mlflow
        mlflow = _mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        mlflow.start_run(run_name=cfg["mlflow"]["run_name"])
        mlflow.log_params({"global_batch": global_bs, "base_lr": base_lr, "epochs": epochs,
                           "mask_ratio": pt.get("mask_ratio", 0.75), "backbone": pt["backbone"],
                           "corpus_dir": corpus_dir, "world_size": world, "n_tiles": len(ds)})
        logger.info("MAE pretrain: %d tiles, global_batch=%d, base_lr=%.2e, %d epochs, %d steps",
                    len(ds), global_bs, base_lr, epochs, total_steps)

    step = 0
    last_loss = float("nan")
    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            lr = _lr_at(step, total_steps, warmup, base_lr, min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            image = batch["image"].to(device, non_blocking=True)
            patch_mask = batch["patch_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(image, patch_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if pt.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), pt["grad_clip"])
            opt.step()
            last_loss = float(loss.detach())
            if is_main and step % pt.get("log_every", 50) == 0:
                logger.info("epoch %d step %d/%d loss %.4f lr %.2e",
                            epoch, step, total_steps, last_loss, lr)
                mlflow.log_metric("recon_loss", last_loss, step=step)
                mlflow.log_metric("lr", lr, step=step)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        if is_main and (epoch + 1) % pt.get("ckpt_every_epochs", 20) == 0:
            _save_encoder(model, out_dir / f"encoder_epoch{epoch + 1}.pt")

    if is_main:
        _save_encoder(model, out_dir / "encoder_final.pt")
        summary = {"run_name": cfg["mlflow"]["run_name"], "backbone": pt["backbone"],
                   "n_tiles": len(ds), "global_batch": global_bs, "epochs": epochs,
                   "steps": step, "final_recon_loss": last_loss,
                   "encoder_ckpt": str(out_dir / "encoder_final.pt")}
        (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(out_dir / "run_summary.json"))
        mlflow.end_run()
        logger.info("Done: %d steps, final recon loss %.4f → %s",
                    step, last_loss, out_dir / "encoder_final.pt")
    if world > 1:
        dist.destroy_process_group()
    return 0


def _save_encoder(model, path: Path) -> None:
    """Save the encoder-only state_dict (unwrap DDP) for model.encoder_init."""
    enc = (model.module if isinstance(model, DistributedDataParallel) else model).encoder
    torch.save({"encoder": enc.state_dict()}, path)
    logger.info("Saved encoder → %s", path)


if __name__ == "__main__":
    raise SystemExit(main())
