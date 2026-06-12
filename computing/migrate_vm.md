# VM Migration Handover: ml-training-vm → a100-8x-train

> Living doc. If you are Claude Code starting a session on `a100-8x-train`, read this first —
> it tells you what was migrated, where everything lives, and how to resume work.
> Canonical infra facts stay in [infrastructure.md](infrastructure.md); experiment program SSoT is
> `training/experiments.md`; project diary is `current_working_status.md`.

## What / why

- **2026-06-12**: production training moved from `ml-training-vm` (1× A100-40GB, us-west1-b)
  to **`a100-8x-train` (a2-ultragpu-8g: 8× A100-80GB, us-central1-a)**, created 03:22 UTC after a
  336-attempt spin-retry under the approved 8-GPU `NVIDIA_A100_80GB_GPUS` us-central1 quota.
- The H100 (a3-highgpu-2g) spin-retry was **stopped** the same day — the 8× A100 node covers the
  ablation program. `ml-training-vm` is **STOPPED (not deleted)**; its boot disk still holds the
  original `/mnt/outputs` (also archived to GCS, see below).
- The old L4 dev VM (`gpu-vm-l4`) was already deprecated.

## Machine map (a100-8x-train)

| Thing | Location | Notes |
|---|---|---|
| Repo | `~/RTSmappingDL`, branch `phase1-prep` | clone of github.com/whrc/RTSmappingDL |
| Outputs / checkpoints / MLflow | `/mnt/outputs` (on the 500 GB pd-ssd boot disk) | migrated copy of the old VM's outputs; MLflow file store at `/mnt/outputs/mlflow` (URI `file:///outputs/mlflow` inside Docker) |
| GCS archive of outputs | `gs://rts-mapping-v2/runs/ml-training-vm-outputs/` | snapshot taken at migration; also the durable archive for finished runs |
| Docker image | `us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2` | auth: `docker login us-west1-docker.pkg.dev -u oauth2accesstoken` with `gcloud auth print-access-token` |
| User ADC | `~/.config/gcloud/application_default_credentials.json` | copied from old VM; mounted into containers as `/gcp_adc.json` |
| GPUs | 8× A100-80GB, indices 0–7 | one experiment per GPU via `GPU=N` (see runbook) |
| Local SSDs | 8× NVMe, **unformatted/unused** | nothing in the pipeline needs local scratch; set up RAID0 only if needed |
| Home scripts | `~/create_{h100,a100}_vm.sh` + logs | historical; spin-retries are no longer running |

Docker needs `sudo` on this box (same as the old VM).

## Runbook: launching experiments

One sequential queue per GPU (the queue script is single-GPU; parallelism = several queues):

```bash
cd ~/RTSmappingDL
GPU=0 nohup bash scripts/run_ablation_queue.sh <configA> >> /mnt/outputs/queue_gpu0.log 2>&1 &
GPU=1 nohup bash scripts/run_ablation_queue.sh <configB> >> /mnt/outputs/queue_gpu1.log 2>&1 &
```

- Args are config basenames (`configs/<name>.yaml`); container + out-dir + log are named `<name>`.
- The script skips configs whose `run_summary.md` shows a real result; a summary with
  `best_epoch | -1` is a crash artifact and is rerun.
- Don't put two configs with the same name on two GPUs (container name collision).
- Win gate: Δ(`val_realistic_pr_auc_geomean`) ≥ G=0.025 over μ₀=0.5683 + no precision regression
  (`training/experiments.md §1.4`).

## Experiment backlog at handover (2026-06-12)

| Run | Status |
|---|---|
| `phase3_boundary_ignore_w2` | ☐ rerun (crashed 2026-06-12 on transient GCS 404s) |
| `phase3_boundary_ignore_w3` | ☐ rerun (same) |
| `phase3_wd_5e2` | ☐ rerun (same) |
| `phase3_aug_strong` | ☐ rerun (same) |
| `phase3_loss_compound_2to1_seed44` | ☐ run — tiebreak: compound 2:1 is borderline (seed42=0.6035, seed43=0.5760; 2-seed mean Δ=+0.021 < G=0.025) |
| After those | §5.3 data-scaling slope fit · §8.1 Phase-5 gate eval · §6.4 Phase-3 lock |

Results to date live in `current_working_status.md` (Status section) and `/mnt/outputs/*/run_summary.md`.

## Migration checklist

- [x] Queue script hardened (pipefail, crash-artifact rerun, `GPU=N`) + seed44 config committed
- [ ] `/mnt/outputs` archived to `gs://rts-mapping-v2/runs/ml-training-vm-outputs/`
- [ ] a100-8x-train provisioned (docker + toolkit verified, ADC copied, `rts-train:v2` pulled)
- [ ] `/mnt/outputs` restored from GCS on a100-8x-train
- [ ] Repo cloned (`phase1-prep`), home scripts copied
- [ ] Validation: backlog queues launched, each run past epoch 1, MLflow writing
- [ ] H100 spin-retry killed
- [ ] Final outputs re-sync, `ml-training-vm` stopped
- [ ] `infrastructure.md` VM inventory + `current_working_status.md` updated

(Claude: update these boxes as you complete steps; if a session dies mid-migration, resume from the
first unchecked box.)
