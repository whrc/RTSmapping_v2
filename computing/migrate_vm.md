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

## Dataset transition: v2.0 → v2.1 (2026-06-12, resolved as intentional)

The rewrite of `gs://abrupt_thaw/RTS_MODEL_V2/DATA/TRAINING_DATA/` on 2026-06-12 (~04:30–05:45
UTC) was initially logged here as a data-loss incident; **user confirmed it is the intentional
v2.1 drop**: the new `metadata.csv` holds **1,757 quality-filtered positives**
(`Version` ∈ batch1: 1169, batch2: 261, batch3: 327) — this is the canonical positive training
set going forward. Still in preparation upstream: **negative tiles** (~12 h ETA as of
2026-06-12) and the final **EXTRA** channel definition (inference EXTRA prepared after that
settles).

What this means:
- **v2.0 is gone and stays gone**: `metadata_phase0c.csv` contents, `splits_phase0c.yaml`,
  `normalization_stats.json`, `EXTRA/`, and all v2.0 tiles were overwritten with versioning
  Suspended. The old negatives are unrecoverable from our side; the frozen v2.0 tile inventory
  (15,528 rows incl. ~13.7k negative tile IDs + centroids) survives at
  `/mnt/outputs/phase0c_frozen_metadata.csv` and can be handed to the data team if regeneration
  of the same negative pool is wanted.
- **All v2.0-tied numbers are stale** (μ₀=0.5683, σ₀, gate G, every phase result). Once
  negatives land: freeze a v2.1 snapshot in `gs://rts-mapping-v2/training/v2.1/` (lesson from
  this incident — snapshots live in our own bucket), regenerate splits + normalization stats,
  re-run the Phase-0c 3-seed baseline, then re-test past decisions per
  `docs/v21_staleness_audit.md`.
- The 2026-06-11 04:10 UTC "transient" 404 crashes were the leading edge of this rewrite.

## Experiment backlog at handover (2026-06-12) — superseded by the v2.1 re-baseline

The pre-rewrite backlog (boundary_ignore_w2/w3, wd_5e2, aug_strong, compound_2to1_seed44
tiebreak) referenced the deleted v2.0 snapshot and **cannot be run as-is**. The successor plan
is the staleness audit + v2.1 re-baseline queue in `docs/v21_staleness_audit.md`; the §5.3
slope fit and §8.1 gate eval were completed from existing `/mnt/outputs` results (see
`docs/phase2_data_scaling.md`).

Results to date live in `current_working_status.md` (Status section) and `/mnt/outputs/*/run_summary.md`.

## Migration checklist

- [x] Queue script hardened (pipefail, crash-artifact rerun, `GPU=N`) + seed44 config committed
- [x] `/mnt/outputs` archived to `gs://rts-mapping-v2/runs/ml-training-vm-outputs/` (28.6 GB, size-verified)
- [x] a100-8x-train provisioned (docker + nvidia-container-toolkit installed & 8-GPU verified, ADC copied, `rts-train:v2` pulled)
- [x] `/mnt/outputs` transferred (tar-over-ssh; GCS restore blocked — new VM's gsutil runs as the compute SA which lacks `rts-mapping-v2` access, and `gcloud auth activate-refresh-token` rejects the ADC token (`unauthorized_client`); user `gcloud auth login` on the new VM would fix this)
- [x] Repo cloned (`phase1-prep` @ d6515b7), home scripts copied
- [x] Validation: environment good — 122 fast tests + cuda EMA test pass in-container on GPU 0. Backlog queues launched but **all 5 crashed on the deleted dataset** (see blocker above) — NOT an environment problem.
- [x] H100 spin-retry killed (2026-06-12 ~10:00 UTC)
- [x] Final outputs re-sync; `ml-training-vm` stopped
- [x] `infrastructure.md` VM inventory + `current_working_status.md` updated

(Claude: update these boxes as you complete steps; if a session dies mid-migration, resume from the
first unchecked box.)
