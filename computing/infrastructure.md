# Computing Infrastructure

**Single source of truth for compute facts**: GCP projects, storage buckets, the VM inventory,
regions, the compute budget, and the data storage map. This doc is *facts*; the step-by-step
how-tos live in [vm_instruction.md](vm_instruction.md) (daily VM/SSH workflow) and
[docker_training.md](docker_training.md) (build/run the training container).

> If you change an infra fact (a VM, a bucket, a path), update it **here** and reflect anything
> code reads from in the relevant `configs/*.yaml` (the config is the SSoT for values code loads;
> see [README](../README.md) §SSoT).

---

## 1. Overview

Two GCP projects are in play, and it is important not to confuse them:

- **PDG project** (`pdg-project-406720`) — where the **compute** lives: all VMs, the **$70k
  compute credit**, the Docker Artifact Registry, and the PDG bucket `rts-mapping-v2`. This is an
  **organization-managed** project (we do not control its IAM).
- **`abruptthawmapping`** (proj# 801926669176) — the user's own (non-PDG) project. Its bucket
  `abrupt_thaw` currently holds the v2.0 training data.

```
abruptthawmapping project (non-PDG)        pdg-project-406720 (PDG, org-managed)
┌────────────────────────────┐            ┌──────────────────────────────────────┐
│ gs://abrupt_thaw/          │  reads     │  VMs (L4 / A100 / planned multi-GPU)   │
│   RTS_MODEL_V2/DATA/...    │ ─────────► │  $70k compute credit (exp. Sep 2026)   │
│   (current training data)  │ (cross-    │  Artifact Registry (rts-train:v2)      │
└────────────────────────────┘  project   │  gs://rts-mapping-v2/  (recommended    │
                                 egress!)  │      home for compute-adjacent data)   │
                                           └──────────────────────────────────────┘
```

**Key implication:** the credit covers *compute only*. Storage and **network egress are billed
separately**, and reading `abrupt_thaw` (a non-PDG bucket) from PDG VMs can incur cross-project /
cross-region egress. The standing recommendation is to **co-locate compute-adjacent data in the
PDG bucket `rts-mapping-v2`, in the same region as the VMs** (see §4–§4b).

---

## 2. GCP projects

| Project | ID / number | Role | Notes |
|---------|-------------|------|-------|
| PDG | `pdg-project-406720` | VMs, $70k credit, Artifact Registry, bucket `rts-mapping-v2` | **Org-managed** — IAM not user-editable; IAP not authorized; Cloud Build SA cannot push to Artifact Registry. See §8. |
| abruptthawmapping | id `abruptthawmapping`, number `801926669176` | Holds bucket `abrupt_thaw` (current training data) | Non-PDG. `abruptthawmapping` is a **project**, not a bucket — `gs://abruptthawmapping/` does not exist. |

---

## 3. Cost & quota (read this first)

- **$70,000 GCP credit, compute-only, in the PDG project, expiring September 2026.** It must be
  **substantially spent** before expiry.
- **Compute-only caveat:** the credit does **not** cover storage or network egress. Watch
  Filestore, bucket storage, and cross-project/cross-region egress separately.
- **Spend plan** (from `current_working_status.md`):
  - Ablation program — **short bursts, ~$5–15k** on a multi-GPU node.
  - Bulk of budget — **pan-arctic inference + EXTRA-channel generation + multi-year/ensemble runs,
    ~$40–55k.**
- **Cost discipline:**
  - Always **stop VMs when idle** (idle costs below).
  - Consider **Spot/preemptible** GPUs for ablation sweeps (large savings; checkpoints make
    interruptions cheap). Keep production / final-seed runs on-demand.

| VM | On / idle cost (approx) |
|----|--------------------------|
| `a100-8x-train` (8× A100-80GB, a2-ultragpu-8g) | ~$30/hr running |
| `gpu-vm-l4` (L4) / `ml-training-vm` (single A100) | stopped/deprecated |
| Planned 8×H100 node | higher; pending quota |

> Prices are rough planning figures — confirm against current GCP pricing for the chosen
> machine type, region, and on-demand vs Spot.

---

## 4. Storage: buckets & on-VM tiers

### Buckets

| Bucket | Project | Purpose | Notes |
|--------|---------|---------|-------|
| `gs://abrupt_thaw/` | abruptthawmapping (non-PDG) | Current home of v2.0 training data under `RTS_MODEL_V2/DATA/` | Reading from PDG VMs crosses projects → egress. |
| `gs://rts-mapping-v2/` | PDG | **Recommended** home for all compute-adjacent data going forward (staged training data, outputs, artifacts, deployment packages, inference I/O) | Co-located with VMs → no cross-project egress. |

**Region co-location:** VMs run in **us-west1**. Keep buckets in **us-west1** (or US multi-region)
to avoid cross-region egress. *Verify the current region of both buckets* — this was not
confirmable during authoring (no bucket-list permission on the authoring credential).

### On-VM storage tiers

| Tier | Path | Durability | Use for |
|------|------|-----------|---------|
| Boot disk | `~` / local | **Treat as ephemeral** — do not rely on it across restarts | Repo checkout, venv, transient files |
| Filestore | `/mnt/argo_filestore` | Persistent, **1 TB, shared across VMs** | Large datasets, shared scratch |
| Host outputs | `/mnt/outputs` → mounted `/outputs` in Docker | Persists on the host VM | Checkpoints, logs, MLflow at `/mnt/outputs/mlflow` |
| gcsfuse mount | `/data` (inside Docker) | View onto GCS | Training data at run time; cache flags in `configs/baseline.yaml:training.gcsfuse` |

---

## 4b. Data storage map (current → recommended)

Where each kind of data lives today and where it should live. `rts-mapping-v2` prefixes below are
a **recommendation to confirm** — once authoritative, mirror any path code reads into the relevant
`configs/*.yaml` (SSoT for values).

| Data | What it is | Current location | Recommended location | Notes |
|------|-----------|------------------|----------------------|-------|
| **Training images (input)** | PLANET-RGB tiles, EXTRA tiles, labels | `gs://abrupt_thaw/RTS_MODEL_V2/DATA/TRAINING_DATA/{PLANET-RGB,EXTRA,labels}/` | Stage/co-locate in `gs://rts-mapping-v2/training/<version>/` in the VM region | Raw GeoTIFF, 512×512, EPSG:3857. Cuts cross-project egress; the credit is compute-only. |
| **Training metadata/config inputs** | `metadata.csv`, `splits.yaml`, `splits_summary.json`, `normalization_stats.json` | alongside training data in `.../TRAINING_DATA/` | travels with the training data | `normalization_stats.json` must travel with the model too (training.md §4). |
| **Region geometry** | `circumpolar_subregions.geojson` | `gs://abrupt_thaw/RTS_MODEL_V2/DATA/circumpolar_subregions.geojson` | with training inputs in `rts-mapping-v2` | Read by `scripts/create_splits.py`. |
| **Training outputs** | checkpoints, train/val logs | `/mnt/outputs/<run>/` on host VM | keep hot on `/mnt/outputs`, **sync finals** to `gs://rts-mapping-v2/runs/<run>/` | Host disk is not durable long-term; archive keepers to GCS. |
| **Experiment artifacts** | MLflow runs + artifacts, in-container `requirements_frozen`, preview visualizations, phase reports (e.g. `report.html`) | `/mnt/outputs/mlflow` (per-VM, local file store) | shared concurrency-safe backend + artifact bucket for the multi-node phase (see §7) | Per-VM store does not aggregate across parallel nodes. |
| **Deployment packages** | model weights + `normalization_stats.json` bundle (`scripts/package_model.py`) | examples currently point at stale `gs://abruptthawmapping/models/...` | `gs://rts-mapping-v2/models/<model-id>/` | Script default paths are example strings (see §10 backlog). |
| **Inference images (input)** | 2025 PlanetScope basemap, pan-arctic | **TBD** | a defined `gs://rts-mapping-v2/inference/inputs/` prefix, co-located with inference compute | Pan-arctic ≈ 7.5M tiles at default stride (inference.md §3.2). |
| **Inference outputs** | per-tile probability rasters, merged regional COGs, vectorized polygons | "GCS" (unspecified in inference.md) | `gs://rts-mapping-v2/inference/{probs,cogs,vectors}/` with a fixed prefix convention | Per-tile probs persist first, then a merge pass (inference.md §). |
| **Scratch / cache** | gcsfuse file cache, intermediate tiles | `/mnt/argo_filestore` or local scratch | same | Ephemeral; not a system of record. |

---

## 5. VM inventory

All VMs are in the **PDG project**. Daily start/stop/SSH workflow:
[vm_instruction.md](vm_instruction.md). Migration runbook/handover: [migrate_vm.md](migrate_vm.md).

### Existing

| VM | Zone | GPU | Role | Idle cost |
|----|------|-----|------|-----------|
| **`a100-8x-train`** | us-central1-a | **8× NVIDIA A100-80GB** (`a2-ultragpu-8g`) | **Production training node** (since 2026-06-12) — per-GPU parallel queues via `GPU=N scripts/run_ablation_queue.sh` | ~$30/hr running; stopping risks losing stockout-won capacity |
| `ml-training-vm` | us-west1-b | single NVIDIA A100-40GB | **STOPPED 2026-06-12** (superseded; boot disk kept) | — |
| `gpu-vm-l4` | us-west1-a | NVIDIA L4 (23 GB) | **Deprecated** | — |

### Planned

- **8×H100 upgrade (`a3-highgpu-8g`) when quota lands** — the 2×H100 (`a3-highgpu-2g`) spin-retry
  was stopped 2026-06-12 after the 8×A100 node landed.
  - Still required for multi-node scale-out: a parallel-experiment orchestrator and a
    concurrency-safe MLflow backend (see §7) — single-node per-GPU queues work today.
    `torch.compile` is on the roadmap.
- **Inference compute — open decision.** Pan-arctic inference is large (≈7.5M tiles). Options:
  1. **Reuse the multi-GPU node** for batch inference between ablation bursts (simplest; one VM to
     manage).
  2. **Dedicated inference VM(s)** sized for throughput, possibly Spot.
  3. **PDG distributed workflow** — hand tiling/orchestration to the PDG team (Luigi/Todd, §9);
     inference.md §10 already assumes a workflow handles VM orchestration + tile partitioning, one
     GPU per VM, no multi-GPU within a VM.
  *Recommendation:* default to (3) for the full pan-arctic run if PDG workflow capacity is
  available (it scales horizontally and matches the existing inference.md design), and use (1) for
  smaller regional / temporal-shift test runs. Decide before the inference phase.

### Zones & GPU availability

us-west1 is primary. GPU capacity varies by zone — if a start fails, fall back per
[vm_instruction.md Appendix A.1](vm_instruction.md) (us-west1-c, us-west2-a/b, us-central1-a).
Large multi-GPU and H100 machines need explicit quota and may only be available in specific zones.

---

## 6. Container & registry

- Image: `us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2`
  (base `nvcr.io/nvidia/pytorch:24.05-py3`, Python 3.10, CUDA 12.4).
- **Built locally on the L4 VM**, not via Cloud Build — the org project's Cloud Build SA lacks
  Artifact Registry push permission (§8). Auth via ADC (`docker login us-west1-docker.pkg.dev`).
- Build/run details: [docker_training.md](docker_training.md).

---

## 7. Experiment-tracking infra

- **Now:** MLflow tracking URI is `file:///outputs/mlflow` (SSoT:
  `configs/baseline.yaml:mlflow.tracking_uri`), persisted to `/mnt/outputs/mlflow` on the host.
  This is a **per-VM local file store** — it does not aggregate runs across machines, and the file
  backend is not safe for concurrent writers.
- **For the multi-GPU / parallel-orchestrator phase:** move to a **concurrency-safe shared
  backend** (e.g. an MLflow server backed by SQLite/Postgres on a shared path or small service)
  with artifacts in a GCS bucket, so parallel runs log into one tracking store. Tracked as a next
  step in `current_working_status.md`.
- MLflow 2.x does **not** support `gs://` as a *tracking* URI (only as an artifact store).

---

## 8. Access & permissions gotchas (org-managed PDG project)

The PDG project is organization-managed; we cannot edit its IAM. Consequences:

- **No IAM self-service** — cannot grant roles to service accounts or users.
- **Cloud Build cannot push to Artifact Registry** — build the Docker image locally on the VM
  (§6).
- **IAP tunneling is not authorized** — connect to VMs over their **external IP** (managed by
  `vmup.ps1`, see vm_instruction.md). `gcloud compute start-iap-tunnel` returns `4033 not
  authorized`.
- **GCS auth from VMs/containers** uses Application Default Credentials — run
  `gcloud auth application-default login` once, then mount
  `~/.config/gcloud/application_default_credentials.json` into the container
  (`GOOGLE_APPLICATION_CREDENTIALS`).
- **Authorized networks** — after a network change, your SSH source IP may need re-authorizing
  (vm_instruction.md Appendix A.2).
- Some credentials lack `compute.instances.list` / bucket `storage.objects.list` — use your own
  authenticated `gcloud`/`gsutil` for inventory operations.

---

## 9. Contacts

- **Luigi / Todd (PDG team)** — pan-arctic inference workflow and VM orchestration / tile
  partitioning (inference.md §10). Coordinate with them before the pan-arctic inference run.

---

## 10. Known discrepancies / cleanup backlog

`abruptthawmapping` is a **project**, not a bucket; the real data bucket is `abrupt_thaw`. Several
files still use `gs://abruptthawmapping/...` as if it were a bucket (or as example/default paths).
These are **flagged, not changed here** (out of scope for this doc; changing script defaults is a
separate, deliberate task):

- `inference/inference.md` §2.1 (Storage row, `gs://abruptthawmapping/`)
- `training/experiments.md` (mlflow `tracking_uri` example)
- `configs/baseline.yaml` (`data_root` comment example)
- `scripts/package_model.py`, `scripts/evaluate_test.py`, `scripts/inference_feasibility.py`
  (`gs://abruptthawmapping/models/...` example paths)
- `scripts/check_data_content.py` (`--bucket gs://abruptthawmapping/...` example/default)

When these are cleaned up, point them at `gs://rts-mapping-v2/...` (or `gs://abrupt_thaw/...` for
existing training data) per §4b.

---

## See also

- [vm_instruction.md](vm_instruction.md) — start/stop VMs, SSH config, Python env, file transfer.
- [docker_training.md](docker_training.md) — build the image, run training in the container, mounts.
- [current_working_status.md](../current_working_status.md) — live status, budget plan, next steps.
