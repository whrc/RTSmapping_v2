# Computing Infrastructure

> ✅ **MIGRATION COMPLETE — 2026-09-01.** Everything now runs in `abruptthawmapping`; nothing of
> ours remains in `pdg-project-406720`. Access to PDG closed **2026-08-31**, a week earlier than the
> 09-07 funding cliff this doc was originally written against. **§4 (buckets) is current.** The
> narrative in §1–§3 and §5–§8 still describes the *PDG* arrangement and is kept as the record of
> why things are shaped the way they are — read it as history, not as instructions, and take live
> paths from §4. [pdg_migration.md](pdg_migration.md) owns the old→new path map (rule: the bucket
> changes, the prefix does not). The **machine inventory** lives in [README.md](README.md).

**Single source of truth for compute facts**: GCP projects, storage buckets, regions, quota,
and the data storage map. This doc is *facts*; the step-by-step
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
  `abrupt_thaw` currently holds the v2-alpha training data.

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

- **~$70,000 GCP credit in the PDG project, expiring September 2026** (must be substantially spent),
  **plus ~$100k borrowable from other teams (~$170k total) — budget is not the binding constraint**
  (per user, 2026-06-15). Size compute for *wallclock*, not cost.
- **Credit scope:** the ~$70k credit **covers everything** (compute, storage, egress) — per user; not a
  compute-only ceiling. So cross-region egress from the us-central1 master reading us-west1 quads is a
  non-issue cost-wise. (Superseded 2026-07-07: an earlier bullet argued for "run inference in us-west1
  to keep reads egress-free" — withdrawn. Inference runs on the us-central1 master reading cross-region;
  the throughput bottleneck was the **write path**, fixed in code, not the reads. See §4 region note.)
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

> For the **artifact → location** map (every durable artifact produced since project start → its
> bucket/path/region/owner-project, and which copy is source-of-truth vs backup), see
> [`artifact_inventory.md`](artifact_inventory.md). This section owns the bucket *facts*; that doc owns
> *where each artifact lives*.

### Buckets

| Bucket | Project | Purpose | Notes |
|--------|---------|---------|-------|
| `gs://abrupt_thaw/` | abruptthawmapping | Original/legacy data; v2-alpha training source under `RTS_MODEL_V2/DATA/` | US multi-region. |
| `gs://rts-arctic-us/` | abruptthawmapping | Compute-adjacent **training** data, run/MLflow mirrors, `label_sources/`, deployment packages, migration archives | **US (multi-region)**. *(was `gs://rts-mapping-v2/`)* |
| `gs://rts-arctic-usw1/` | abruptthawmapping | **Planet basemap quads** (`global_quarterly/<y>/q3/`), **Sentinel-2 imagery** (`S2_RGB/<y>_<region>/`), **primary inference I/O** (`inference/…`), `rescued_pdg/` | **us-west1 (single region)**. NDVI is computed on-the-fly from B8/B4 at read time. Banks products under `inference/banks/`. *(was `gs://rts-mapping-v2-usw1/` + `gs://pdg-planet-data/`, now merged into one bucket)* |
| `gs://rts-arctic-usc1/` | abruptthawmapping | `ee_mirror/`, `staging/` | **us-central1 (single region)**. *(was `gs://rts-mapping-v2-usc1/`)* |

Planet and S2 now share one us-west1 bucket, which removes the cross-project read that shaped the
2026-07 co-location debate below. Coldline lifecycle applies to `imagery/`, `global_quarterly/` and
`S2_RGB/`; `inference/` is deliberately excluded — 41.5 M ~2.5 KB objects would each bill a 128 KiB
minimum.

**Region / throughput — CORRECTED 2026-07-07 (supersedes the 2026-07-06 "move the data to us-central1"
plan).** The pan-Arctic inference bottleneck was **not** read locality — it was the **output write**. On
the A100 master the per-tile probability-COG write ran *synchronously* in the batch loop and opened a
**new `storage.Client` per tile**, stalling the GPU to ~2.8 tiles/s at 0% util. Fixed in code (commit
`d56e7ef`: async thread-pool writes in `inference/runner.py::_ProbWriter` + one cached GCS client in
`inference/writer.py`) → the real worker now sustains **~33 tiles/s per A100 reading in-region GCS, a
~12× gain with no data movement.** Controlled benchmark (2026-07-07): making *reads* local changed
nothing (2.8 → 3.0 t/s); making *writes* local gave 29–36 t/s — the write was ~90% of the cost.

Consequences:
- The earlier rationale — *"cross-region reads (448 ms/512×512 window) are the bottleneck, so anchor on
  the secured us-central1 GPU and move the 14 TB of quads to `gs://rts-mapping-v2-usc1`"* — is
  **withdrawn as the *primary* bottleneck**. With synchronous writes, the write dominated (2.8 t/s) and
  hid the read cost; fixing writes exposed reads as the **second** bottleneck (see the measured launch
  below). The 14 TB bulk transfer is still **not worth it** for a run already in flight.
- **Inference runs on the us-central1 master (`a100-8x-train`) reading directly from the us-west1
  buckets** (`pdg-planet-data` quads + `rts-mapping-v2-usw1` S2).
- **MEASURED AT THE ACTUAL SOUTH LAUNCH (2026-07-07, git_sha `7b7d74c`, 3-model ensemble, 8×A100):**
  at the default 8 DataLoader workers the run was **I/O-bound, not GPU- or CPU-bound** — ~12 t/s/A100,
  GPU util bursty 0↔100%, with ~61 idle vCPUs + ~780 GB free RAM. The 33 t/s figure was an **in-region**
  read rate; the production path reads **cross-region** (us-central1 ← us-west1), so once the write no
  longer stalls the GPU the cross-region read latency became the ceiling.
- **TUNED (applied):** `--num-workers 8→16` spent the idle CPU on more concurrent reads → **~24 t/s/A100
  (~217 t/s aggregate) → South ETA ≈ 2.3 days**, GPU util now dense 68–100% (near the in-region ceiling).
  16 is the launcher default (`scripts/launch_south_inference.sh`). Pushing higher isn't free — the
  per-worker §11.3 quad-cache fragments (a shard's spatially-sorted tiles split across more workers →
  fewer per-worker quad repeats → more GCS opens), so ~16 is the sweet spot, not "max".
- **Remaining lever (not needed):** an **in-region** us-west1 fleet/staging would recover the last bit
  toward 33 t/s, at stockout risk / 14 TB movement — not worth it at ~2.3 d. The GPU-scarcity fact still
  holds (the master took ~500 retries; never stop it).
- The write + fork-safety fixes are **baked into `rts-infer:v1`** (rebuilt+pushed 2026-07-07,
  `rts.git_sha=7b7d74c`). **Training** also runs on the same master (unavailable for v3 during the run).
  *(Historical: the us-west1-anchored and the 2026-07-06 us-central1 "move data" plans are retained in
  git + `docs/inference_launch_audit.md`.)*

### On-VM storage tiers

| Tier | Path | Durability | Use for |
|------|------|-----------|---------|
| Boot disk | `~` / local | **Treat as ephemeral** — do not rely on it across restarts | Repo checkout, venv, transient files |
| ~~Filestore~~ | ~~`/mnt/argo_filestore`~~ | **Not ours, and not mounted** — checked 2026-08-27: the 2 TB `argo-filestore` is a PDG instance, absent from the master and holding nothing of ours | — |
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
| **Inference images (input)** | 2025 PlanetScope basemap, pan-arctic | `gs://pdg-planet-data/` (**us-west1**, verified) | leave in place; run inference in us-west1 to read egress-free | Pan-arctic ≈ 7.5M tile-inferences at default stride (inference.md §3.2). |
| **Inference outputs + 2025 EXTRA** | per-tile probability COGs, merged regional COGs, vectors, 2025 EXTRA tiles | n/a (new) | `gs://woodwell-rts-inference-arts-south/` (**us-west1**, single region) — co-located with input + fleet | Egress-free; per-tile probs persist first, then merge/vectorize (post-inference §). |
| **Scratch / cache** | gcsfuse file cache, intermediate tiles | local scratch (`/mnt/nvme_scratch`) | same | Ephemeral; not a system of record. |

---

## 5. VM inventory

> **Moved.** The machine registry — codename, function, status, and which machines are *not*
> ours — now lives in [README.md](README.md) and is the SSoT for it. What remains here is
> zone/GPU-availability and quota context. Register new VMs there, not here.

All VMs listed below are in the **PDG project** and are retiring with it. Daily start/stop/SSH workflow:
[vm_instruction.md](vm_instruction.md). Migration runbook/handover: [migrate_vm.md](migrate_vm.md).

### Existing

| VM | Zone | GPU | Role | Idle cost |
|----|------|-----|------|-----------|
| **`a100-8x-train`** | us-central1-a | **8× NVIDIA A100-80GB** (`a2-ultragpu-8g`) | **Production training node** (since 2026-06-12) — per-GPU parallel queues via `GPU=N scripts/run_ablation_queue.sh` | ~$30/hr running; stopping risks losing stockout-won capacity |
| `ml-training-vm` | us-west1-b | single NVIDIA A100-40GB | **STOPPED 2026-06-12** (superseded; boot disk kept) | — |
| `gpu-vm-l4` | us-west1-a | NVIDIA L4 (23 GB) | **Deprecated** | — |

### Conventions (2026-06-23)

- **VM naming:** every project-owned VM is named with the **`rts-`** prefix (GCE names disallow
  underscores) so our instances are identifiable in the shared org project. **Not all VMs in the
  project are ours** — e.g. the pre-existing `download-vm` (n2-standard-4, us-west1-a) is **not ours;
  do not touch it**.
- **Control node (SUPERSEDED 2026-08-27):** the office PC `ARCHITECTURE` now drives VM
  lifecycle, reaching hosts over IAP — see [control_node.md](control_node.md). The convention
  below existed only because IAP was not authorized in the PDG project (§8); it does not
  carry over to `abruptthawmapping`.
- *(historical)* `a100-8x-train` (us-central1-a) drives the other VMs' lifecycle + jobs via
  `gcloud compute instances …` and `gcloud compute ssh rts-… --command "…"` over **external IP**
  (IAP is not authorized, §8) — so no VSCode Remote-SSH target switch is needed to operate them.

### Granted quota (us-west1, 2026-06-23)

For the inference fleet + S2 download VM: **`NVIDIA_L4_GPUS` 8→32** and **`CPUS` 246→480** (both
granted in full). `GPUS_ALL_REGIONS` stays 360 (≫ the ~40 needed; the +40→400 ask was denied but is
irrelevant). Preemptible-L4 not raised (still 8) → request `PREEMPTIBLE_NVIDIA_L4_GPUS=32` only if the
bulk inference pass is run on Spot.

**Live re-check (2026-06-26, Phase-3 pre-flight):** `NVIDIA_L4_GPUS` limit 32 **usage 1**, `CPUS` 480
usage 52 — confirmed. The **1 L4 in use is phantom** — stale GCP quota accounting from the deleted `ml-training-vm` (deleted
2026-06-24, confirmed 404; no instance/reservation/commitment holds an L4). Quota *usage* has no
`gcloud` release; it self-clears or needs a GCP support case (limit increases don't reset stale usage).
Net: only **31 schedulable**, so 4× `g2-standard-96` (needs 32) fails on the 4th VM until it clears.
**Default the fleet to 3× `g2-standard-96` (24 L4)** (unaffected); go to 4 once the phantom clears (or a
32→40 grant). The A100 master's 8 workers partly offset the missing 4th VM. Fleet scripts: `create_inference_fleet.sh` (sequential create,
fail-stop, on-demand) + `inference_fleet_startup.sh` (per-VM worker launch) + `inference_watchdog.sh`
(auto-stop the L4 fleet on completion; never the A100 master). Validate the startup on ONE VM before the rest.

### Planned

- **Inference fleet `rts-infer-usw1` — 4× `g2-standard-96` = 32× NVIDIA L4, us-west1** (decided
  2026-06-17, `inference.md §2.1`; **quota granted 2026-06-23**, creatable now). Pan-arctic inference
  (forward-only bf16, GCS-I/O-bound → no A100 needed); co-located with `pdg-planet-data` (us-west1).
  ~$15/hr on-demand (~$5/hr Spot). Spot for the bulk pass (resume via `inference_log.json`), on-demand
  for finals; **stop when idle** (L4 low-stockout). Outputs → `gs://woodwell-rts-inference-arts-south`.
  **Creation deferred to inference-time** (avoid idle GPU billing; a stopped VM does not hold L4 stock,
  so a reservation would be the only true capacity hedge — not worth the standing cost yet).
- **S2 download VM `rts-s2-download` — CPU-only (e.g. `c2d-standard-32`), Spot, us-west1** (doc
  `docs/s2_extra_data_prep.md §7`). GEE export is server-side → no GPU. Drives the Sentinel-2 RGB +
  NDVI-EXTRA download; stop-when-idle. Created fresh (the existing `download-vm` is not ours). Created
  at the bulk-run step, after the visual-QC gate.
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

The active production node is in **us-central1-a** (`a100-8x-train`); us-west1 hosted the now-stopped
single-A100 / L4 dev VMs. GPU capacity varies by zone — if a start fails, fall back per
[vm_instruction.md Appendix A.1](vm_instruction.md) (us-west1-c, us-west2-a/b, us-central1-a).
Large multi-GPU and H100 machines need explicit quota and may only be available in specific zones.
**Stopping `a100-8x-train` risks losing the stockout-won 8×A100 capacity** — prefer keeping it busy
over stopping for short idles (`experiments.md §13`).

---

## 6. Container & registry

- Image: `us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2`
  (base `nvcr.io/nvidia/pytorch:24.05-py3`, Python 3.10, CUDA 12.4).
- **Built locally on the L4 VM**, not via Cloud Build — the org project's Cloud Build SA lacks
  Artifact Registry push permission (§8). Auth via ADC (`docker login us-west1-docker.pkg.dev`).
- Build/run details: [docker_training.md](docker_training.md).
- **`earthengine-api` is NOT in the current `:v2` image** (it post-dates the build) — the 2024 EXTRA
  + tiling were done in Colab. The Sentinel-2 download (`scripts/export_s2_composites.py`,
  `generate_extra_tiles.py`, `qc_s2_preview.py`) needs it; for now `pip install earthengine-api` at
  container start (CPU-only, not latency-sensitive). **TODO:** rebuild the image with current
  `requirements.txt` (which already pins `earthengine-api>=1.0`) and bump the tag.

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
- **GCS + Earth Engine auth from VMs/containers** use Application Default Credentials — run
  `gcloud auth application-default login` once, then mount
  `~/.config/gcloud/application_default_credentials.json` into the container
  (`GOOGLE_APPLICATION_CREDENTIALS`). On `a100-8x-train` ADC was (re)created **2026-06-23** as
  `yyang@woodwellclimate.org` (the prior path was a broken empty *directory* — writes/EE silently
  failed; remove it before re-login). Same ADC powers `ee.Initialize(project='pdg-project-406720',
  opt_url='https://earthengine-highvolume.googleapis.com')`.
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
