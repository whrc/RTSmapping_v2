# RTS Segmentation Model v2: Docker Training Setup

## 1. Overview

Containerize the training pipeline for reproducible execution on GCP VMs.

**Workflow**: Develop locally via VSCode Remote-SSH → Cloud Build (build image) → GCP VM (run training)

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  VSCode Remote  │      │  Cloud Build    │      │    GCP VM       │
│  (L4 dev VM)    │      │                 │      │  (A100/H100)    │
│                 │      │                 │      │                 │
│ - Develop code  │ ──── │ - Build image   │ ──── │ - Pull image    │
│ - Test on L4    │      │ - Push to GCR   │      │ - Run training  │
│ - No Colab      │      │ - No local env  │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

**Dev environment**: VSCode Remote-SSH connected to `gpu-vm-l4` (see `computing/vm_instruction.md`).
No Colab in this workflow.

---

## 2. Project Structure

```
RTSmappingDL/
├── Dockerfile.train
├── requirements.txt
├── .dockerignore
├── configs/
│   └── baseline.yaml
├── data/              ← data loading modules
├── models/            ← model definitions
├── losses/            ← loss functions
├── utils/             ← shared utilities
└── scripts/
    ├── train.py
    ├── inference.py
    ├── check_data.py
    └── create_splits.py
```

---

## 3. Dockerfile.train

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies for geospatial + gcsfuse
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    fuse \
    && rm -rf /var/lib/apt/lists/*

# Install gcsfuse for GCS bucket mounting
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-jammy main" \
    | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update && apt-get install -y gcsfuse \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY data/ /app/data/
COPY models/ /app/models/
COPY losses/ /app/losses/
COPY utils/ /app/utils/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Mount points
RUN mkdir -p /data /outputs

# Default command
ENTRYPOINT ["python", "-u"]
CMD ["scripts/train.py", "--config", "configs/baseline.yaml"]
```

---

## 4. requirements.txt

```
# Segmentation
segmentation-models-pytorch
albumentations

# Geospatial
rasterio
geopandas

# Experiment tracking
mlflow[gcs]          # includes GCS artifact store support

# Utilities
tqdm
pyyaml
pandas
scikit-learn
scipy
```

Use latest stable versions. Pin exact versions in `requirements_frozen.txt` after verifying compatibility (saved as MLflow artifact per run).

---

## 5. .dockerignore

```
.git/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
notebooks/
*.ipynb
docs/
```

---

## 6. Development Workflow (VSCode Remote-SSH on L4 VM)

### 6.1 Connect to Dev VM

```bash
# From local machine
gcloud compute instances start gpu-vm-l4 --zone=us-west1-a
gcloud compute ssh gpu-vm-l4 --zone=us-west1-a
```

Or use VSCode Remote-SSH extension (see `vm_instruction.md`).

### 6.2 Mount GCS Data

```bash
# In the VM (or inside container with --privileged)
mkdir -p /data
gcsfuse --implicit-dirs abruptthawmapping /data
```

### 6.3 Run Code Directly on VM (fast iteration, no Docker rebuild)

For rapid development cycles, run directly in the VM's Python environment:

```bash
source ~/ml-env/bin/activate
python scripts/check_data.py --config configs/baseline.yaml
python scripts/train.py --config configs/baseline.yaml
```

### 6.4 Run via Docker (mirrors production environment)

```bash
docker run --rm --gpus '"device=0"' \
    --privileged \           # required for gcsfuse inside container
    -v /mnt/outputs:/outputs \
    -e MLFLOW_TRACKING_URI="gs://abruptthawmapping/mlflow/" \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp_key.json \
    gcr.io/abruptthawmapping/rts-train:v2 \
    scripts/train.py --config configs/baseline.yaml
```

---

## 7. Build Image with Cloud Build

Once code works on the dev VM, build the production Docker image.

### 7.1 Authenticate and Set Project

```bash
gcloud config set project abruptthawmapping
gcloud auth configure-docker
```

### 7.2 Build and Push

```bash
# From repo root
gcloud builds submit --tag gcr.io/abruptthawmapping/rts-train:v2 . --timeout=1800
```

This takes ~10–15 minutes (base image pull + pip install).

### 7.3 Verify Image Exists

```bash
gcloud container images list-tags gcr.io/abruptthawmapping/rts-train
```

---

## 8. Run on Production VM (A100/H100)

### 8.1 SSH to Production VM

```bash
gcloud compute ssh ml-training-vm --zone=us-west1-b
```

### 8.2 Pull Image

```bash
docker pull gcr.io/abruptthawmapping/rts-train:v2
```

### 8.3 Test GPU Access

```bash
docker run --rm --gpus all gcr.io/abruptthawmapping/rts-train:v2 \
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 8.4 Run Training (Single GPU)

```bash
docker run --rm --gpus '"device=0"' \
    --privileged \
    -v /mnt/outputs:/outputs \
    -e MLFLOW_TRACKING_URI="gs://abruptthawmapping/mlflow/" \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp_key.json \
    gcr.io/abruptthawmapping/rts-train:v2 \
    scripts/train.py --config configs/baseline.yaml
```

### 8.5 Run Training (Multi-GPU — DDP, when implemented)

```bash
docker run --rm --gpus all \
    --shm-size=32g \
    --privileged \
    -v /mnt/outputs:/outputs \
    -e MLFLOW_TRACKING_URI="gs://abruptthawmapping/mlflow/" \
    gcr.io/abruptthawmapping/rts-train:v2 \
    -m torch.distributed.run \
    --nproc_per_node=8 \
    scripts/train.py --config configs/baseline.yaml
```

Note: `--shm-size=32g` required for DataLoader workers with multi-GPU.

### 8.6 Run in Background (Detached)

```bash
docker run -d --gpus all \
    --privileged \
    --shm-size=32g \
    --name rts-training \
    -v /mnt/outputs:/outputs \
    -e MLFLOW_TRACKING_URI="gs://abruptthawmapping/mlflow/" \
    gcr.io/abruptthawmapping/rts-train:v2 \
    scripts/train.py --config configs/baseline.yaml

# Monitor logs
docker logs -f rts-training

# Stop
docker stop rts-training
```

---

## 9. Volume Mounts and Environment Variables

| Container Path | Host/Config | Mode | Purpose |
|----------------|-------------|------|---------|
| `/data` | GCS via gcsfuse (inside container) | ro | Training data |
| `/outputs` | `/mnt/outputs` on VM | rw | Checkpoints, logs |

| Variable | Value | Purpose |
|----------|-------|---------|
| `MLFLOW_TRACKING_URI` | `gs://abruptthawmapping/mlflow/` | MLflow GCS backend |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON | GCS authentication |
| `CUDA_VISIBLE_DEVICES` | all (default) | GPU selection |
| `NCCL_DEBUG` | `INFO` | Multi-GPU debugging (set when needed) |

---

## 10. Iteration Workflow

```
1. Edit code via VSCode Remote-SSH on L4 VM
2. Run directly on L4 VM to test (fast feedback, no Docker rebuild)
3. When ready for production run:
   gcloud builds submit --tag gcr.io/abruptthawmapping/rts-train:v2 .
4. On production VM, pull new image:
   docker pull gcr.io/abruptthawmapping/rts-train:v2
5. Run training
```

---

## 11. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Cloud Build timeout | Large image | Add `--timeout=1800` (30 min) |
| `CUDA out of memory` | Batch too large | Reduce `batch_size` in config |
| `NCCL timeout` | Multi-GPU comm failure | Set `NCCL_DEBUG=INFO` |
| `Permission denied` on /outputs | Volume ownership | `--user $(id -u):$(id -g)` |
| `rasterio` import error | GDAL missing | Verify Dockerfile has `libgdal-dev` |
| Image not found on VM | Auth issue | Run `gcloud auth configure-docker` |
| gcsfuse fails in container | Missing --privileged | Add `--privileged` flag |

---

## 12. Checklist

### Before Cloud Build
- [ ] Code runs on dev VM without errors
- [ ] All imports work
- [ ] `scripts/check_data.py` passes
- [ ] Training loop runs for a few steps on L4 VM
- [ ] requirements.txt complete

### After Cloud Build
- [ ] Image appears in GCR
- [ ] Can pull image on production VM
- [ ] GPU accessible in container
- [ ] Training runs and saves checkpoints to GCS outputs
- [ ] MLflow logs appear in `gs://abruptthawmapping/mlflow/`

---

## 13. Next Steps

After training works end-to-end:
1. Run full training experiment (baseline.yaml)
2. Verify checkpoint saving/loading
3. Confirm MLflow tracking on GCS
4. Build inference container (separate Dockerfile.inference when inference spec is ready)
