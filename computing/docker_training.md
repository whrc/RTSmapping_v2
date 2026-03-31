# RTS Segmentation Model v2: Docker Training Setup

## Why Docker?

Docker packages the entire training environment (PyTorch, CUDA, GDAL, Python libraries) into a single image. This means:
- **Reproducibility**: The exact same environment runs on the L4 dev VM, the A100 production VM, and any future machine
- **No "works on my machine" bugs**: If it runs in the container, it runs everywhere
- **Clean VMs**: No need to install dependencies on each VM manually

## Workflow Overview

```
1. Develop and test code directly on L4 VM (no Docker, fast iteration)
2. When code is ready → build Docker image via Cloud Build
3. Pull image on production VM → run training
```

You do **not** develop inside Docker. You do **not** build Docker locally on Windows. Docker is only for packaging tested code into a reproducible environment for production runs.

---

## Part 1: Develop on L4 VM (No Docker)

This is where you spend most of your time. See `vm_instruction.md` for full setup.

Connect via VSCode Remote-SSH, edit code, and run directly:
```bash
source ~/ml-env/bin/activate
cd ~/RTSmappingDL
python scripts/check_data.py --config configs/baseline.yaml
python scripts/train.py --config configs/baseline.yaml  # short test run
```

**Key requirement**: Keep `~/ml-env` aligned with `requirements.txt` so that code that works on the VM will also work in the Docker container. Install new packages in both places — `pip install <package>` on the VM, and add it to `requirements.txt` for Docker.

---

## Part 2: Project Files for Docker

### Dockerfile.train

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies for geospatial + GCS mounting
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev gdal-bin fuse \
    && rm -rf /var/lib/apt/lists/*

# gcsfuse for GCS bucket mounting
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

# Mount points for data and outputs
RUN mkdir -p /data /outputs

ENTRYPOINT ["python", "-u"]
CMD ["scripts/train.py", "--config", "configs/baseline.yaml"]
```

### requirements.txt

```
segmentation-models-pytorch
albumentations
rasterio
geopandas
mlflow[gcs]
tqdm
pyyaml
pandas
scikit-learn
scipy
```

After a successful training run, freeze exact versions for reproducibility:
```bash
pip freeze > requirements_frozen.txt
```
This frozen file is saved as an MLflow artifact per run.

### .dockerignore

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

## Part 3: Build the Image

You build with **Cloud Build** (Google's remote build service), not on your VM. This avoids consuming VM disk and compute.

### One-time auth setup
```bash
gcloud config set project abruptthawmapping
gcloud auth configure-docker
```

### Build and push
From the repo root on the L4 VM:
```bash
gcloud builds submit --tag gcr.io/abruptthawmapping/rts-train:v2 . --timeout=1800
```
Takes ~10–15 minutes. The `--timeout=1800` gives 30 minutes for the build.

### Verify
```bash
gcloud container images list-tags gcr.io/abruptthawmapping/rts-train
```

---

## Part 4: Run on Production VM

### Pull the image
SSH into the production VM (see `vm_instruction.md` Part 7), then:
```bash
docker pull gcr.io/abruptthawmapping/rts-train:v2
```

### Test GPU access
```bash
docker run --rm --gpus all gcr.io/abruptthawmapping/rts-train:v2 \
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### Run training (single GPU)
```bash
docker run --rm --gpus '"device=0"' \
    --privileged \
    -v /mnt/outputs:/outputs \
    -e MLFLOW_TRACKING_URI="gs://abruptthawmapping/mlflow/" \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp_key.json \
    gcr.io/abruptthawmapping/rts-train:v2 \
    scripts/train.py --config configs/baseline.yaml
```

### Run training (multi-GPU, when DDP is implemented)
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
`--shm-size=32g` is required for DataLoader workers with multi-GPU.

### Run in background (detached)
```bash
docker run -d --gpus all \
    --privileged \
    --shm-size=32g \
    --name rts-training \
    -v /mnt/outputs:/outputs \
    -e MLFLOW_TRACKING_URI="gs://abruptthawmapping/mlflow/" \
    gcr.io/abruptthawmapping/rts-train:v2 \
    scripts/train.py --config configs/baseline.yaml

# Monitor
docker logs -f rts-training

# Stop
docker stop rts-training
```

---

## Part 5: Container Mounts and Environment

### Volume Mounts

| Container Path | Source | Mode | Purpose |
|----------------|--------|------|---------|
| `/data` | GCS via gcsfuse (mounted inside container) | read | Training data |
| `/outputs` | `/mnt/outputs` on host VM | read/write | Checkpoints, logs |

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `MLFLOW_TRACKING_URI` | `gs://abruptthawmapping/mlflow/` | MLflow GCS backend |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to service account JSON | GCS authentication |

### GCS Authentication

1. Create a service account with Storage Object Viewer + Storage Object Creator roles
2. Download the JSON key file
3. Mount the key file into the container and set the environment variable

---

## Iteration Cycle

```
1. Edit code on L4 VM via VSCode Remote-SSH
2. Test directly on L4 (no Docker rebuild needed)
3. When ready for production:
     gcloud builds submit --tag gcr.io/abruptthawmapping/rts-train:v2 .
4. On production VM:
     docker pull gcr.io/abruptthawmapping/rts-train:v2
5. Run training
```

**The key insight**: You only rebuild the Docker image when moving from development to production. During development, Docker is not in the loop.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Cloud Build timeout | Large image | Add `--timeout=1800` |
| `CUDA out of memory` | Batch too large | Reduce `batch_size` in config |
| `NCCL timeout` | Multi-GPU communication failure | Set `NCCL_DEBUG=INFO` to diagnose |
| `Permission denied` on /outputs | Volume ownership mismatch | Add `--user $(id -u):$(id -g)` |
| `rasterio` import error | GDAL missing | Verify Dockerfile has `libgdal-dev` |
| Image not found on VM | Auth issue | Run `gcloud auth configure-docker` |
| gcsfuse fails in container | Missing flag | Add `--privileged` to docker run |

## Checklist

### Before Cloud Build
- [ ] Code runs on L4 VM without errors
- [ ] All imports work
- [ ] `scripts/check_data.py` passes
- [ ] Training loop runs for a few steps
- [ ] `requirements.txt` is complete

### After Cloud Build
- [ ] Image appears in GCR (`gcloud container images list-tags`)
- [ ] GPU accessible in container
- [ ] Training runs and saves checkpoints
- [ ] MLflow logs appear in `gs://abruptthawmapping/mlflow/`