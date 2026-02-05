# RTS Segmentation Model v2: Docker Training Setup

## 1. Overview

Containerize the training pipeline for reproducible execution on GCP H100 GPUs.

**Workflow**: Colab (develop/test) → Cloud Build (build image) → PDG VM (run training)

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Google Colab   │      │  Cloud Build    │      │    PDG VM       │
│                 │      │                 │      │                 │
│ - Develop code  │ ──── │ - Build image   │ ──── │ - Pull image    │
│ - Test on T4/A100│      │ - Push to GCR   │      │ - Run on H100s  │
│ - No Docker     │      │ - No local env  │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

---

## 2. Project Structure

Maintain this structure in your repo/GCS:

```
rts-segmentation-v2/
├── Dockerfile.train
├── requirements.txt
├── .dockerignore
├── configs/
│   └── baseline.yaml
├── src/
│   ├── data/
│   ├── models/
│   ├── losses/
│   └── ...
└── scripts/
    └── train.py
```

---

## 3. Dockerfile.train

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies for geospatial
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/

# Create mount points
RUN mkdir -p /data /outputs

# Default command
ENTRYPOINT ["python", "-u"]
CMD ["scripts/train.py", "--config", "configs/baseline.yaml"]
```

---

## 4. requirements.txt

```
# Segmentation
segmentation-models-pytorch==0.3.4
albumentations==1.4.11

# Geospatial
rasterio==1.3.10
geopandas==1.0.1

# Experiment tracking
mlflow==2.15.0

# Utilities
tqdm==4.66.5
pyyaml==6.0.2
pandas==2.2.2
scikit-learn==1.5.1
```

---

## 5. .dockerignore

```
.git/
__pycache__/
*.pyc
data/
outputs/
mlruns/
*.egg-info/
.pytest_cache/
notebooks/
*.ipynb
```

---

## 6. Phase 1: Develop and Test in Colab

### 6.1 Setup Colab Environment

```python
# Mount Google Drive (for code) or clone from repo
from google.colab import drive
drive.mount('/content/drive')

# Or clone repo
!git clone https://github.com/your-org/rts-segmentation-v2.git
%cd rts-segmentation-v2

# Install same dependencies as Docker (test compatibility)
!pip install segmentation-models-pytorch==0.3.4 \
             albumentations==1.4.11 \
             rasterio==1.3.10 \
             geopandas==1.0.1 \
             mlflow==2.15.0
```

### 6.2 Mount GCS Data

```python
# Authenticate
from google.colab import auth
auth.authenticate_user()

# Mount GCS bucket via gcsfuse
!echo "deb https://packages.cloud.google.com/apt gcsfuse-jammy main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
!curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
!sudo apt-get update && sudo apt-get install -y gcsfuse

!mkdir -p /content/data
!gcsfuse abruptthawmapping /content/data
```

### 6.3 Test Training Code

```python
# Test data loading
from src.data.dataset import RTSDataset
dataset = RTSDataset('/content/data/PLANET-RGB', '/content/data/labels')
print(f"Dataset size: {len(dataset)}")

# Test model
import segmentation_models_pytorch as smp
model = smp.UnetPlusPlus(encoder_name='efficientnet-b7', in_channels=3, classes=1)
print("Model created")

# Test forward pass
import torch
x = torch.randn(2, 3, 512, 512).cuda()
model = model.cuda()
out = model(x)
print(f"Output shape: {out.shape}")

# Test a few training steps
# ... your training loop ...
```

### 6.4 Iterate Until Code Works

- Fix bugs in Colab (fast feedback loop)
- Test data augmentation pipeline
- Verify loss computation
- Run a few epochs to confirm training progresses

---

## 7. Phase 2: Build Image with Cloud Build

Once code works in Colab, build the Docker image.

### 7.1 Authenticate and Set Project

```python
from google.colab import auth
auth.authenticate_user()

!gcloud config set project abruptthawmapping
```

### 7.2 Build and Push with Cloud Build

```python
# Navigate to project directory
%cd /content/rts-segmentation-v2

# Build and push in one command
# Cloud Build will:
#   1. Upload your code to Cloud Storage
#   2. Build the Docker image remotely
#   3. Push to Google Container Registry
!gcloud builds submit --tag gcr.io/abruptthawmapping/rts-train:v2 .
```

This takes ~10-15 minutes (mostly pulling the base image and pip install).

### 7.3 Verify Image Exists

```python
!gcloud container images list --repository=gcr.io/abruptthawmapping
!gcloud container images list-tags gcr.io/abruptthawmapping/rts-train
```

---

## 8. Phase 3: Run on PDG VM

### 8.1 SSH to VM

```bash
gcloud compute ssh rts-vm --zone=us-central1-a
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
    -v /mnt/data:/data:ro \
    -v /mnt/outputs:/outputs \
    gcr.io/abruptthawmapping/rts-train:v2 \
    scripts/train.py --config configs/baseline.yaml
```

### 8.5 Run Training (Multi-GPU with DDP)

```bash
docker run --rm --gpus all \
    --shm-size=32g \
    -v /mnt/data:/data:ro \
    -v /mnt/outputs:/outputs \
    gcr.io/abruptthawmapping/rts-train:v2 \
    -m torch.distributed.run \
    --nproc_per_node=8 \
    scripts/train.py --config configs/baseline.yaml
```

**Note**: `--shm-size=32g` is required for DataLoader workers with multi-GPU.

### 8.6 Run in Background (Detached)

```bash
docker run -d --gpus all \
    --shm-size=32g \
    --name rts-training \
    -v /mnt/data:/data:ro \
    -v /mnt/outputs:/outputs \
    gcr.io/abruptthawmapping/rts-train:v2 \
    -m torch.distributed.run \
    --nproc_per_node=8 \
    scripts/train.py --config configs/baseline.yaml

# Check logs
docker logs -f rts-training

# Stop if needed
docker stop rts-training
```

---

## 9. Volume Mounts

| Container Path | Host Path | Mode | Purpose |
|----------------|-----------|------|---------|
| `/data` | `/mnt/data` or GCS | ro | Training data |
| `/outputs` | `/mnt/outputs` | rw | Checkpoints, logs |

---

## 10. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | all | GPU selection |
| `MLFLOW_TRACKING_URI` | `/outputs/mlruns` | MLflow storage |
| `NCCL_DEBUG` | - | Set to `INFO` for multi-GPU debugging |

---

## 11. Iteration Workflow

When you need to change code:

```
1. Edit code in Colab
2. Test changes in Colab
3. Rebuild image:
   !gcloud builds submit --tag gcr.io/abruptthawmapping/rts-train:v2 .
4. On PDG VM, pull new image:
   docker pull gcr.io/abruptthawmapping/rts-train:v2
5. Run training
```

For quick iterations, you can mount code as a volume (dev mode):

```bash
# On VM - mount local code instead of baked-in code
docker run --rm --gpus all \
    -v /home/user/rts-segmentation-v2/src:/app/src:ro \
    -v /home/user/rts-segmentation-v2/scripts:/app/scripts:ro \
    -v /mnt/data:/data:ro \
    -v /mnt/outputs:/outputs \
    gcr.io/abruptthawmapping/rts-train:v2 \
    scripts/train.py --config configs/baseline.yaml
```

---

## 12. Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Cloud Build timeout | Large image | Add `--timeout=1800` (30 min) |
| `CUDA out of memory` | Batch too large | Reduce `batch_size` in config |
| `NCCL timeout` | Multi-GPU comm failure | Set `NCCL_DEBUG=INFO` |
| `Permission denied` on /outputs | Volume ownership | `--user $(id -u):$(id -g)` |
| `rasterio` import error | GDAL missing | Verify Dockerfile has `libgdal-dev` |
| Image not found on VM | Auth issue | Run `gcloud auth configure-docker` |

---

## 13. Checklist

### Before Cloud Build
- [ ] Code runs in Colab without errors
- [ ] All imports work
- [ ] Training loop runs for a few steps
- [ ] requirements.txt has all dependencies

### After Cloud Build
- [ ] Image appears in GCR
- [ ] Can pull image on VM
- [ ] GPU accessible in container
- [ ] Training runs and saves checkpoints

---

## 14. Next Steps

After training works end-to-end:
1. Run full training experiment
2. Verify checkpoint saving/loading
3. Confirm MLflow tracking
4. Build inference container (docker_inference.md)