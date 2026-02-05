# GCP VM Guide for ML Training (Windows)

Instructions: https://docs.google.com/document/d/1BFwFRtXIYNjjQ7ovyEp6O1v31oTO8dSn8IDPotUBxhM/edit?tab=t.0#heading=h.lesqdidddz9v


## Quick Reference

| VM Name | Zone | GPU | Use Case |
|---------|------|-----|----------|
| gpu-vm-l4 | us-west1-a | NVIDIA L4 | Testing, lighter workloads |
| ml-training-vm | us-west1-b | NVIDIA A100 | Production training |

Persistent storage: `/mnt/argo_filestore` (1TB shared)

---

## Part 1: Connecting to the VM

### Step 1: Open Google Cloud SDK Shell
Launch "Google Cloud SDK Shell" from Windows Start Menu.

### Step 2: Set the correct project
```
gcloud config set project pdg-project-406720
```

### Step 3: Authorize your IP (run once per session/IP change)
First get your IP:
```
curl ifconfig.me
```

Then authorize it (replace YOUR_IP with the output):
```
gcloud container clusters update autopilot-cluster-1 --region us-west1 --enable-master-authorized-networks --master-authorized-networks YOUR_IP/32
```

### Step 4: Check VM status
```
gcloud compute instances list
```

### Step 5: Start the VM
For testing (L4 GPU):
```
gcloud compute instances start gpu-vm-l4 --zone=us-west1-a
```

For production (A100 GPU):
```
gcloud compute instances start ml-training-vm --zone=us-west1-b
```

### Step 6: SSH into the VM
```
gcloud compute ssh gpu-vm-l4 --zone=us-west1-a
```
or
```
gcloud compute ssh ml-training-vm --zone=us-west1-b
```

### Step 7: Verify GPU
Once inside the VM:
```
nvidia-smi
```

---

## Part 2: Environment Setup (First Time Only)

### Create Python virtual environment
```bash
python3 -m venv ~/ml-env
source ~/ml-env/bin/activate
pip install --upgrade pip
```

### Install PyTorch with CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Install common ML packages
```bash
pip install numpy pandas matplotlib scikit-learn tqdm wandb tensorboard
pip install rasterio geopandas shapely  # for geospatial work
pip install segmentation-models-pytorch  # for semantic segmentation
pip install transformers accelerate  # for foundation models
```

### Verify CUDA is working
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

---

## Part 3: Transferring Files

### Option A: Use persistent Filestore (recommended for large datasets)
Data persists across VM restarts:
```bash
ls /mnt/argo_filestore/
# Create your own directory
mkdir -p /mnt/argo_filestore/yili
```

### Option B: Upload from local machine
From your LOCAL Cloud SDK Shell (not the VM):
```
gcloud compute scp "C:\path\to\local\file.py" gpu-vm-l4:~/file.py --zone=us-west1-a
```

Upload a folder:
```
gcloud compute scp --recurse "C:\path\to\folder" gpu-vm-l4:~/folder --zone=us-west1-a
```

### Option C: Clone from GitHub
Inside the VM:
```bash
git clone https://github.com/your-repo/your-project.git
```

### Option D: Download from Google Cloud Storage
```bash
gsutil cp gs://bucket-name/file.tif ~/data/
gsutil -m cp -r gs://bucket-name/folder ~/data/  # recursive, parallel
```

---

## Part 4: Running Training

### Basic training workflow
```bash
# Activate environment
source ~/ml-env/bin/activate

# Navigate to project
cd ~/your-project

# Run training
python train.py --config config.yaml
```

### Run training in background (survives SSH disconnect)
Using nohup:
```bash
nohup python train.py > training.log 2>&1 &
```

Using screen (recommended):
```bash
screen -S training
source ~/ml-env/bin/activate
python train.py
# Press Ctrl+A then D to detach
# Reconnect later with: screen -r training
```

Using tmux:
```bash
tmux new -s training
source ~/ml-env/bin/activate
python train.py
# Press Ctrl+B then D to detach
# Reconnect later with: tmux attach -t training
```

### Monitor GPU usage
```bash
# One-time check
nvidia-smi

# Continuous monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Or use nvitop (install first: pip install nvitop)
nvitop
```

### Monitor training logs
```bash
tail -f training.log
```

---

## Part 5: Using Jupyter Notebook

### On the VM, start Jupyter:
```bash
source ~/ml-env/bin/activate
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

### On your LOCAL machine, set up port forwarding:
Open a NEW Cloud SDK Shell window and run:
```
gcloud compute ssh gpu-vm-l4 --zone=us-west1-a -- -L 8888:localhost:8888 -N -f
```

### Access Jupyter:
Open browser to: http://localhost:8888

---

## Part 6: Shutting Down

### Exit the VM
```bash
exit
```

### Stop the VM (important - prevents charges!)
From local Cloud SDK Shell:
```
gcloud compute instances stop gpu-vm-l4 --zone=us-west1-a
```
or
```
gcloud compute instances stop ml-training-vm --zone=us-west1-b
```

### Verify it stopped
```
gcloud compute instances list
```
Status should show "TERMINATED".

---

## Troubleshooting

### SSH connection fails
- Check if VM is running: `gcloud compute instances list`
- Re-authorize your IP (it may have changed)

### CUDA out of memory
- Reduce batch size
- Enable mixed precision training (AMP)
- Use gradient checkpointing

### Disconnected during training
- If you used screen/tmux, reconnect with `screen -r` or `tmux attach`
- If you used nohup, training continues - check logs with `tail -f training.log`

### Files missing after restart
- VM boot disk resets some temp files
- Use `/mnt/argo_filestore/` for persistent storage

---

## Cost-Saving Tips

1. **Always stop VMs when not in use** - GPU VMs are expensive
2. **Use L4 VM for debugging** - cheaper than A100
3. **Use A100 only for final training runs**
4. **Store large datasets in Filestore** - don't re-upload each time
5. **Use preemptible/spot instances** for fault-tolerant training (ask Todd to set up)