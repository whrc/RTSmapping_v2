# GCP VM Guide for ML Training (Windows)

## Overview

**Development workflow**: VSCode Remote-SSH connected to `gpu-vm-l4`. All code editing, running, and debugging happens on the VM. Your local Windows machine is just a thin client — no local Python environment needed.

**Two tools, two jobs**:
- **Google Cloud SDK Shell** (Windows app): Start/stop VMs, manage GCP resources
- **VSCode Remote-SSH**: Develop on the running VM (edit files, run scripts, use terminal)

## Quick Reference

| VM Name | Zone | GPU | Use Case |
|---------|------|-----|----------|
| gpu-vm-l4 | us-west1-a | NVIDIA L4 (23 GB) | Development, testing, lighter workloads |
| ml-training-vm | us-west1-b | NVIDIA A100 | Production training |

Persistent storage: `/mnt/argo_filestore` (1TB shared)

---

## Part 1: Start the VM

All commands in this section run in **Google Cloud SDK Shell** (launch from Windows Start Menu).

### 1.1 Set Project (once per session)
```
gcloud config set project pdg-project-406720
```

### 1.2 Check VM Status
```
gcloud compute instances list
```

### 1.3 Authorize Your IP (if SSH fails after IP change)
```
curl ifconfig.me
```
Then (replace YOUR_IP):
```
gcloud container clusters update autopilot-cluster-1 --region us-west1 --enable-master-authorized-networks --master-authorized-networks YOUR_IP/32
```

### 1.4 Start the VM
```
gcloud compute instances start gpu-vm-l4 --zone=us-west1-a
```

**Watch out**: `gpu-vm-l4` uses a lowercase **L**, not the number **1**. These look identical in most terminal fonts.

---

## Part 2: VSCode Remote-SSH Setup (One-Time)

### 2.1 Install the Extension

In VSCode: Extensions panel (Ctrl+Shift+X) → search **"Remote - SSH"** → Install.

### 2.2 Generate SSH Keys

Open Google Cloud SDK Shell and SSH into the VM once. This creates the SSH key pair automatically:
```
gcloud compute ssh gpu-vm-l4 --zone=us-west1-a
```
Once connected, note two things:
- The **external IP** shown during connection (e.g., `136.109.212.78`)
- Your **VM username** — run `whoami` (e.g., `ext_rtsmapping_woodwellclimate_o`)

Type `exit` to disconnect.

### 2.3 Create the SSH Config File

Create or edit the file `C:\Users\<YourWindowsName>\.ssh\config` (no file extension) with a plain text editor. Contents:

```
Host gpu-vm-l4
    HostName 136.109.212.78
    User ext_rtsmapping_woodwellclimate_o
    IdentityFile "C:\Users\Yili Yang\.ssh\google_compute_engine"
```

**Important**:
- Replace `136.109.212.78` with the VM's current external IP
- Replace `Yili Yang` with your actual Windows username folder
- Use **quotes** around the `IdentityFile` path if it contains spaces
- The file must be named exactly `config` — not `config.txt`. Turn on "File name extensions" in File Explorer (View → Show → File name extensions) to verify.

### 2.4 Fix File Permissions

Windows SSH requires strict permissions on both the config file and the private key. Without this, VSCode will fail to connect with "bad permissions" errors.

For **each** of these two files in `C:\Users\<YourWindowsName>\.ssh\`:
- `config`
- `google_compute_engine` (the private key, no extension)

Do the following:
1. Right-click → **Properties** → **Security** tab → **Advanced**
2. Click **Disable inheritance** → choose **"Remove all inherited permissions from this object"**
3. Click **Add** → **Select a principal** → type your Windows username → **Check Names** → **OK**
4. Grant **Full control** → **OK**
5. Ensure only your user appears in the permissions list — remove all others (especially "OWNER RIGHTS")
6. **Apply** → **OK**

### 2.5 Trust the VM Host Key

Open a regular **Command Prompt** or **PowerShell** (not Cloud SDK Shell) and run:
```
ssh -i "C:\Users\Yili Yang\.ssh\google_compute_engine" ext_rtsmapping_woodwellclimate_o@136.109.212.78
```

When prompted "Are you sure you want to continue connecting?", type **yes**. This saves the VM's fingerprint to `known_hosts` so future connections proceed without prompting. Then `exit`.

### 2.6 Connect from VSCode

1. Press **Ctrl+Shift+P** → type **"Remote-SSH: Connect to Host"**
2. Select **gpu-vm-l4**
3. If asked for platform, select **Linux**
4. VSCode opens a new window, installs its server on the VM, and connects

**Verify**: Bottom-left corner should show **"SSH: gpu-vm-l4"** in blue. Open a terminal (Ctrl+`) and run `nvidia-smi` to confirm GPU access.

---

## Part 3: Daily Workflow

### 3.1 Start Your Session

1. Open **Google Cloud SDK Shell** → start the VM:
   ```
   gcloud compute instances start gpu-vm-l4 --zone=us-west1-a
   ```
2. Open **VSCode** → Ctrl+Shift+P → "Remote-SSH: Connect to Host" → `gpu-vm-l4`
3. Open your project folder: **File → Open Folder** → navigate to `~/RTSmappingDL`

### 3.2 End Your Session

1. Close the VSCode remote window
2. In Google Cloud SDK Shell, **stop the VM** (prevents charges):
   ```
   gcloud compute instances stop gpu-vm-l4 --zone=us-west1-a
   ```
3. Verify it stopped:
   ```
   gcloud compute instances list
   ```
   Status should show "TERMINATED".

**Always stop VMs when not in use.** GPU VMs are expensive even when idle.

---

## Part 4: Environment Setup (First Time Only)

All commands run in the VSCode integrated terminal (which is on the VM).

### 4.1 Clone the Repo
```bash
git clone https://github.com/whrc/RTSmappingDL.git
cd RTSmappingDL
```

### 4.2 Create Python Virtual Environment
```bash
python3 -m venv ~/ml-env
source ~/ml-env/bin/activate
pip install --upgrade pip
```

### 4.3 Install Dependencies

Install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:
```bash
pip install -r requirements.txt
```

### 4.4 Verify CUDA
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### 4.5 Activate Environment in Future Sessions
```bash
source ~/ml-env/bin/activate
```

---

## Part 5: Transferring Files

### Option A: Persistent Filestore (large datasets)
```bash
ls /mnt/argo_filestore/
mkdir -p /mnt/argo_filestore/yili
```

### Option B: Google Cloud Storage
```bash
gsutil cp gs://abruptthawmapping/path/to/file ~/data/
gsutil -m cp -r gs://abruptthawmapping/folder ~/data/  # recursive, parallel
```

### Option C: Upload from Local Machine
From Google Cloud SDK Shell (not the VM):
```
gcloud compute scp "C:\path\to\local\file" gpu-vm-l4:~/file --zone=us-west1-a
gcloud compute scp --recurse "C:\path\to\folder" gpu-vm-l4:~/folder --zone=us-west1-a
```

---

## Part 6: Running Training

### Run a Script
```bash
source ~/ml-env/bin/activate
cd ~/RTSmappingDL
python scripts/check_data.py --config configs/baseline.yaml
```

### Run in Background (survives SSH disconnect)

Using screen (recommended):
```bash
screen -S training
source ~/ml-env/bin/activate
python scripts/train.py --config configs/baseline.yaml
# Ctrl+A then D to detach
# Reconnect later: screen -r training
```

Using tmux:
```bash
tmux new -s training
source ~/ml-env/bin/activate
python scripts/train.py --config configs/baseline.yaml
# Ctrl+B then D to detach
# Reconnect later: tmux attach -t training
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

---

## Part 7: Production Training (A100/H100)

### Start and Connect
```
gcloud compute instances start ml-training-vm --zone=us-west1-b
gcloud compute ssh ml-training-vm --zone=us-west1-b
```

To use VSCode Remote-SSH with the production VM, add a second entry to your SSH config with the production VM's IP and repeat the setup steps from Part 2.

### Stop When Done
```
gcloud compute instances stop ml-training-vm --zone=us-west1-b
```

---

## Part 8: GPU-Task Rules

| Task | VM | Rationale |
|------|----|-----------|
| Code editing, exploration, debugging | L4 VM | Cheapest GPU; sufficient for single-step tests |
| Data validation, `check_data.py` | L4 VM | No heavy compute needed |
| Short training runs, sanity checks | L4 VM | Fast feedback loop |
| Full experiment training | A100/H100 VM | High throughput needed |
| Pan-arctic inference | PDG workflow VMs | Coordinate with Luigi/Todd |

**Rules**:
1. Always stop VMs when not in use
2. Develop and iterate on L4 — same Docker image as production, cheaper cost
3. Use A100/H100 only for full training runs — confirm runs are ready before switching
4. Data lives in GCS — never upload full dataset to VM local disk; use gcsfuse
5. Use preemptible/spot instances for long training runs when possible

---

## Troubleshooting

### SSH Connection

| Problem | Cause | Fix |
|---------|-------|-----|
| `resource not found` when starting VM | Typo: `gpu-vm-14` (number) vs `gpu-vm-l4` (letter L) | Use lowercase L |
| `Bad permissions` on config or key file | Windows file permissions too open | Fix permissions: right-click → Properties → Security → Advanced → remove inheritance, grant only your user Full control |
| `Permission denied (publickey)` | Key file permissions or wrong IdentityFile path | Fix key permissions; verify path in SSH config |
| `authenticity of host can't be established` | First-time connection to this IP | Type `yes` to accept and save the host fingerprint |
| VSCode shows `gpu-vm-l4` not in host list | Config file named `config.txt` instead of `config` | Enable file extensions in Explorer, rename to `config` |
| VSCode asks for platform | First connection to this host | Select **Linux** |
| Connection fails after VM restart | External IP changed | Check new IP with `gcloud compute instances list`, update SSH config `HostName` |

### GPU and Training

| Problem | Cause | Fix |
|---------|-------|-----|
| CUDA out of memory | Batch size too large | Reduce batch size, enable mixed precision (AMP) |
| Disconnected during training | SSH dropped | If using screen/tmux, reconnect; if using nohup, check logs |
| Files missing after restart | VM boot disk reset | Use `/mnt/argo_filestore/` or GCS for persistent storage |