# GCP VM Guide for ML Training (Windows)

## Overview

**Development workflow**: VSCode Remote-SSH connected to `gpu-vm-l4`. All code editing, running, and debugging happens on the VM. Your local Windows machine is just a thin client — no local Python environment needed.

**Two tools, two jobs**:
- **`vmup.ps1`** (PowerShell script): Start a VM and auto-update its IP in your SSH config. One command per day.
- **VSCode Remote-SSH**: Develop on the running VM (edit files, run scripts, use terminal).

## Quick Reference

| VM Name | Zone | GPU | Use Case |
|---------|------|-----|----------|
| gpu-vm-l4 | us-west1-a | NVIDIA L4 (23 GB) | Development, testing, lighter workloads |
| ml-training-vm | us-west1-b | NVIDIA A100 | Production training |

Persistent storage: `/mnt/argo_filestore` (1TB shared)

---

## Part 1: Daily Workflow

### 1.1 Start the VM

In PowerShell (or double-click the desktop shortcut if you made one):
```powershell
& "C:\Users\Yili Yang\vmup.ps1"
```

For the A100:
```powershell
& "C:\Users\Yili Yang\vmup.ps1" -VM ml-training-vm -Zone us-west1-b
```

The script starts the VM and automatically writes the new external IP into your SSH config. Output should end with `VM up. IP: <something>`.

### 1.2 Connect from VSCode

1. **Ctrl+Shift+P** → **"Remote-SSH: Connect to Host"**
2. Select **gpu-vm-l4** (or **ml-training-vm**)
3. Bottom-left should show **SSH: gpu-vm-l4** in blue
4. Open a terminal (Ctrl+`) and run `nvidia-smi` to confirm GPU access

### 1.3 Activate the Python Environment

In the VSCode terminal:
```bash
source ~/ml-env/bin/activate
cd ~/RTSmappingDL
```

### 1.4 Develop / Train

- **L4**: Run scripts directly for quick feedback
- **A100**: Wrap long jobs in `screen -S training` so they survive disconnects (Ctrl+A then D to detach; `screen -r training` to reconnect)

### 1.5 Stop the VM

In Google Cloud SDK Shell (or any PowerShell with gcloud configured):
```
gcloud compute instances stop gpu-vm-l4 --zone=us-west1-a
```

Or for the A100:
```
gcloud compute instances stop ml-training-vm --zone=us-west1-b
```

**Cost reminder**: L4 ~$0.35/hr idle; A100 ~$4.50/hr idle. **Always stop VMs when done.**

### When to use which VM

- **L4**: Code editing, debugging, `check_data.py`, quick sanity checks
- **A100**: Full training runs, pan-arctic inference (only after confirming readiness on L4)

---

## Part 2: Automation Setup (`vmup.ps1`)

This section sets up the script that Part 1 depends on. Do this once.

### 2.1 Allow PowerShell to Run Local Scripts

Open PowerShell **as Administrator** and run once:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Confirm with **Y**.

### 2.2 Save the Script

Create the file `C:\Users\Yili Yang\vmup.ps1` with this content:

```powershell
param([string]$VM = "gpu-vm-l4", [string]$Zone = "us-west1-a")

Write-Host "Starting $VM in $Zone..."
gcloud compute instances start $VM --zone=$Zone | Out-Null

$ip = gcloud compute instances describe $VM --zone=$Zone `
      --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
$ip = $ip.Trim()

if (-not $ip) {
    Write-Host "Failed to get IP. Check VM status." -ForegroundColor Red
    exit 1
}

$configPath = "$HOME\.ssh\config"
$config = Get-Content $configPath -Raw

# Update HostName line under "Host <VM>" block (handles both LF and CRLF)
$pattern = "(Host\s+$VM\s*[\r\n]+\s*HostName\s+)\S+"
$config = $config -replace $pattern, "`${1}$ip"

Set-Content $configPath -Value $config -NoNewline

Write-Host "VM up. IP: $ip" -ForegroundColor Green
Write-Host "VSCode: Remote-SSH -> $VM" -ForegroundColor Cyan
```

**Important**: the script depends on Part 3.3 — it edits the `Host gpu-vm-l4` and `Host ml-training-vm` blocks in your SSH config. Those blocks must exist before the script can update them.

### 2.3 (Optional) Desktop Shortcut

For one-click VM start:

1. Right-click on Desktop → **New** → **Shortcut**
2. Location:
   ```
   powershell.exe -ExecutionPolicy Bypass -File "C:\Users\Yili Yang\vmup.ps1"
   ```
3. Name: **Start GPU L4**

For the A100, repeat with this location:
```
powershell.exe -ExecutionPolicy Bypass -File "C:\Users\Yili Yang\vmup.ps1" -VM ml-training-vm -Zone us-west1-b
```

### 2.4 What the Script Does Not Cover

- **Zone fallback** — if `us-west1-a` has no capacity, the script fails. Manually start in another zone (see Appendix A.1) and edit the IP in your SSH config that one time.
- **Stopping VMs** — intentional, so the stop command is explicit and not accidentally automated.
- **First-time SSH key/config/permissions setup** — covered in Part 3.

---

## Part 3: First-Time Setup

Skip this part if your VSCode Remote-SSH already connects successfully. Only needed on a new machine or after a reinstall.

### 3.1 Install the VSCode Extension

In VSCode: Extensions panel (Ctrl+Shift+X) → search **"Remote - SSH"** → Install.

### 3.2 Generate SSH Keys

Open Google Cloud SDK Shell and SSH into the VM once. This creates the SSH key pair automatically:
```
gcloud config set project pdg-project-406720
gcloud compute instances start gpu-vm-l4 --zone=us-west1-a
gcloud compute ssh gpu-vm-l4 --zone=us-west1-a
```
Once connected, note two things:
- The **external IP** shown during connection (e.g., `136.109.212.78`)
- Your **VM username** — run `whoami` (e.g., `ext_rtsmapping_woodwellclimate_o`)

Type `exit` to disconnect.

### 3.3 Create the SSH Config File

Create or edit `C:\Users\Yili Yang\.ssh\config` (no file extension) with a plain text editor. Contents:

```
Host gpu-vm-l4
    HostName 136.109.212.78
    User ext_rtsmapping_woodwellclimate_o
    IdentityFile "C:\Users\Yili Yang\.ssh\google_compute_engine"

Host ml-training-vm
    HostName 0.0.0.0
    User ext_rtsmapping_woodwellclimate_o
    IdentityFile "C:\Users\Yili Yang\.ssh\google_compute_engine"
```

**Important**:
- Replace `136.109.212.78` with the L4's current external IP. The A100's `HostName 0.0.0.0` is a placeholder — `vmup.ps1` will fill in the real IP the first time you start it.
- Use **quotes** around the `IdentityFile` path since it contains a space.
- The file must be named exactly `config` — not `config.txt`. Turn on "File name extensions" in File Explorer (View → Show → File name extensions) to verify.

### 3.4 Fix File Permissions

Windows SSH requires strict permissions on both the config file and the private key. Without this, SSH fails with "bad permissions" errors.

For **each** of these two files in `C:\Users\Yili Yang\.ssh\`:
- `config`
- `google_compute_engine` (the private key, no extension)

Do the following:
1. Right-click → **Properties** → **Security** tab → **Advanced**
2. Click **Disable inheritance** → choose **"Remove all inherited permissions from this object"**
3. Click **Add** → **Select a principal** → type your Windows username → **Check Names** → **OK**
4. Grant **Full control** → **OK**
5. Ensure only your user appears in the permissions list — remove all others (especially "OWNER RIGHTS")
6. **Apply** → **OK**

### 3.5 Trust the VM Host Key

Open a regular **PowerShell** (not Cloud SDK Shell) and run:
```
ssh gpu-vm-l4
```

When prompted "Are you sure you want to continue connecting?", type **yes**. This saves the VM's fingerprint to `known_hosts`. Then `exit`.

Repeat for `ml-training-vm` after starting it the first time.

### 3.6 Set Up the Automation Script

Now do Part 2 above (`vmup.ps1` setup). Test it:
```powershell
& "C:\Users\Yili Yang\vmup.ps1"
ssh gpu-vm-l4
```

Should connect cleanly with no prompts.

### 3.7 First Connection from VSCode

1. **Ctrl+Shift+P** → **"Remote-SSH: Connect to Host"** → **gpu-vm-l4**
2. If asked for platform, select **Linux**
3. VSCode installs its server on the VM (takes ~1 min the first time) and connects

**Verify**: Bottom-left shows **"SSH: gpu-vm-l4"**. Open a terminal and run `nvidia-smi`.

### 3.8 Set Up the Python Environment (on the VM)

In the VSCode integrated terminal:

Clone the repo:
```bash
git clone https://github.com/whrc/RTSmappingDL.git
cd RTSmappingDL
```

Create the venv:
```bash
python3 -m venv ~/ml-env
source ~/ml-env/bin/activate
pip install --upgrade pip
```

Install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install project dependencies:
```bash
pip install -r requirements.txt
```

Verify CUDA:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

---

## Part 4: Transferring Files

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

## Part 5: GPU-Task Rules

1. **Always stop VMs when done** — L4 costs ~$0.35/hr idle; A100 costs ~$4.50/hr idle
2. **Develop on L4**, use A100 only for full training runs (after confirming readiness on L4)
3. **Data lives in GCS** — use gcsfuse or `gsutil`, never upload full datasets to VM local disk
4. **Pan-arctic inference** — coordinate with Luigi/Todd (PDG workflow VMs)

---

## Appendix A: Manual Operations

### A.1 Zone Fallback (when start fails)

If the default zone has no capacity, try alternatives in order. In Google Cloud SDK Shell:
```
gcloud compute instances start gpu-vm-l4 --zone=us-west1-c
gcloud compute instances start gpu-vm-l4 --zone=us-west2-a
gcloud compute instances start gpu-vm-l4 --zone=us-west2-b
gcloud compute instances start gpu-vm-l4 --zone=us-central1-a
```

Same pattern for `ml-training-vm`. After a successful start in a non-default zone, manually edit the `HostName` line in your SSH config that one time, since `vmup.ps1` assumes the default zone.

### A.2 Authorize Your IP (if SSH fails after a network change)

Get your current IP:
```
curl -4 ifconfig.me
```

Then update the cluster authorized networks (replace YOUR_IP):
```
gcloud container clusters update autopilot-cluster-1 --region us-west1 --enable-master-authorized-networks --master-authorized-networks YOUR_IP/32
```

### A.3 List All VMs and Their IPs
```
gcloud compute instances list
```

---

## Troubleshooting

### SSH Connection

| Problem | Cause | Fix |
|---------|-------|-----|
| `resource not found` when starting VM | Typo: `gpu-vm-14` (number) vs `gpu-vm-l4` (letter L) | Use lowercase L |
| `Bad permissions` on config or key file | Windows file permissions too open | Redo Part 3.4 |
| `Permission denied (publickey)` | Key file permissions or wrong IdentityFile path | Fix key permissions; verify path in SSH config has quotes |
| `extra arguments at end of line` | Unquoted path with space in `IdentityFile` | Wrap path in double quotes |
| `authenticity of host can't be established` | First-time connection to this IP | Type `yes` to accept and save the fingerprint |
| Connection fails after VM restart | External IP changed | Run `vmup.ps1` — it refreshes the IP in your config |
| `vmup.ps1` not recognized | Wrong working directory | Use full path: `& "C:\Users\Yili Yang\vmup.ps1"` |
| `vmup.ps1` cannot be loaded (execution policy) | First-time PowerShell setup not done | Redo Part 2.1 |

### IAP Tunnel (avoid)

Attempting `gcloud compute start-iap-tunnel` returns error `4033: 'not authorized'`. This is expected — IAP isn't configured for your account on this project, and fixing it requires admin permissions. Stick with the direct-IP workflow via `vmup.ps1`.

### GPU and Training

| Problem | Cause | Fix |
|---------|-------|-----|
| CUDA out of memory | Batch size too large | Reduce batch size, enable mixed precision (AMP) |
| Disconnected during training | SSH dropped | If using screen/tmux, reconnect; if using nohup, check logs |
| Files missing after restart | VM boot disk reset | Use `/mnt/argo_filestore/` or GCS for persistent storage |