# Computing — machines, and which doc answers which question

**Single source of truth for the host registry**: every machine this project runs on, its
codename, what it does, and whether it still exists.

> **Register every machine here.** A new VM gets a row the moment it is created; a deleted
> one gets its row struck through, not removed — a machine that used to exist is a fact
> someone will need. Machines that are *not ours* get rows too: that is the only reliable
> way to stop a future reader reaching for them.

Infra facts other than machines — projects, buckets, regions, quota, the storage map — live
in [infrastructure.md](infrastructure.md), which points here rather than repeating the
inventory.

---

## 1. Host registry

| Codename | Kind | Where | Function | Status |
|---|---|---|---|---|
| **`ARCHITECTURE`** | Office PC (Windows) | Woodwell office | **Control node** — code editing, `gcloud compute` lifecycle, VSCode Remote-SSH origin. Never runs unattended work. | **active — configured 2026-08-27** |
| `rts-ops` | `e2-standard-2`, no external IP (IAP) | `abruptthawmapping` / us-west1-a | Unattended loops: Planet acquisition, S2 export driver, cron alerters. Shared with Heidi. | **active — created 2026-08-27**; checkout/Docker/venv/image done, ADC + webhook + cron outstanding ([pdg_migration.md §5b](pdg_migration.md)) |
| `rts-review` | `e2-small`, static IP **34.83.225.204**, `:80` public | `abruptthawmapping` / us-west1-a | Polygon-rating app for the review campaign | **active — deployed 2026-08-28**; reviewers not yet cut over |
| ~~`a100-8x-train`~~ | `a2-ultragpu-8g`, 8× A100-80GB | PDG / us-central1-a | Training, pan-Arctic inference, and — by accretion — every other running process | retiring 2026-09-06 (PDG funding ends 09-07) |
| ~~`rts-review-vm`~~ | `e2-small`, `8.229.247.193` | PDG / us-west1-a | Rating app, superseded by `rts-review` | retiring 2026-09-06 |
| ~~`ml-training-vm`~~ | single A100-40GB | PDG / us-west1-b | Production training before the 8×A100 landed | deleted 2026-06-24 |
| ~~`gpu-vm-l4`~~ | NVIDIA L4 | PDG / us-west1-a | Dev/test box | deleted |
| ~~`rts-infer-usw1` fleet~~ | 3–4× `g2-standard-96` (24–32× L4) | PDG / us-west1 | 2025 South inference fleet | torn down after the South run |
| `download-vm` | `n2-standard-4` | PDG / us-west1-a | **Not ours** — pre-existing PDG instance | do not touch |
| `gke-water-cluster-*` | `n2-highmem-4` ×2 | PDG / us-west1-b/c | **Not ours** — another team's GKE pool | do not touch |

**Naming**: every project-owned VM is prefixed `rts-` (GCE names disallow underscores), so
ours are identifiable in a shared org project.

## 2. Control plane vs execution plane

`ARCHITECTURE` is where you type. `rts-ops` is where unattended things run.

The split matters because the loops this project depends on run for days — a 131-hour Planet
ordering run, a GEE export that idles for hours between tasks, cron alerters every ten
minutes. None of that may depend on a desk machine's sleep, reboot or VPN. Anything that must
survive you closing your laptop goes on `rts-ops` under `tmux` or cron.

The previous convention made `a100-8x-train` the control node. That existed only because IAP
tunnelling was not authorized in the PDG project, so lifecycle commands had to originate from
inside it. `abruptthawmapping` has no such restriction, so control moves to the desk and
`rts-ops` needs no external IP at all.

Setup: [control_node.md](control_node.md).

## 3. Which doc answers which question

| Question | Doc |
|---|---|
| What machines exist? What is each for? | **this file** |
| Which project, bucket, region? What is the quota? | [infrastructure.md](infrastructure.md) |
| How do I drive it all from my office PC? | [control_node.md](control_node.md) |
| How do I build and run the containers? | [docker_training.md](docker_training.md) |
| Where does artifact X live? | [artifact_inventory.md](artifact_inventory.md) |
| Day-to-day on a VM — env, file transfer | [vm_instruction.md](vm_instruction.md) |
| Why did everything move in Sept 2026? | [pdg_migration.md](pdg_migration.md) |
| What is the exact order of the final cutover? | [cutover_runbook.md](cutover_runbook.md) |
