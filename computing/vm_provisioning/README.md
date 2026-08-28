# VM provisioning scripts

Captured from `a100-8x-train:/home/ext_yyang_woodwellclimate_org/` on 2026-08-28, hours
before that machine was due for deletion. They had never been in git — they lived only on
the box they created, which is the worst possible place for the record of how a machine
was built.

Both scripts do the same thing: retry `gcloud compute instances create` across two zones
until GPU capacity frees up, and **stop immediately on any non-capacity error**, because a
quota, permission or spec problem is not something retrying can fix. That distinction is
the reason these exist rather than a one-line create command.

| Script | Machine | GPUs | Zones tried |
|---|---|---|---|
| `create_a100_vm.sh` | `a2-ultragpu-8g` | 8× A100-80GB | `us-central1-a`, `us-central1-c` |
| `create_h100_vm.sh` | `a3-highgpu-2g` | 2× H100-80GB | `us-central1-a`, `us-central1-c` |

Both pin `--image-family=common-cu129-ubuntu-2204-nvidia-580` from
`deeplearning-platform-release`, a 500 GB pd-ssd boot disk, and
`--maintenance-policy=TERMINATE` (GPU VMs cannot live-migrate).

They target **`pdg-project-406720`, which is retired**. Anything reusing them must change
`--project` and the zones, and check the image family still exists — CUDA image families
are retired on their own schedule. Kept for the machine spec and the retry logic, not as
a runnable command.

`--metadata=install-nvidia-driver=True` is what makes the DLVM image install the driver on
first boot; without it the box comes up with GPUs and no driver.

Environment setup *after* creation is `computing/vm_instruction.md`; this directory only
covers getting the hardware.
