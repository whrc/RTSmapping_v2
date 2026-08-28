#!/usr/bin/env bash
# Create the L4 inference fleet (rts-infer-{1..N}), plan Phase 3.
#
# Each VM is a g2-standard-96 (8x L4, us-west1), on-demand, running
# inference_fleet_startup.sh (pulls rts-infer:v1, launches 8 queue workers).
# The fleet auto-balances against the 8 A100-master workers via the shared GCS
# shard-claim queue. Resources are strictly the rts-infer-* instances we own
# (shared PDG project — never touch anything else).
#
# Quota (live 2026-06-26): NVIDIA_L4_GPUS=32 (1 already used elsewhere -> 31
# schedulable), CPUS=480. 4x g2-standard-96 needs 32 L4 -> the 4th may fail
# while that 1 L4 is held; default N=3 (24 L4) is safe. Bump to 4 only after the
# phantom L4 clears or a 32->40 grant. Spot is NOT viable (PREEMPTIBLE_L4=8).
#
# Creates VMs ONE AT A TIME and stops on the first failure (plan: validate one
# VM before the rest; never exceed quota). Re-running skips VMs that already
# exist. Teardown: `gcloud compute instances stop rts-infer-*` (stop, NOT delete
# — keep for re-runs/products; the auto-stop watchdog does this at completion).
#
# Usage:
#   N_VMS=3 RUN_BASE=gs://rts-mapping-v2-usw1/inference/2025q3_south \
#   QUAD_INDEX=gs://.../quad_index_2025q3.csv \
#   S2_INDEX=gs://.../s2_index_2025_south.csv \
#   bash computing/create_inference_fleet.sh
set -euo pipefail

PROJECT="${PROJECT:-abruptthawmapping}"
ZONE="${ZONE:-us-west1-a}"
MACHINE="${MACHINE:-g2-standard-96}"          # 8x L4 inherent to the machine type
GPUS_PER_VM="${GPUS_PER_VM:-8}"
N_VMS="${N_VMS:-3}"                            # safe default given the 31 schedulable L4
IMAGE="${IMAGE:-us-west1-docker.pkg.dev/abruptthawmapping/rts/rts-infer:v1}"
DL_WORKERS="${DL_WORKERS:-8}"                  # DataLoader workers per GPU worker
# DLVM image with NVIDIA drivers + Docker + nvidia-container-toolkit preinstalled.
# (common-cu123 was retired by Google — family verified live 2026-07-05 audit.)
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
BOOT_DISK_GB="${BOOT_DISK_GB:-200}"
SA="${SA:-}"                                   # optional explicit service account

# Required run parameters (no defaults — fail loudly if unset).
: "${RUN_BASE:?set RUN_BASE=gs://.../inference/2025q3_south}"
: "${QUAD_INDEX:?set QUAD_INDEX=gs://.../quad_index_2025q3.csv}"
: "${S2_INDEX:?set S2_INDEX=gs://.../s2_index_2025_south.csv}"
PACKAGES="${PACKAGES:-$RUN_BASE/packages/seed42,$RUN_BASE/packages/seed43,$RUN_BASE/packages/seed44}"

STARTUP="$(dirname "$0")/inference_fleet_startup.sh"
[ -f "$STARTUP" ] || { echo "missing $STARTUP"; exit 1; }

echo "Creating $N_VMS x $MACHINE in $ZONE (image $IMAGE)"
echo "  run base: $RUN_BASE"
echo "  packages: $PACKAGES"

# Always request cloud-platform scopes: without them the VM gets the legacy
# default devstorage.read_only scope and every GCS write 403s regardless of
# IAM (found live, 2026-07-05 audit). IAM stays the real access control.
sa_args=(--scopes cloud-platform)
[ -n "$SA" ] && sa_args=(--service-account "$SA" --scopes cloud-platform)

for i in $(seq 1 "$N_VMS"); do
  name="rts-infer-$i"
  if gcloud compute instances describe "$name" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
    echo "  $name already exists — skipping"
    continue
  fi
  echo "=== creating $name ==="
  gcloud compute instances create "$name" \
    --project "$PROJECT" --zone "$ZONE" \
    --machine-type "$MACHINE" \
    --maintenance-policy TERMINATE \
    --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" \
    --boot-disk-size "${BOOT_DISK_GB}GB" --boot-disk-type pd-balanced \
    --labels "owner=rts,purpose=inference,fleet=rts-infer" \
    --metadata "^|^install-nvidia-driver=True|docker-image=$IMAGE|run-base=$RUN_BASE|quad-index=$QUAD_INDEX|s2-index=$S2_INDEX|packages=$PACKAGES|gpus-per-vm=$GPUS_PER_VM|dataloader-workers=$DL_WORKERS" \
    --metadata-from-file "startup-script=$STARTUP" \
    "${sa_args[@]}" \
    || { echo "FAILED to create $name (quota/capacity?). Stopping — fix before continuing."; exit 1; }
  echo "  $name created"
done

echo "Done. Watch with scripts/inference_progress.py --base $RUN_BASE --watch 60"
echo "Per-VM logs: gcloud compute ssh rts-infer-N --zone $ZONE --command 'docker ps; docker logs rts-worker-0'"
