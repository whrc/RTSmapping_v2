#!/usr/bin/env bash
# Auto-retry creating a 2x H100 VM across us-west1-a / us-west1-b until capacity frees up.
#
# PREREQS:
#   1. gcloud must be authed as YOU (not the VM's compute SA, which lacks create perms):
#        gcloud auth login           # browser/device flow
#        gcloud config set project abruptthawmapping
#   2. Paste YOUR exact create command into create_cmd() below — get it from the GCP
#      console create-VM page → "Equivalent code" → gcloud — and replace its hard-coded
#      --zone=... with  --zone="$ZONE"  so the loop can alternate zones.
#
# RUN (detached, survives logout):
#   nohup bash ~/create_h100_vm.sh > ~/create_h100_vm.log 2>&1 &
#   tail -f ~/create_h100_vm.log
#
# It retries ONLY on capacity stockouts; any other error (quota / permission / bad spec)
# stops immediately so you don't loop forever on a problem retries can't fix.
set -u

ZONES=(us-west1-a us-west1-b)
INTERVAL="${INTERVAL:-45}"     # seconds between full a+b sweeps
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"   # 0 = unlimited

# ---------------------------------------------------------------------------
# >>> EDIT THIS: paste your console "Equivalent code", with --zone="$ZONE" <<<
# ---------------------------------------------------------------------------
create_cmd() {
  gcloud compute instances create "h100-2x-train" \
    --project="abruptthawmapping" \
    --zone="$ZONE" \
    --machine-type="a3-highgpu-2g" \
    --maintenance-policy="TERMINATE" \
    --provisioning-model="STANDARD" \
    --image-family="common-cu129-ubuntu-2204-nvidia-580" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-size="500GB" \
    --boot-disk-type="pd-ssd" \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME \
    --local-ssd=interface=NVME \
    --metadata="install-nvidia-driver=True"
    # 4x NVMe local SSD = the canonical a3-highgpu-2g shape (GCP bundled the same 4x375GB in the
    # reservation it generated). They are ephemeral scratch — do NOT put checkpoints there.
    # add --network / --service-account etc. as needed
}
# ---------------------------------------------------------------------------

CAPACITY_RE="ZONE_RESOURCE_POOL_EXHAUSTED|does not have enough resources|resource pool exhausted|currently unavailable|stockout|ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS"

attempt=0
while :; do
  attempt=$((attempt + 1))
  for ZONE in "${ZONES[@]}"; do
    echo "[$(date -u '+%F %T UTC')] attempt #$attempt — trying $ZONE ..."
    OUT="$(create_cmd 2>&1)"; rc=$?
    echo "$OUT"
    if [ $rc -eq 0 ]; then
      echo "[$(date -u '+%F %T UTC')] ✅ CREATED in $ZONE after $attempt attempt(s). It is now BILLING."
      exit 0
    fi
    if echo "$OUT" | grep -qiE "$CAPACITY_RE"; then
      echo "[$(date -u '+%F %T UTC')] capacity stockout in $ZONE — will retry."
    else
      echo "[$(date -u '+%F %T UTC')] ❌ NON-capacity error (quota / permission / spec). Stopping — fix it, then re-run."
      exit 1
    fi
  done
  if [ "$MAX_ATTEMPTS" -gt 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    echo "[$(date -u '+%F %T UTC')] reached MAX_ATTEMPTS=$MAX_ATTEMPTS without capacity. Giving up."
    exit 2
  fi
  sleep "$INTERVAL"
done
