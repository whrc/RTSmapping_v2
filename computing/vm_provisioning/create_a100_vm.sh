#!/usr/bin/env bash
# Auto-retry creating a 8x A100-80GB VM across us-central1-a / us-central1-c until capacity frees up.
#
# PREREQS:
#   1. gcloud must be authed as YOU (not the VM's compute SA, which lacks create perms):
#        gcloud auth login           # browser/device flow
#        gcloud config set project pdg-project-406720
#   2. Paste YOUR exact create command into create_cmd() below — get it from the GCP
#      console create-VM page → "Equivalent code" → gcloud — and replace its hard-coded
#      --zone=... with  --zone="$ZONE"  so the loop can alternate zones.
#
# RUN (detached, survives logout):
#   nohup bash ~/create_a100_vm.sh > ~/create_a100_vm.log 2>&1 &
#   tail -f ~/create_a100_vm.log
#
# It retries ONLY on capacity stockouts; any other error (quota / permission / bad spec)
# stops immediately so you don't loop forever on a problem retries can't fix.
set -u

ZONES=(us-central1-a us-central1-c)
INTERVAL="${INTERVAL:-45}"     # seconds between full a+b sweeps
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"   # 0 = unlimited

# ---------------------------------------------------------------------------
# >>> EDIT THIS: paste your console "Equivalent code", with --zone="$ZONE" <<<
# ---------------------------------------------------------------------------
create_cmd() {
  gcloud compute instances create "a100-8x-train" \
    --project="pdg-project-406720" \
    --zone="$ZONE" \
    --machine-type="a2-ultragpu-8g" \
    --maintenance-policy="TERMINATE" \
    --provisioning-model="STANDARD" \
    --image-family="common-cu129-ubuntu-2204-nvidia-580" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-size="500GB" \
    --boot-disk-type="pd-ssd" \
    --metadata="install-nvidia-driver=True"
    # a2-ultragpu-8g bundles 8x A100-80GB GPUs and local SSD with the machine type —
    # no accelerator/local-ssd flags needed. Local SSD is ephemeral scratch, NOT for checkpoints.
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
