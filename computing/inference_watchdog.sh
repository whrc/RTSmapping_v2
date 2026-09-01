#!/usr/bin/env bash
# Auto-stop watchdog for the inference run (plan Phase 4). Runs on the A100
# master. Polls the GCS queue; when every shard is done it STOPS the L4 fleet
# (rts-infer-*) so idle GPUs stop billing. A hard max-runtime backstop guards
# against runaway cost.
#
# HARD SAFETY (non-negotiable):
#   * Only ever stops instances whose name matches ^rts-infer-[0-9]+$.
#   * NEVER stops/touches the A100 master (a100-8x-train) — scarcity: a stopped
#     A100-80GB may not be reclaimable. The master is intentionally always-on.
#   * Stops (does NOT delete) the L4 VMs — kept for re-runs/products.
#   * Run-active gated: only acts while a `run_active` sentinel exists, so it is
#     inert during dev (set the sentinel at launch, it self-clears at completion).
#
# Usage (on the master, backgrounded):
#   RUN_BASE=gs://rts-arctic-usw1/inference/2025q3_south \
#   nohup bash computing/inference_watchdog.sh > /mnt/outputs/watchdog.log 2>&1 &
set -euo pipefail

PROJECT="${PROJECT:-abruptthawmapping}"
ZONE="${ZONE:-us-west1-a}"
POLL_S="${POLL_S:-120}"
MAX_HOURS="${MAX_HOURS:-48}"          # backstop: stop the fleet after this long no matter what
: "${RUN_BASE:?set RUN_BASE=gs://.../inference/2025q3_south}"

SENTINEL="$RUN_BASE/run_active"
SAFE_RE='^rts-infer-[0-9]+$'
t0=$(date +%s)

log() { echo "[watchdog $(date -u +%H:%M:%S)] $*"; }

n_shards() {
  gcloud storage cat "$RUN_BASE/shards/index.json" 2>/dev/null \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['n_shards'])" 2>/dev/null || echo -1
}
n_done() {
  gcloud storage ls "$RUN_BASE/done/" 2>/dev/null | grep -c '/done/.' || true
}

stop_fleet() {
  local reason="$1"
  log "STOPPING L4 fleet ($reason)"
  # List only our fleet, filter again client-side against the safe regex.
  mapfile -t vms < <(gcloud compute instances list --project "$PROJECT" \
    --filter="name~^rts-infer- AND zone~$ZONE" --format="value(name)" 2>/dev/null || true)
  for v in "${vms[@]}"; do
    if [[ "$v" =~ $SAFE_RE ]]; then
      log "  stopping $v"
      gcloud compute instances stop "$v" --zone "$ZONE" --project "$PROJECT" --quiet || log "  WARN: stop $v failed"
    else
      log "  REFUSING to touch '$v' (does not match $SAFE_RE)"
    fi
  done
  gcloud storage rm "$SENTINEL" 2>/dev/null || true
  log "fleet stopped; run_active cleared"
}

log "watchdog up: base=$RUN_BASE poll=${POLL_S}s max=${MAX_HOURS}h"
while true; do
  if ! gcloud storage ls "$SENTINEL" >/dev/null 2>&1; then
    log "no run_active sentinel — idle, exiting (set $SENTINEL at launch to arm)"
    exit 0
  fi
  total=$(n_shards); done=$(n_done)
  elapsed_h=$(( ($(date +%s) - t0) / 3600 ))
  log "progress: $done/$total shards done (${elapsed_h}h elapsed)"

  if [ "$total" -gt 0 ] && [ "$done" -ge "$total" ]; then
    stop_fleet "all $total shards done"
    exit 0
  fi
  if [ "$elapsed_h" -ge "$MAX_HOURS" ]; then
    stop_fleet "MAX_HOURS=$MAX_HOURS backstop tripped at $done/$total — INVESTIGATE"
    exit 1
  fi
  sleep "$POLL_S"
done
