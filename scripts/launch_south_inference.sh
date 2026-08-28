#!/usr/bin/env bash
# Supervised master-only launch of the pan-Arctic 2025q3_south inference run.
#
# Runs one rts-infer worker container per GPU on the 8xA100 master, each draining
# the GCS shard-claim queue (inference/claim.py + scripts/run_inference_worker.py).
# A HOST-side supervisor wraps every GPU so *silent* failures self-heal:
#   * a worker that crashes, OOMs, or is preempted exits non-zero -> logged + restarted;
#   * a worker whose DataLoader wedges never exits on its own (the Banks GPU-0 hang) —
#     but runner._start_stall_watchdog os._exit(3)s it after inference.stall_timeout_s
#     of no tile progress, which the supervisor then sees as a non-zero exit + restarts.
# A stalled shard's claim goes stale once its heartbeat thread dies with the process,
# so another worker reclaims it and resumes from the per-shard manifest (exactly-once).
# A clean exit 0 (queue drained) retires that GPU's supervisor.
#
# Crash-loop guard: a worker that dies non-zero in under MIN_HEALTHY_S counts as a fast
# failure; MAX_FAST_FAILS consecutive fast failures retires the GPU with a FATAL line
# (so a misconfig can't spin forever). A run that lasts >= MIN_HEALTHY_S resets the count.
#
# NEVER stops/renames/deletes the master VM (A100 scarcity). Stop the whole run with:
#   touch /mnt/outputs/inference/south/STOP      # supervisors finish the current shard, then exit
#
# Usage (detached so it survives the SSH session):
#   nohup bash scripts/launch_south_inference.sh > /mnt/outputs/inference/south/launch.log 2>&1 &
#   watch -n 15 'bash scripts/monitor_jobs.sh rts-infer'   # live view
#
# Env (all defaulted): BASE QUAD_INDEX S2_INDEX CONFIG IMAGE NGPU NUM_WORKERS LOGDIR
#                      STOP_FILE RESTART_DELAY_S MIN_HEALTHY_S MAX_FAST_FAILS GCP_PROJECT ADC_PATH
set -u

BASE="${BASE:-gs://rts-arctic-usw1/inference/2025q3_south}"
QUAD_INDEX="${QUAD_INDEX:-$BASE/quad_index_2025q3.csv}"
S2_INDEX="${S2_INDEX:-$BASE/s2_index_2025_south.csv}"
CONFIG="${CONFIG:-configs/deployment.yaml}"
IMAGE="${IMAGE:-us-west1-docker.pkg.dev/abruptthawmapping/rts/rts-infer:v1}"
NGPU="${NGPU:-8}"
NUM_WORKERS="${NUM_WORKERS:-16}"   # measured 2026-07-07: 8→16 workers ~doubled t/s (12→~24 t/s/A100)
#                                    by hiding cross-region read latency; master has ~61 spare vCPUs
LOGDIR="${LOGDIR:-/mnt/outputs/inference/south/logs}"
STOP_FILE="${STOP_FILE:-/mnt/outputs/inference/south/STOP}"
RESTART_DELAY_S="${RESTART_DELAY_S:-15}"
MIN_HEALTHY_S="${MIN_HEALTHY_S:-180}"   # a run shorter than this (non-zero) is a fast failure
MAX_FAST_FAILS="${MAX_FAST_FAILS:-5}"   # consecutive fast failures before a GPU is retired
ADC="${ADC_PATH:-$HOME/.config/gcloud/application_default_credentials.json}"
# space-separated gs:// seed package dirs (3-seed ensemble)
PACKAGES="${PACKAGES:-$BASE/packages/seed42 $BASE/packages/seed43 $BASE/packages/seed44}"

mkdir -p "$LOGDIR"
rm -f "$STOP_FILE"
pkg_args=(); for p in $PACKAGES; do pkg_args+=(--package "$p"); done

supervise_gpu() {  # $1=gpu index — restarts its worker until queue-drained, STOP, or crash-loop
  local g="$1" n=0 rc t0 dt fast=0
  local name="rts-infer-g${g}" glog="$LOGDIR/gpu_${g}.log"
  while true; do
    if [ -f "$STOP_FILE" ]; then
      echo "[gpu $g] $(date -Is) STOP present — not (re)starting"; break
    fi
    n=$((n+1))
    echo "[gpu $g] $(date -Is) START attempt #$n (container $name)"
    sudo docker rm -f "$name" >/dev/null 2>&1 || true
    t0=$(date +%s)
    # --privileged exposes all GPUs, so pin the device via CUDA_VISIBLE_DEVICES
    # (run_gpu_pool.sh convention). --shm-size for the forkserver DataLoader workers.
    sudo docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES="$g" --privileged \
        --shm-size=16g --name "$name" \
        -v "$ADC:/gcp_adc.json:ro" \
        -e GOOGLE_APPLICATION_CREDENTIALS=/gcp_adc.json \
        -e GOOGLE_CLOUD_PROJECT="${GCP_PROJECT:-abruptthawmapping}" \
        -e PYTHONPATH=/app \
        -e GDAL_HTTP_MAX_RETRY=3 -e GDAL_HTTP_RETRY_DELAY=1 \
        "$IMAGE" \
        scripts/run_inference_worker.py \
          --config "$CONFIG" --base "$BASE" \
          --quad-index "$QUAD_INDEX" --s2-index "$S2_INDEX" \
          "${pkg_args[@]}" --num-workers "$NUM_WORKERS" \
        >> "$glog" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ "$rc" -eq 0 ]; then
      echo "[gpu $g] $(date -Is) worker exited 0 after ${dt}s (queue drained) — retiring GPU $g"; break
    fi
    if [ "$dt" -lt "$MIN_HEALTHY_S" ]; then
      fast=$((fast+1))
      echo "[gpu $g] $(date -Is) worker exited $rc after only ${dt}s (fast failure ${fast}/${MAX_FAST_FAILS})"
      if [ "$fast" -ge "$MAX_FAST_FAILS" ]; then
        echo "[gpu $g] $(date -Is) FATAL: ${MAX_FAST_FAILS} consecutive fast failures — retiring GPU $g. See $glog"; break
      fi
    else
      fast=0
      echo "[gpu $g] $(date -Is) worker exited $rc after ${dt}s (likely stall-watchdog/preemption) — restarting"
    fi
    sleep "$RESTART_DELAY_S"
  done
}

echo "[launch] $(date -Is) 2025q3_south master-only run: NGPU=$NGPU workers/GPU=$NUM_WORKERS"
echo "[launch] $(date -Is) base=$BASE image=$IMAGE"
# Fail fast if the shard queue is missing (nothing to claim).
if ! gsutil -q stat "$BASE/shards/index.json" 2>/dev/null; then
  echo "[launch] $(date -Is) ERROR: $BASE/shards/index.json not found — build the queue first (scripts/shard_tiles.py)"; exit 1
fi

pids=()
for ((g=0; g<NGPU; g++)); do
  supervise_gpu "$g" &
  pids+=($!)
  echo "[launch] $(date -Is) supervisor GPU $g -> pid $!"
done
wait "${pids[@]}"
echo "[launch] $(date -Is) all supervisors exited (queue drained, STOP, or crash-loop retirement)"
