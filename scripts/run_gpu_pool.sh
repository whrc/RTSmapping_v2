#!/usr/bin/env bash
# Dynamic multi-GPU work-pool dispatcher for the ablation program.
#
# Unlike run_ablation_queue.sh (one STATIC sequential queue per GPU — idle GPUs
# when a queue drains early), this keeps NGPU runs in flight at ALL times: when
# any run finishes, the next pending config launches on the freed GPU. Drive one
# *independent* wave per invocation; sequential-elimination dependencies
# (e.g. loss → boundary) are enforced by invoking the pool once per wave.
#
# Args are config BASENAMES → configs/<name>.yaml. Outputs match the versioned
# convention (see /mnt/outputs/README.md):
#   run dir : /outputs/<VERSION>/runs/<name>
#   mlflow  : /outputs/<VERSION>/mlflow/<name>
#   run log : /outputs/<VERSION>/logs/<name>.log
#
# Usage (detach so it survives the session):
#   nohup bash scripts/run_gpu_pool.sh phase2_scale_25 phase2_scale_50 ... \
#       > /mnt/outputs/v1.0/logs/pool.log 2>&1 &
#
# Resumable: skips any config whose run dir already has a non-crash run_summary.md
# (a summary with `best_epoch | -1` is a crash artifact → rerun).
# Env: VERSION (default v1.0), NGPU (default 8), ADC_PATH, GCP_PROJECT.
# Requires bash >= 4.3 (wait -n).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2"
ADC="${ADC_PATH:-$HOME/.config/gcloud/application_default_credentials.json}"
PATCH='sed -i "s/LayerId = cv2.dnn.DictValue/LayerId = object/" /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py 2>/dev/null || true'

VERSION="${VERSION:-v1.0}"
NGPU="${NGPU:-8}"
# Shared, persistent HF cache (foundation-encoder weights) — avoids re-downloading into each
# ephemeral --rm container and the anonymous-Hub rate limits under concurrent downloads.
HF_CACHE="${HF_CACHE:-/mnt/outputs/hf_cache}"
mkdir -p "${HF_CACHE}" 2>/dev/null || sudo mkdir -p "${HF_CACHE}"
OUTROOT="/mnt/outputs/${VERSION}"
mkdir -p "${OUTROOT}"/{runs,mlflow,logs} 2>/dev/null || sudo mkdir -p "${OUTROOT}"/{runs,mlflow,logs}

run_one() {  # $1=name  $2=gpu  — one container to completion (blocking; called in background)
  local name="$1" gpu="$2"
  sudo docker rm "${name}" >/dev/null 2>&1 || true
  # --privileged exposes ALL GPUs regardless of `--gpus device=N`, so pin via
  # CUDA_VISIBLE_DEVICES (see run_ablation_queue.sh note, fixed 2026-06-13).
  sudo docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES="${gpu}" --privileged --shm-size=16g \
      --name "${name}" \
      -v "${REPO}:/app" -v /mnt/outputs:/outputs \
      -v "${HF_CACHE}:/root/.cache/huggingface" -e HF_HOME=/root/.cache/huggingface \
      -v "${ADC}:/gcp_adc.json:ro" \
      -e GOOGLE_APPLICATION_CREDENTIALS=/gcp_adc.json \
      -e GOOGLE_CLOUD_PROJECT="${GCP_PROJECT:-pdg-project-406720}" \
      -e MLFLOW_TRACKING_URI="file:///outputs/${VERSION}/mlflow/${name}" \
      -e GDAL_HTTP_MAX_RETRY=3 -e GDAL_HTTP_RETRY_DELAY=1 \
      --entrypoint bash "$IMAGE" \
      -c "set -o pipefail; ${PATCH} && python scripts/train.py --config configs/${name}.yaml \
            --out-dir /outputs/${VERSION}/runs/${name} 2>&1 | tee /outputs/${VERSION}/logs/${name}.log"
}

queue=("$@")
# GPUS (optional): explicit space-separated GPU indices to use (e.g. "0 1 2 3 4 6"
# to skip GPUs already busy with non-pool runs). Falls back to 0..NGPU-1.
if [ -n "${GPUS:-}" ]; then free_gpus=(${GPUS}); else free_gpus=(); for ((g=0; g<NGPU; g++)); do free_gpus+=("$g"); done; fi
declare -A pid_gpu pid_name   # pid -> gpu, pid -> name
i=0

while (( i < ${#queue[@]} )) || (( ${#pid_gpu[@]} > 0 )); do
  # Fill every free GPU with the next pending config.
  while (( ${#free_gpus[@]} > 0 )) && (( i < ${#queue[@]} )); do
    name="${queue[$i]}"; ((i++))
    out="${OUTROOT}/runs/${name}"
    if [ -f "${out}/run_summary.md" ] && ! grep -q 'best_epoch | -1' "${out}/run_summary.md"; then
      echo "[pool] $(date) SKIP ${name} (run_summary.md exists)"; continue
    fi
    gpu="${free_gpus[0]}"; free_gpus=("${free_gpus[@]:1}")
    run_one "${name}" "${gpu}" &
    pid=$!; pid_gpu[$pid]="${gpu}"; pid_name[$pid]="${name}"
    echo "[pool] $(date) START ${name} on GPU ${gpu} (pid ${pid})"
  done

  (( ${#pid_gpu[@]} > 0 )) || continue
  wait -n   # block until ANY background run exits
  # Reclaim every GPU whose run has finished (wait -n doesn't say which).
  for pid in "${!pid_gpu[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      gpu="${pid_gpu[$pid]}"; free_gpus+=("${gpu}")
      echo "[pool] $(date) END ${pid_name[$pid]} on GPU ${gpu}"
      unset 'pid_gpu[$pid]' 'pid_name[$pid]'
    fi
  done
done
echo "[pool] $(date) ALL DONE (${#queue[@]} configs)"
