#!/usr/bin/env bash
# Sequential single-GPU experiment runner (device 0). Runs each config to
# completion, one at a time, then the next. This is the SLOW single-A100 path
# for the experiment program in `training/experiments.md` while the multi-GPU VM
# is provisioned — NOT the parallel orchestrator (deferred until the VM/GPU count
# + MLflow backend are fixed).
#
# Args are config BASENAMES → configs/<name>.yaml, out-dir /outputs/<name>,
# container name <name>, run log /outputs/<name>.log.
#
# Usage (detach so it survives the session):
#   WAIT_FOR=abl_loss_tversky nohup bash scripts/run_ablation_queue.sh \
#       phase2_scale_25 phase2_scale_50 ... > /mnt/outputs/experiment_queue.log 2>&1 &
#
# Resumable: skips any config whose out-dir already has run_summary.md.
# WAIT_FOR (optional env): wait for this container to finish before starting.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2"
ADC="$HOME/.config/gcloud/application_default_credentials.json"
PATCH='sed -i "s/LayerId = cv2.dnn.DictValue/LayerId = object/" /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py 2>/dev/null || true'

WAIT_FOR="${WAIT_FOR:-}"
if [ -n "$WAIT_FOR" ] && sudo docker ps -q --filter name="$WAIT_FOR" | grep -q .; then
  echo "[queue] $(date) waiting for ${WAIT_FOR} (already running) ..."
  sudo docker wait "$WAIT_FOR" || true
fi

for name in "$@"; do
  out="/mnt/outputs/${name}"
  if [ -f "${out}/run_summary.md" ]; then
    echo "[queue] $(date) SKIP ${name} (run_summary.md exists)"; continue
  fi
  echo "[queue] $(date) START ${name}"
  sudo docker rm "${name}" >/dev/null 2>&1 || true
  sudo docker run --rm --gpus '"device=0"' --privileged --shm-size=16g \
      --name "${name}" \
      -v "${REPO}:/app" -v /mnt/outputs:/outputs \
      -v "${ADC}:/gcp_adc.json:ro" \
      -e GOOGLE_APPLICATION_CREDENTIALS=/gcp_adc.json \
      -e GDAL_HTTP_MAX_RETRY=3 -e GDAL_HTTP_RETRY_DELAY=1 \
      --entrypoint bash "$IMAGE" \
      -c "${PATCH} && python scripts/train.py --config configs/${name}.yaml \
            --out-dir /outputs/${name} 2>&1 | tee /outputs/${name}.log"
  echo "[queue] $(date) END ${name} (exit $?)"
done
echo "[queue] $(date) ALL DONE"
