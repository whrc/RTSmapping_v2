#!/usr/bin/env bash
# Sequential single-GPU ablation runner (device 0).
#
# Runs each config in the queue to completion, one at a time, then the next.
# This is the SLOW single-A100 path for keeping the GPU busy on transfer-safe
# ablations while the multi-GPU VM is provisioned — NOT the parallel orchestrator
# (that's deferred until the VM/GPU count + MLflow backend are fixed).
#
# Usage (detach so it survives the session):
#   nohup bash scripts/run_ablation_queue.sh loss_tversky encoder_b7 ... \
#         > /mnt/outputs/ablation_queue.log 2>&1 &
#
# Resumable: skips any config whose out-dir already has run_summary.md.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2"
ADC="$HOME/.config/gcloud/application_default_credentials.json"
PATCH='sed -i "s/LayerId = cv2.dnn.DictValue/LayerId = object/" /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py 2>/dev/null || true'

# If a baseline/first ablation is still running on device 0, wait for it.
if sudo docker ps -q --filter name=abl_loss_compound | grep -q .; then
  echo "[queue] $(date) waiting for abl_loss_compound (already running) ..."
  sudo docker wait abl_loss_compound || true
fi

for name in "$@"; do
  out="/mnt/outputs/abl_${name}"
  if [ -f "${out}/run_summary.md" ]; then
    echo "[queue] $(date) SKIP ${name} (run_summary.md exists)"; continue
  fi
  echo "[queue] $(date) START ${name}"
  sudo docker rm "abl_${name}" >/dev/null 2>&1 || true
  sudo docker run --rm --gpus '"device=0"' --privileged --shm-size=16g \
      --name "abl_${name}" \
      -v "${REPO}:/app" -v /mnt/outputs:/outputs \
      -v "${ADC}:/gcp_adc.json:ro" \
      -e GOOGLE_APPLICATION_CREDENTIALS=/gcp_adc.json \
      -e GDAL_HTTP_MAX_RETRY=3 -e GDAL_HTTP_RETRY_DELAY=1 \
      --entrypoint bash "$IMAGE" \
      -c "${PATCH} && python scripts/train.py --config configs/abl_${name}.yaml \
            --out-dir /outputs/abl_${name} 2>&1 | tee /outputs/abl_${name}.log"
  echo "[queue] $(date) END ${name} (exit $?)"
done
echo "[queue] $(date) ALL DONE"
