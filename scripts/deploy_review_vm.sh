#!/usr/bin/env bash
# Deploy the review app to a GCE VM, reachable over plain HTTP with no sign-in.
#
# Why a VM rather than the Cloud Run service (post-inference/review_campaign.md §10.3):
# opening Cloud Run to reviewers — or to anyone at all, `--allow-unauthenticated`
# included — is an IAM policy write (`run.services.setIamPolicy`) this project's
# operators do not hold. A VM's front door is a firewall rule instead, which they
# do hold, so this is the only open path that needs no project admin.
#
# What that costs, deliberately accepted by the user on 2026-08-04:
#   * NO AUTHENTICATION — anyone who reaches the IP can rate, and can read the
#     crops. Attribution falls back to the name a reviewer types.
#   * NO TLS — a bare IP cannot hold a real certificate. Traffic is clear text.
#   * ~$13/mo always-on, and it is yours to patch and restart.
#
# The image travels through the campaign bucket rather than Artifact Registry:
# the runtime SA already reads that bucket, whereas granting it registry access
# would be another blocked IAM write.
#
# Idempotent: re-running rebuilds, re-uploads and recreates the container.
#
# Usage: scripts/deploy_review_vm.sh
set -euo pipefail

PROJECT="${REVIEW_PROJECT:-abruptthawmapping}"
REGION="${REVIEW_REGION:-us-west1}"           # same region as the bucket
ZONE="${REVIEW_ZONE:-us-west1-a}"
VM="${REVIEW_VM:-rts-review}"
TAG="rts-review"                              # firewall target tag
BUCKET="${REVIEW_BUCKET:-rts-arctic-usw1}"
PREFIX="${REVIEW_PREFIX:-inference/2025q3_south/review}"
CROPS="${REVIEW_CROP_PREFIX:-inference/2025q3_south/internal/review_crops}"
MANIFEST="${REVIEW_MANIFEST:-gs://$BUCKET/$PREFIX/manifest.parquet}"
IMAGE="rts-review:vm"
TARBALL="inference/2025q3_south/internal/deploy/rts-review.tar.gz"
SA="rts-review-app@$PROJECT.iam.gserviceaccount.com"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOCKER="docker"
if ! $DOCKER info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
  else
    echo "docker is not usable (tried plain and sudo -n)" >&2
    exit 1
  fi
fi

echo "== build + ship the image through the bucket =="
$DOCKER build -f "$REPO/computing/Dockerfile.review" -t "$IMAGE" "$REPO"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
$DOCKER save "$IMAGE" | gzip -1 > "$TMP/rts-review.tar.gz"
gcloud storage cp "$TMP/rts-review.tar.gz" "gs://$BUCKET/$TARBALL"

echo "== firewall: open $TAG:80 to the internet =="
gcloud compute firewall-rules describe "rts-review-allow-http" --project "$PROJECT" \
    >/dev/null 2>&1 || \
  gcloud compute firewall-rules create "rts-review-allow-http" --project "$PROJECT" \
      --network default --direction INGRESS --action allow --rules tcp:80 \
      --source-ranges 0.0.0.0/0 --target-tags "$TAG" \
      --description "Open HTTP for the RTS review campaign app (no auth, by design)"

echo "== static IP =="
gcloud compute addresses describe "$VM-ip" --project "$PROJECT" --region "$REGION" \
    >/dev/null 2>&1 || \
  gcloud compute addresses create "$VM-ip" --project "$PROJECT" --region "$REGION"
IP="$(gcloud compute addresses describe "$VM-ip" --project "$PROJECT" \
        --region "$REGION" --format='value(address)')"

# Runs on every boot, so a reboot restores service without anyone logging in.
# The image is fetched with the metadata token rather than gcloud/gsutil, which
# are not guaranteed to be on the base image.
STARTUP="$TMP/startup.sh"
cat > "$STARTUP" <<EOF
#!/bin/bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y docker.io
OBJ=\$(python3 -c "import urllib.parse;print(urllib.parse.quote('$TARBALL', safe=''))")
TOKEN=\$(curl -s -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl -sf -H "Authorization: Bearer \$TOKEN" -o /var/tmp/rts-review.tar.gz \
  "https://storage.googleapis.com/storage/v1/b/$BUCKET/o/\$OBJ?alt=media"
docker load -i /var/tmp/rts-review.tar.gz
docker rm -f rts-review >/dev/null 2>&1 || true
docker run -d --restart always --name rts-review -p 80:8080 \
  -e PORT=8080 \
  -e REVIEW_BUCKET=$BUCKET \
  -e REVIEW_PREFIX=$PREFIX \
  -e REVIEW_CROP_PREFIX=$CROPS \
  -e REVIEW_MANIFEST=$MANIFEST \
  $IMAGE
EOF

echo "== VM =="
if gcloud compute instances describe "$VM" --project "$PROJECT" --zone "$ZONE" \
     >/dev/null 2>&1; then
  echo "  $VM exists — refreshing the container in place"
  gcloud compute instances add-metadata "$VM" --project "$PROJECT" --zone "$ZONE" \
      --metadata-from-file "startup-script=$STARTUP" >/dev/null
  gcloud compute instances reset "$VM" --project "$PROJECT" --zone "$ZONE" >/dev/null
else
  gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" \
      --machine-type e2-small \
      --image-family debian-12 --image-project debian-cloud \
      --boot-disk-size 20GB --boot-disk-type pd-balanced \
      --service-account "$SA" --scopes https://www.googleapis.com/auth/cloud-platform \
      --tags "$TAG" --address "$IP" \
      --metadata-from-file "startup-script=$STARTUP"
fi

echo
echo "Booting; the first start installs docker and loads the image (~2 min)."
for _ in $(seq 1 40); do
  CODE="$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://$IP/api/progress" || true)"
  if [ "$CODE" = "200" ]; then
    echo "Ready: http://$IP/"
    curl -s "http://$IP/api/progress"
    echo
    exit 0
  fi
  sleep 15
done

echo "Not serving yet. Check the boot log with:"
echo "  gcloud compute instances get-serial-port-output $VM --zone $ZONE --project $PROJECT | tail -40"
exit 1
