#!/usr/bin/env bash
# Deploy the review app to Cloud Run behind Identity-Aware Proxy.
#
# Why this shape (post-inference/review_campaign.md §10):
#   * No service-account key exists anywhere, and no signing permission is
#     needed either: the app streams crops out of the bucket itself, so the
#     runtime SA needs only objectViewer + objectCreator on the ONE bucket.
#   * IAP is enabled directly on the service (no load balancer), so reviewers
#     sign in with the Google accounts they already have and the app receives a
#     verified identity rather than a typed-in name.
#
# Idempotent: re-running only updates the revision.
#
# WHAT THIS ACCOUNT CANNOT DO (checked 2026-08-03 for yyang@woodwellclimate.org):
# the two IAP-related bindings below are IAM policy writes, and this project's
# operators hold neither `run.services.setIamPolicy` nor
# `iap.webServiceVersions.setIamPolicy`. The script therefore treats them as
# best-effort: everything else still deploys, and any binding that is refused is
# reprinted at the end as a verbatim command for whoever holds project admin.
# (The OAuth consent screen/brand, flagged earlier as a possible blocker, already
# exists in this project — no action needed there.)
#
# Usage:
#   scripts/deploy_cloud_run.sh
#   scripts/deploy_cloud_run.sh reviewer1@woodwellclimate.org reviewer2@...
set -euo pipefail

PROJECT="${REVIEW_PROJECT:-abruptthawmapping}"
REGION="${REVIEW_REGION:-us-west1}"          # same region as the bucket
SERVICE="${REVIEW_SERVICE:-rts-review}"
BUCKET="${REVIEW_BUCKET:-rts-arctic-usw1}"
PREFIX="${REVIEW_PREFIX:-inference/2025q3_south/review}"
CROPS="${REVIEW_CROP_PREFIX:-inference/2025q3_south/internal/review_crops}"
MANIFEST="${REVIEW_MANIFEST:-gs://$BUCKET/$PREFIX/manifest.parquet}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/rts/rts-review:v1"
SA="rts-review-app@$PROJECT.iam.gserviceaccount.com"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NEEDS_ADMIN=()

# Run one IAM binding, and if it is refused, remember the command instead of
# aborting: a partial deploy that names what is missing beats no deploy at all.
try_binding() {
  local what="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "  granted: $what"
  else
    echo "  DENIED (needs project admin): $what"
    NEEDS_ADMIN+=("$*")
  fi
}

echo "== runtime service account =="
gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud iam service-accounts create rts-review-app --project "$PROJECT" \
      --display-name "RTS review app"

# Read crops + manifest, write claims/verdicts. Bucket-scoped, and the only
# permission the app itself needs — crops are streamed, never signed.
for ROLE in roles/storage.objectViewer roles/storage.objectCreator; do
  gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
      --member "serviceAccount:$SA" --role "$ROLE" --quiet >/dev/null
done

echo "== build + push image =="
# Built here and pushed directly, NOT through Cloud Build: Cloud Build runs as
# the compute default SA, which lacks `artifactregistry.repositories.upload
# Artifacts` on pdg-artifact-registry, and granting it needs an IAM policy write
# we do not hold (§10.3). This account can push, so it does. Bonus: the image
# deployed is byte-for-byte the one tested locally.
DOCKER="docker"
if ! $DOCKER info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
  else
    echo "docker is not usable (tried plain and sudo -n)" >&2
    exit 1
  fi
fi
# Log in through the token rather than `gcloud auth configure-docker`, which
# writes to ~/.docker and would be invisible to a sudo'd daemon client.
gcloud auth print-access-token \
  | $DOCKER login -u oauth2accesstoken --password-stdin "https://$REGION-docker.pkg.dev" >/dev/null
$DOCKER build -f "$REPO/computing/Dockerfile.review" -t "$IMAGE" "$REPO"
$DOCKER push "$IMAGE"

echo "== deploy behind IAP =="
gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" \
    --image "$IMAGE" --service-account "$SA" \
    --no-allow-unauthenticated --iap \
    --memory 1Gi --max-instances 3 \
    --set-env-vars "REVIEW_BUCKET=$BUCKET,REVIEW_PREFIX=$PREFIX,REVIEW_CROP_PREFIX=$CROPS,REVIEW_MANIFEST=$MANIFEST"

echo "== access bindings =="
# IAP fronts the service, so IAP itself must be allowed to invoke it.
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
try_binding "IAP service agent -> run.invoker" \
  gcloud run services add-iam-policy-binding "$SERVICE" \
    --project "$PROJECT" --region "$REGION" \
    --member "serviceAccount:service-$PROJECT_NUMBER@gcp-sa-iap.iam.gserviceaccount.com" \
    --role roles/run.invoker --quiet

for EMAIL in "$@"; do
  try_binding "$EMAIL -> iap.httpsResourceAccessor" \
    gcloud iap web add-iam-policy-binding \
      --member "user:$EMAIL" --role roles/iap.httpsResourceAccessor \
      --region "$REGION" --resource-type cloud-run --service "$SERVICE" \
      --project "$PROJECT" --quiet
done

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" \
        --region "$REGION" --format='value(status.url)')"
echo
echo "Deployed: $URL"

if [ ${#NEEDS_ADMIN[@]} -gt 0 ]; then
  cat <<'MSG'

NOT REACHABLE YET. The bindings below were refused; until someone with project
admin runs them, IAP has no route to the service and no reviewer can sign in.
Send them these commands verbatim, or ask for roles/run.admin + roles/iap.admin
on this project and re-run this script yourself.
MSG
  printf '\n'
  for CMD in "${NEEDS_ADMIN[@]}"; do echo "  $CMD"; done
  printf '\n'
else
  echo "Check:    open it in a browser; /api/progress must report 301 batches."
fi
echo "More reviewers later: scripts/deploy_cloud_run.sh <email> ..."
