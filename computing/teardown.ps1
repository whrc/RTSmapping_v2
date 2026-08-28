# PDG teardown — run on ARCHITECTURE as rtsmapping@ (or yyang@, which holds owner).
#
# Only run this once `computing/pdg_migration.md` §5 row 1 reads PASS. It did as of
# 2026-08-28: all seven legs compared object-by-object, `missing = 0` everywhere.
#
#   gcloud config set account rtsmapping@woodwellclimate.org
#   ./computing/teardown.ps1
#
# Idempotent: every step tolerates the thing already being gone. Steps 0 and 1 were
# already done on 2026-08-28 and are left in so the script tells the whole story.

$ErrorActionPreference = "Continue"
$PDG = "pdg-project-406720"

# ------------------------------------------------------------------ 0. soft-delete
# MUST come before any bucket delete. All three buckets carried the 7-day default
# (604800 s), which keeps deleted objects BILLABLE for a week — the exact opposite of
# the point of deleting 60 TB. Cleared 2026-08-28; re-asserted here because getting
# this wrong is silent and expensive.
Write-Host "=== 0. soft-delete must be OFF before any bucket delete ==="
foreach ($b in @("rts-mapping-v2", "rts-mapping-v2-usw1", "rts-mapping-v2-usc1")) {
    $r = gcloud storage buckets describe "gs://$b" --format="value(soft_delete_policy.retentionDurationSeconds)" 2>$null
    if ($r -ne "0") {
        Write-Host "  gs://$b retention=$r — clearing"
        gcloud storage buckets update "gs://$b" --clear-soft-delete | Out-Null
    } else {
        Write-Host "  gs://$b retention=0 OK"
    }
}

# ------------------------------------------------------------------ 1. review app
# The Cloud Run service was superseded on 2026-08-04 and never torn down; deleted
# 2026-08-28. The VM is stopped. Reviewers are on http://34.83.225.204/ — verified
# ahead of the old app (31 batches vs 30) before anything here was touched.
Write-Host "`n=== 1. retired review deployments ==="
gcloud run services delete rts-review --project=$PDG --region=us-west1 --quiet 2>$null
gcloud compute instances delete rts-review-vm --project=$PDG --zone=us-west1-a --quiet
# Release the static IP only AFTER its VM is gone, or the delete is refused as in-use.
gcloud compute addresses delete rts-review-vm-ip --project=$PDG --region=us-west1 --quiet

# ------------------------------------------------------------------ 2. the master
# Stopped 2026-08-28 with --discard-local-ssd=false. Its 741 GB boot disk and the
# 21 GB local-SSD t65 build are both verified drained — t65 by CRC32C, file by file,
# because a matching file count is not a matching file (§5 row 9).
Write-Host "`n=== 2. a100-8x-train ==="
gcloud compute instances delete a100-8x-train --project=$PDG --zone=us-central1-a --quiet

# ------------------------------------------------------------------ 3. our images
# rts-infer:v1, rts-review:v1 and rts-train:v2 all exist in
# us-west1-docker.pkg.dev/abruptthawmapping/rts (verified 2026-08-28).
# LEAVE lake_drainage_test — it is not ours.
Write-Host "`n=== 3. our images (NOT lake_drainage_test) ==="
$REG = "us-west1-docker.pkg.dev/$PDG/pdg-artifact-registry"
foreach ($img in @("rts-train", "rts-infer", "rts-review")) {
    Write-Host "  deleting $img"
    gcloud artifacts docker images delete "$REG/$img" --project=$PDG --delete-tags --quiet 2>$null
}

# ------------------------------------------------------------------ 4. the buckets
# Irreversible, and the point of the exercise. 60 TB.
Write-Host "`n=== 4. buckets ==="
foreach ($b in @("rts-mapping-v2", "rts-mapping-v2-usw1", "rts-mapping-v2-usc1")) {
    Write-Host "  deleting gs://$b"
    gcloud storage rm --recursive "gs://$b" --quiet
}

# ------------------------------------------------------------------ 5. NOT ours
# pdg-planet-data is PDG's bucket. Its funding lapses with the project regardless,
# and deleting our global_quarterly/ prefix would mean ~5 M destructive API calls in
# someone else's project for no saving of ours. Hand it back to Luigi/Todd.
Write-Host "`n=== 5. pdg-planet-data: NOT deleted, handed back ==="

# ------------------------------------------------------------------ 6. verify
Write-Host "`n=== 6. verification — nothing rts-* should remain ==="
Write-Host "-- instances --";  gcloud compute instances list --project=$PDG --format="table(name,status)"
Write-Host "-- disks --";      gcloud compute disks list     --project=$PDG --format="table(name,sizeGb,users.basename())"
Write-Host "-- addresses --";  gcloud compute addresses list --project=$PDG --format="table(name,address,status)"
Write-Host "-- buckets --";    gcloud storage ls --project=$PDG
Write-Host "-- images --";     gcloud artifacts docker images list $REG --project=$PDG --format="value(package)"
Write-Host "-- cloud run --";  gcloud run services list --project=$PDG --format="table(metadata.name)"
Write-Host "`nExpected to remain: download-vm and gke-water-cluster-* (not ours),"
Write-Host "pdg-planet-data (handed back), lake_drainage_test (not ours)."
