# Grant the Storage Transfer Service agent what it needs to copy PDG -> abruptthawmapping.
#
# Why STS rather than the `gcloud storage rsync` in pdg_migration.md §4 "Copy method":
# the deadline moved to 2026-08-31 and rts-ops is a 2-vCPU box. STS is Google-managed,
# massively parallel, resumable, and does not depend on a small VM staying alive for
# ~47 M object operations. The runbook assumed we held no setIamPolicy in the org-managed
# PDG project — true at PROJECT level, false at BUCKET level, which is what STS needs.
#
# Run on ARCHITECTURE as rtsmapping@ (verified 2026-08-28 to hold storage.buckets.setIamPolicy
# on all four PDG source buckets). Idempotent - safe to re-run.
#
#   gcloud config set account rtsmapping@woodwellclimate.org
#   gcloud config set project abruptthawmapping
#   ./computing/grant_sts_access.ps1

$ErrorActionPreference = "Stop"

$STS = "serviceAccount:project-801926669176@storage-transfer-service.iam.gserviceaccount.com"

# --- sources: read + list (PDG, org-managed, retiring 2026-08-31) ---
$sources = @("pdg-planet-data", "rts-mapping-v2", "rts-mapping-v2-usw1", "rts-mapping-v2-usc1")
foreach ($b in $sources) {
    Write-Host "source  gs://$b"
    gcloud storage buckets add-iam-policy-binding "gs://$b" --member=$STS --role=roles/storage.objectViewer      --quiet | Out-Null
    gcloud storage buckets add-iam-policy-binding "gs://$b" --member=$STS --role=roles/storage.legacyBucketReader --quiet | Out-Null
}

# --- sinks: write (ours) ---
$sinks = @("rts-arctic-us", "rts-arctic-usw1", "rts-arctic-usc1")
foreach ($b in $sinks) {
    Write-Host "sink    gs://$b"
    gcloud storage buckets add-iam-policy-binding "gs://$b" --member=$STS --role=roles/storage.objectUser         --quiet | Out-Null
    gcloud storage buckets add-iam-policy-binding "gs://$b" --member=$STS --role=roles/storage.legacyBucketWriter --quiet | Out-Null
}

# --- Planet delivery: the SAME service account Heidi already orders with ---
# planet-orders@ already holds storage.objectUser on pdg-planet-data and has a
# user-managed key from 2023 that is in use right now. It simply has no standing on
# the new bucket. Grant that, and the PDG_PL_ORDERS_KEY Heidi already types keeps
# working - no new key, and nothing needed from Planet.
$PLANET = "serviceAccount:planet-orders@abruptthawmapping.iam.gserviceaccount.com"
Write-Host "planet  gs://rts-arctic-usw1"
gcloud storage buckets add-iam-policy-binding gs://rts-arctic-usw1 --member=$PLANET --role=roles/storage.objectUser --quiet | Out-Null

# --- verify: every bucket must list the agent, or the transfer fails at run time, not create time ---
Write-Host "`n--- verification ---"
foreach ($b in ($sources + $sinks)) {
    $n = (gcloud storage buckets get-iam-policy "gs://$b" --format=json |
          ConvertFrom-Json).bindings |
          Where-Object { $_.members -contains $STS.Substring(15) -or $_.members -contains $STS }
    $roles = ($n | ForEach-Object { $_.role }) -join ", "
    if ($roles) { Write-Host ("  OK      {0,-22} {1}" -f $b, $roles) }
    else        { Write-Host ("  MISSING {0}" -f $b) -ForegroundColor Red }
}
