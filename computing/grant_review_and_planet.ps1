# The remaining IAM for the 2026-08-31 cutover. Run on ARCHITECTURE as rtsmapping@.
# Idempotent - safe to re-run. Everything here is blocked from automation because it
# creates identities or grants privilege; the rest of the migration needs none of it.
#
#   gcloud config set account rtsmapping@woodwellclimate.org
#   gcloud config set project abruptthawmapping
#   ./computing/grant_review_and_planet.ps1

$ErrorActionPreference = "Stop"
$PROJECT = "abruptthawmapping"

# ---------------------------------------------------------------- 1. Planet delivery
# planet-orders@ already holds storage.objectUser on pdg-planet-data and has a
# user-managed key Heidi is ordering with right now. It just has no standing on the new
# bucket. Grant this and her PDG_PL_ORDERS_KEY keeps working unchanged - no new key,
# and nothing needed from Planet. (Missed on the first grant_sts_access.ps1 run.)
Write-Host "=== Planet delivery ==="
gcloud storage buckets add-iam-policy-binding gs://rts-arctic-usw1 `
    --member="serviceAccount:planet-orders@$PROJECT.iam.gserviceaccount.com" `
    --role=roles/storage.objectUser --quiet | Out-Null
Write-Host "  planet-orders@ -> objectUser on rts-arctic-usw1"

# ---------------------------------------------------------------- 2. Review app identity
# scripts/deploy_review_vm.sh derives SA="rts-review-app@$PROJECT...", so the new project
# needs its own. Roles mirror exactly what the PDG one holds on rts-mapping-v2-usw1
# (verified 2026-08-28): objectViewer + objectCreator, nothing at project level.
# objectCreator not objectUser: the app writes verdicts and must never delete a crop.
Write-Host "=== Review app identity ==="
$sa = "rts-review-app@$PROJECT.iam.gserviceaccount.com"
$exists = gcloud iam service-accounts list --project=$PROJECT --format="value(email)" | Where-Object { $_ -eq $sa }
if (-not $exists) {
    gcloud iam service-accounts create rts-review-app --project=$PROJECT `
        --display-name="RTS review campaign app" --quiet | Out-Null
    Write-Host "  created $sa"
} else {
    Write-Host "  $sa already exists"
}
foreach ($role in @("roles/storage.objectViewer", "roles/storage.objectCreator")) {
    gcloud storage buckets add-iam-policy-binding gs://rts-arctic-usw1 `
        --member="serviceAccount:$sa" --role=$role --quiet | Out-Null
    Write-Host "  $sa -> $($role.Replace('roles/storage.','')) on rts-arctic-usw1"
}

# ---------------------------------------------------------------- verify
Write-Host "`n--- verification ---"
$policy = gcloud storage buckets get-iam-policy gs://rts-arctic-usw1 --format=json | ConvertFrom-Json
foreach ($who in @("planet-orders", "rts-review-app")) {
    $roles = ($policy.bindings | Where-Object { $_.members -match $who } | ForEach-Object { $_.role.Replace('roles/storage.','') }) -join ", "
    if ($roles) { Write-Host ("  OK      {0,-16} {1}" -f $who, $roles) }
    else        { Write-Host ("  MISSING {0}" -f $who) -ForegroundColor Red }
}
