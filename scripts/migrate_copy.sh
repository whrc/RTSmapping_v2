#!/usr/bin/env bash
# Copy one GCS prefix to another, chunked by sub-prefix (computing/pdg_migration.md §4).
#
# One giant `gcloud storage rsync` over 41.7M objects is a single opaque operation
# that cannot be resumed or partially verified. This splits the work at the first
# level below the source prefix and runs N chunks concurrently, so a failure costs
# one chunk and a re-run skips what already landed (`-n`).
#
# Usage:
#   scripts/migrate_copy.sh gs://src/prefix gs://dst/prefix [concurrency]
#
#   scripts/migrate_copy.sh \
#     gs://rts-mapping-v2-usw1/inference/2025q3_south/probs \
#     gs://rts-arctic-usw1/inference/2025q3_south/probs 32
#
# Copies are server-side; this host only orchestrates. Re-runnable: `-n` means an
# object already at the destination is never rewritten, so re-running after any
# failure is cheap and safe.
#
# Verify with: python scripts/gcs_parity.py --src <SRC> --dst <DST>

set -uo pipefail

SRC="${1:?usage: migrate_copy.sh gs://src/prefix gs://dst/prefix [concurrency]}"
DST="${2:?usage: migrate_copy.sh gs://src/prefix gs://dst/prefix [concurrency]}"
JOBS="${3:-32}"

SRC="${SRC%/}"
DST="${DST%/}"

LOGDIR="${MIGRATE_LOGDIR:-/mnt/outputs/migration/logs}"
mkdir -p "$LOGDIR"

echo "Listing sub-prefixes of ${SRC}/ …"
mapfile -t CHUNKS < <(gcloud storage ls "${SRC}/" | grep '/$' || true)

if [[ ${#CHUNKS[@]} -eq 0 ]]; then
    echo "No sub-prefixes — copying ${SRC} in one pass."
    CHUNKS=("${SRC}/")
fi

echo "${#CHUNKS[@]} chunks, ${JOBS}-way concurrency, logs in ${LOGDIR}"

fail=0
run_chunk() {
    local chunk="$1"
    local name; name="$(basename "${chunk%/}")"
    local log="${LOGDIR}/$(basename "${SRC}")__${name}.log"
    if gcloud storage cp -r -n "${chunk}" "${DST}/" >"$log" 2>&1; then
        echo "ok   ${name}"
    else
        echo "FAIL ${name}  (see ${log})"
        return 1
    fi
}
export -f run_chunk
export SRC DST LOGDIR

printf '%s\n' "${CHUNKS[@]}" \
    | xargs -P "$JOBS" -I{} bash -c 'run_chunk "$@"' _ {} \
    || fail=1

if [[ $fail -ne 0 ]]; then
    echo
    echo "At least one chunk failed. Re-run this same command — completed objects are"
    echo "skipped by -n, so a re-run only retries what is missing."
    exit 1
fi

echo
echo "All ${#CHUNKS[@]} chunks copied. Now verify:"
echo "  python scripts/gcs_parity.py --src ${SRC} --dst ${DST}"
