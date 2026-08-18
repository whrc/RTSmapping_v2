#!/usr/bin/env bash
# Run one year's acquisition end to end, supervised, inside tmux.
#
# Mirrors scripts/launch_south_inference.sh: a non-zero exit is restarted, a
# clean exit retires the year, repeated fast failures stop rather than spin, and
# a STOP sentinel ends it deliberately.
#
# The keys are read once into this shell's environment and never touch disk
# (Option A of the plan). Every restart inherits them, so a supervised restart
# needs nobody at the keyboard -- but a VM REBOOT loses them and you will need
# to start this again.
#
#   tmux new -s planet
#   ./planetscope-download/run_year.sh 2022
#   ./planetscope-download/run_year.sh 2022 --workers 8 --limit 200   # threading test
#   Ctrl-b d                                   # detach; it keeps running
#   tmux attach -t planet                      # come back any time
#   touch planetscope-download/status/STOP     # stop after the current step
#
set -uo pipefail

YEAR="${1:?usage: run_year.sh <year> [extra args passed to order_basemaps.py]}"; shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

# Runtime outputs go OUTSIDE the repo. The checkout belongs to whoever cloned it
# and collaborators sign in as their own OS Login user with no write access to
# it, so writing here would fail for everyone but the owner. /mnt/outputs is
# world-writable on the VM.
PSD_WORK="${PSD_WORK:-/mnt/outputs/planetscope-download}"
DATA="$PSD_WORK/data"; STATUS="$PSD_WORK/status"; LOGS="$PSD_WORK/logs"
STOP_FILE="$STATUS/STOP"
GRIDS="$DATA/circumpolar_basemap_grids_${YEAR}.geojson"
SOUTH="$DATA/circumpolar_south_planet_basemap_grids_${YEAR}.geojson"

RESTART_DELAY_S="${RESTART_DELAY_S:-30}"
MIN_HEALTHY_S="${MIN_HEALTHY_S:-300}"   # a non-zero exit sooner than this is a fast failure
MAX_FAST_FAILS="${MAX_FAST_FAILS:-5}"

# The acquisition deps (geopandas, google-cloud-storage, ...) are not on the
# system python. Prefer the shared venv; PSD_PYTHON overrides.
PYTHON="${PSD_PYTHON:-/mnt/outputs/planetscope-venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON="python3"
export PYTHONDONTWRITEBYTECODE=1   # the repo's __pycache__ is not ours to write

# --- preflight: fail BEFORE asking for keys ----------------------------------
# Typing a Planet key and only then discovering a missing directory or import is
# a bad trade for everyone involved.
preflight_fail=0
# umask 0 so a run started by one OS Login user leaves directories the next one
# can write: this is shared scratch under a 777 /mnt/outputs, and the alternative
# is the second person silently failing to log.
if ! (umask 000 && mkdir -p "$DATA" "$STATUS" "$LOGS") 2>/dev/null; then
  echo "[run] ERROR: cannot create $PSD_WORK"
  echo "[run]        set PSD_WORK to a directory you can write, e.g."
  echo "[run]        PSD_WORK=\$HOME/planetscope-download ./planetscope-download/run_year.sh $YEAR"
  preflight_fail=1
fi
if ! "$PYTHON" -c 'import geopandas, shapely, pandas, requests; from google.cloud import storage' 2>/dev/null; then
  echo "[run] ERROR: '$PYTHON' is missing the acquisition dependencies."
  echo "[run]        Expected the shared venv at /mnt/outputs/planetscope-venv."
  echo "[run]        Rebuild it with:"
  echo "[run]          python3 -m venv /mnt/outputs/planetscope-venv"
  echo "[run]          /mnt/outputs/planetscope-venv/bin/pip install -r $HERE/requirements.txt"
  echo "[run]        or point PSD_PYTHON at an interpreter that has them."
  preflight_fail=1
fi
if [ ! -r "$REPO/domain/circumpolar_south_domain.geojson" ]; then
  echo "[run] ERROR: cannot read $REPO/domain/circumpolar_south_domain.geojson"
  preflight_fail=1
fi
[ "$preflight_fail" -eq 0 ] || { echo "[run] preflight failed — nothing was started, no keys were asked for."; exit 2; }

chmod 777 "$PSD_WORK" "$DATA" "$STATUS" "$LOGS" 2>/dev/null || true   # best effort; only the creator can
echo "[run] preflight OK — python=$PYTHON  work=$PSD_WORK"
rm -f "$STOP_FILE"

# --- keys: prompted once, exported, never written to disk --------------------
if [ -z "${PL_BM_API_KEY:-}" ]; then
  read -rsp "Planet API key (PL_BM_API_KEY): " PL_BM_API_KEY; echo; export PL_BM_API_KEY
fi
if [ -z "${PDG_PL_ORDERS_KEY:-}" ]; then
  read -rsp "GCS delivery credential (PDG_PL_ORDERS_KEY): " PDG_PL_ORDERS_KEY; echo
  export PDG_PL_ORDERS_KEY
fi
[ -n "$PL_BM_API_KEY" ] && [ -n "$PDG_PL_ORDERS_KEY" ] || { echo "[run] both keys are required"; exit 2; }

cd "$REPO"

echo "[run] $(date -Is) year=$YEAR  repo=$REPO"

# --- steps 1 and 2 are quick and idempotent; skip if already done ------------
if [ ! -s "$GRIDS" ]; then
  echo "[run] $(date -Is) step 1: searching Planet for ${YEAR}q3 quads"
  "$PYTHON" planetscope-download/search_basemap_grids.py --year "$YEAR" --output "$GRIDS" \
    2>&1 | tee -a "$LOGS/step1_${YEAR}.log" || { echo "[run] step 1 failed"; exit 1; }
else
  echo "[run] $(date -Is) step 1: $GRIDS exists — skipping"
fi

if [ ! -s "$SOUTH" ]; then
  echo "[run] $(date -Is) step 2: clipping to the circumpolar-south domain"
  "$PYTHON" planetscope-download/filter_to_domain.py --year "$YEAR" \
    --grids "$GRIDS" --output "$SOUTH" \
    2>&1 | tee -a "$LOGS/step2_${YEAR}.log" || { echo "[run] step 2 failed"; exit 1; }
else
  echo "[run] $(date -Is) step 2: $SOUTH exists — skipping"
fi

# --- step 3: the long one, supervised ---------------------------------------
n=0; fast=0
while true; do
  if [ -f "$STOP_FILE" ]; then
    echo "[run] $(date -Is) STOP present — not (re)starting. rm $STOP_FILE to resume."; break
  fi
  n=$((n+1))
  echo "[run] $(date -Is) step 3: ordering, attempt #$n"
  t0=$(date +%s)
  "$PYTHON" planetscope-download/order_basemaps.py \
      --year "$YEAR" --grids "$SOUTH" --status-dir "$STATUS" "$@" \
      >> "$LOGS/orders_${YEAR}.log" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))

  if [ "$rc" -eq 0 ]; then
    echo "[run] $(date -Is) ordering complete for $YEAR after ${dt}s"; break
  fi
  if [ "$rc" -eq 2 ]; then
    echo "[run] $(date -Is) FATAL rc=2 (bad or expired credentials) — see $LOGS/orders_${YEAR}.log"
    echo "[run] restart this script with a fresh key; delivered quads are skipped on resume."; break
  fi
  if [ "$rc" -eq 1 ]; then
    echo "[run] $(date -Is) finished with failed quads — sweeping them up once"
    "$PYTHON" planetscope-download/order_basemaps.py --year "$YEAR" --grids "$SOUTH" \
        --status-dir "$STATUS" --retry-failed "$STATUS/failed_orders_${YEAR}.csv" \
        >> "$LOGS/orders_${YEAR}.log" 2>&1
    echo "[run] $(date -Is) sweep-up exited $?"; break
  fi

  # rc=3 is the stall watchdog; anything else is a crash. Both are restartable.
  if [ "$dt" -lt "$MIN_HEALTHY_S" ]; then
    fast=$((fast+1))
    echo "[run] $(date -Is) exited $rc after only ${dt}s (fast failure ${fast}/${MAX_FAST_FAILS})"
    if [ "$fast" -ge "$MAX_FAST_FAILS" ]; then
      echo "[run] $(date -Is) FATAL: ${MAX_FAST_FAILS} consecutive fast failures — stopping. See $LOGS/orders_${YEAR}.log"; break
    fi
  else
    fast=0
    echo "[run] $(date -Is) exited $rc after ${dt}s (stall watchdog or crash) — restarting"
  fi
  sleep "$RESTART_DELAY_S"
done

echo "[run] $(date -Is) done. Status: $STATUS/${YEAR}.json"
"$PYTHON" planetscope-download/check_status.py --status-dir "$STATUS" || true
