#!/usr/bin/env bash
# Task-agnostic live job monitor. Lists every running project container — any
# workload: GPU training, GPU inference, or CPU work like region assembly — with
# its uptime, CPU%, memory, and the latest progress line parsed from its OWN log
# (so it needs no per-task knowledge). A compact GPU summary follows, annotated
# with the owning container when one declares CUDA_VISIBLE_DEVICES. Read-only;
# safe to run against live jobs. Supersedes the GPU-only gpu_runs.sh.
#
# Usage:
#   bash scripts/monitor_jobs.sh [NAME_FILTER]      # default: all containers
#   watch -n 15 'bash scripts/monitor_jobs.sh'
set -u
FILTER="${1:-}"

# name<TAB>status for every running container (one docker call); optional name filter
FILTER_ARGS=()
[ -n "$FILTER" ] && FILTER_ARGS=(--filter "name=${FILTER}")
mapfile -t ROWS < <(sudo docker ps "${FILTER_ARGS[@]}" \
  --format '{{.Names}}'$'\t''{{.Status}}' 2>/dev/null | sort)

if [ "${#ROWS[@]}" -eq 0 ]; then
  echo "no running containers${FILTER:+ matching name~${FILTER}}"
else
  names=$(printf '%s\n' "${ROWS[@]}" | cut -f1)
  # one docker stats call for CPU%/MEM of all matching containers
  declare -A CPU MEM
  while IFS='|' read -r n c m; do CPU[$n]="$c"; MEM[$n]="${m%% /*}"; done < <(
    sudo docker stats --no-stream --format '{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}' \
      $names 2>/dev/null)

  printf "%-22s %-10s %-7s %-15s %s\n" NAME UP CPU MEM PROGRESS
  for row in "${ROWS[@]}"; do
    n=$(printf '%s' "$row" | cut -f1)
    up=$(printf '%s' "$row" | cut -f2 | sed 's/^Up //; s/ (.*//')
    # last log line carrying a progress signal; strip the logging prefix.
    prog=$(sudo docker logs --tail 300 "$n" 2>&1 \
      | grep -aiE '[0-9]+/[0-9]+|blocks|tile|shard|epoch|pr_auc|ETA|[0-9]+%|wrote|mosaic|done' \
      | tail -1 \
      | sed -E 's/^[0-9-]+ [0-9:]+[^|]*\| *[A-Za-z]+ *\| *[^|]*\| *//' \
      | cut -c1-58)
    printf "%-22s %-10s %-7s %-15s %s\n" "$n" "${up:-—}" "${CPU[$n]:-—}" "${MEM[$n]:-—}" "${prog:-—}"
  done
fi

# --- GPU summary (only shown if nvidia-smi exists) ---
if command -v nvidia-smi >/dev/null 2>&1; then
  # map GPU index -> owning container via each container's CUDA_VISIBLE_DEVICES
  declare -A OWNER
  for c in $(sudo docker ps --format '{{.Names}}'); do
    for g in $(sudo docker inspect "$c" --format '{{range .Config.Env}}{{println .}}{{end}}' \
                 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | tr ',' ' '); do
      [ -n "$g" ] && OWNER[$g]="$c"
    done
  done
  echo
  printf "%-4s %5s %9s  %s\n" GPU UTIL MEM OWNER
  while IFS=, read -r idx util mem; do
    idx=${idx// /}; util=${util// /}; mem=${mem// /}
    printf "%-4s %4s%% %7sMB  %s\n" "$idx" "$util" "$mem" "${OWNER[$idx]:-—}"
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
             --format=csv,noheader,nounits)
fi
