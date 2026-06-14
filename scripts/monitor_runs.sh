#!/usr/bin/env bash
# Live ablation-progress dashboard. For each run under <VERSION>/runs, shows the
# current epoch, latest val PR-AUC, best_smoothed (from run_summary.md once done),
# Δ vs the locked baseline μ₀, and the gate verdict. Reads logs + run_summary.md
# only (no MLflow) — safe to run anytime, including against live runs.
#
# Usage:
#   bash scripts/monitor_runs.sh [VERSION] [name-glob]
#   watch -n 60 'bash scripts/monitor_runs.sh v1.0 "phase*"'
set -u
VERSION="${1:-v1.0}"
GLOB="${2:-*}"
LOGS="/mnt/outputs/${VERSION}/logs"
RUNS="/mnt/outputs/${VERSION}/runs"
MU0=0.791174; BAR=0.802349   # locked baseline μ₀ and winner bar μ₀+G (docs/phase0_baseline.md)

printf "%-30s %6s %10s %12s %8s %-6s %s\n" RUN EPOCH VAL_PRAUC BEST_SMTH "Δvsμ0" GATE AGE
for d in "${RUNS}"/${GLOB}/; do
  [ -d "$d" ] || continue
  name=$(basename "$d"); log="${LOGS}/${name}.log"; sm="${d}run_summary.md"
  ep=$(grep -hoE 'epoch=[0-9]+ train_loss' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
  val=$(grep -hoE 'pr_auc_geomean=[0-9.]+' "$log" 2>/dev/null | tail -1 | cut -d= -f2)
  best=$(grep -E 'best_smoothed' "$sm" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)
  if [ -n "$best" ]; then
    delta=$(awk -v b="$best" -v m="$MU0" 'BEGIN{printf "%+.4f", b-m}')
    gate=$(awk -v b="$best" -v bar="$BAR" 'BEGIN{print (b>=bar)?"WIN":"below"}')
  else delta="—"; gate="run"; fi
  age="—"; [ -f "$log" ] && age="$(( ($(date +%s) - $(stat -c %Y "$log")) / 60 ))m"
  printf "%-30s %6s %10s %12s %8s %-6s %s\n" "$name" "${ep:-—}" "${val:-—}" "${best:-—}" "$delta" "$gate" "$age"
done
echo "baseline μ₀=${MU0}  winner bar=${BAR} (μ₀+G, G=0.0112)"
