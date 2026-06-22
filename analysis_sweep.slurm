#!/bin/bash
#SBATCH --job-name=ucn_analyze_sweep
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

mkdir -p slurm_logs

: "${SWEEP_ROOT:?set SWEEP_ROOT, e.g. output/2021/sweep_nll_epoch2}"

PYTHON=${PYTHON:-python3}
PLOT_SCRIPT=${PLOT_SCRIPT:-apps/plot_observables.py}
RDE_CSV=${RDE_CSV:-analysis/runinfo_2021_all.csv}
LIFETIME_MIN=${LIFETIME_MIN:-0}
LIFETIME_MAX=${LIFETIME_MAX:-21}
LIFETIME_STEP=${LIFETIME_STEP:-1}
DEADTIME_K0=${DEADTIME_K0:-1e-6}
DEADTIME_WINDOW_S=${DEADTIME_WINDOW_S:-60}

for value_dir in "$SWEEP_ROOT"/*; do
  [ -d "$value_dir" ] || continue
  echo "[analyze] $value_dir"

  $PYTHON "$PLOT_SCRIPT" \
    --dir "$value_dir" \
    --out "$value_dir/plots" \
    --rde-csv "$RDE_CSV" \
    --lifetime-threshold-min "$LIFETIME_MIN" \
    --lifetime-threshold-max "$LIFETIME_MAX" \
    --lifetime-threshold-step "$LIFETIME_STEP" \
    --deadtime-k0 "$DEADTIME_K0" \
    --deadtime-window-s "$DEADTIME_WINDOW_S"
done
