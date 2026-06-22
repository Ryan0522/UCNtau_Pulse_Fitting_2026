#!/bin/bash
#SBATCH --job-name=ucn_sweep
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-0

set -euo pipefail

mkdir -p slurm_logs

: "${BASE:?set BASE=config/default_2021.json}"
: "${SWEEP_NAME:?set SWEEP_NAME, e.g. sweep_nll_epoch2}"
: "${PARAM:?set PARAM, e.g. fit_settings.discovery_delta_nll_cut}"
: "${LABEL:?set LABEL, e.g. nll}"
: "${VALUES:?set VALUES as a quoted space-separated list, e.g. '5 10 15'}"

EXE=${EXE:-./build/run_batch_analysis}
SHARDS=${SHARDS:-1}
PYTHON=${PYTHON:-python3}
SWEEP_SCRIPT=${SWEEP_SCRIPT:-apps/sweep_config.py}

$PYTHON "$SWEEP_SCRIPT" \
  --base "$BASE" \
  --exe "$EXE" \
  --sweep-name "$SWEEP_NAME" \
  --param "$PARAM" \
  --label "$LABEL" \
  --values $VALUES \
  --shards "$SHARDS"