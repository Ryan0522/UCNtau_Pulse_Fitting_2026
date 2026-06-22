#!/bin/bash
# Edit only this top block for a new sweep.

set -euo pipefail

BASE=${BASE:-config/default_2021.json}
EXE=${EXE:-./build/run_batch_analysis}
SWEEP_NAME=${SWEEP_NAME:-sweep_nll_delta}
PARAM=${PARAM:-fit_settings.discovery_delta_nll_cut}
LABEL=${LABEL:-nll}
VALUES=${VALUES:-"5 10 15 20 25 30"}
SHARDS=${SHARDS:-16}
RDE_CSV=${RDE_CSV:-analysis/runinfo_2021_all.csv}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-output/2021}

N_VALUES=$(wc -w <<< "$VALUES")
N_TASKS=$((N_VALUES * SHARDS))
ARRAY_MAX=$((N_TASKS - 1))
SWEEP_ROOT="$OUTPUT_FOLDER/$SWEEP_NAME"

mkdir -p slurm_logs

echo "Submitting sweep:"
echo "  BASE        = $BASE"
echo "  EXE         = $EXE"
echo "  SWEEP_NAME  = $SWEEP_NAME"
echo "  PARAM       = $PARAM"
echo "  LABEL       = $LABEL"
echo "  VALUES      = $VALUES"
echo "  SHARDS      = $SHARDS"
echo "  ARRAY       = 0-$ARRAY_MAX"
echo "  SWEEP_ROOT  = $SWEEP_ROOT"

run_jobid=$(sbatch --parsable \
  --account=chenyliu \
  --partition=secondary \
  --array=0-${ARRAY_MAX} \
  --export=ALL,BASE="$BASE",EXE="$EXE",SWEEP_NAME="$SWEEP_NAME",PARAM="$PARAM",LABEL="$LABEL",VALUES="$VALUES",SHARDS="$SHARDS" \
  slurm/sweep_run.sh)

echo "Submitted sweep job: $run_jobid"

an_jobid=$(sbatch --parsable \
  --account=chenyliu \
  --partition=secondary \
  --dependency=afterok:$run_jobid \
  --export=ALL,SWEEP_ROOT="$SWEEP_ROOT",RDE_CSV="$RDE_CSV" \
  slurm/analyze_sweep.sh)

echo "Submitted dependent analysis job: $an_jobid"
