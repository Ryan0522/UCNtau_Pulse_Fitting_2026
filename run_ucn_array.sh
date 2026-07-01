#!/bin/bash
#SBATCH --job-name=ucn_pulse
#SBATCH --account=chenyliu
#SBATCH --partition=secondary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-31
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/ucn_%A_%a.out
#SBATCH --error=logs/ucn_%A_%a.err

ROOT_CONDA_ENV="/projects/illinois/eng/physics/chenyliu/Ryan_ciyouh2/root"
BASE_CONFIG="config/default_2021_fast.json"
GENERATED_CONFIG_DIR="config/generated"
TASK_CONFIG="${GENERATED_CONFIG_DIR}/config_${SLURM_ARRAY_TASK_ID}.json"
EXECUTABLE="./build/run_batch_analysis"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export ROOT_HIST=0

module purge
module load miniconda3
module load python
source activate "${ROOT_CONDA_ENV}"

mkdir -p logs "${GENERATED_CONFIG_DIR}"

echo "Job ID:      ${SLURM_JOB_ID}"
echo "Array task:  ${SLURM_ARRAY_TASK_ID}"
echo "Array count: ${SLURM_ARRAY_TASK_COUNT}"
echo "Host:        $(hostname)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"

echo "root-config: $(which root-config)"
echo "c++:         $(which c++)"
echo "ld:          $(which ld)"
root-config --version
root-config --cxx

if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "ERROR: executable not found: ${EXECUTABLE}"
    exit 2
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "ERROR: base config not found: ${BASE_CONFIG}"
    exit 3
fi

python3 - << EOF
import json
from pathlib import Path

base_config = Path("${BASE_CONFIG}")
task_config = Path("${TASK_CONFIG}")

with base_config.open() as f:
    cfg = json.load(f)

cfg["shard_index"] = int("${SLURM_ARRAY_TASK_ID}")
cfg["num_shards"] = int("${SLURM_ARRAY_TASK_COUNT}")

task_config.parent.mkdir(parents=True, exist_ok=True)

with task_config.open("w") as f:
    json.dump(cfg, f, indent=2)

print(f"Wrote {task_config}")
print(f"shard_index={cfg['shard_index']}, num_shards={cfg['num_shards']}")
print(f"output_folder={cfg.get('output_folder')}")
EOF

echo "Running: ${EXECUTABLE} ${TASK_CONFIG}"
"${EXECUTABLE}" "${TASK_CONFIG}"

echo "Done task ${SLURM_ARRAY_TASK_ID}"