# UCN Pulse Fitting: Build and Slurm Array Run Notes

This repo builds a C++20 pulse-fitting executable, `run_batch_analysis`, using ROOT and `nlohmann_json`.

## Assumed environment

```bash
ROOT_CONDA_ENV=/projects/illinois/eng/physics/chenyliu/Ryan_ciyouh2/root
```

This environment should provide:

```bash
root-config
x86_64-conda-linux-gnu-c++
x86_64-conda-linux-gnu-cc
ld
```

Do not mix this ROOT with a different system compiler or GCC module unless you intentionally reconfigure against that stack.

## Build from a clean shell

From the repo root:

```bash
module purge
module load miniconda3

source activate /projects/illinois/eng/physics/chenyliu/Ryan_ciyouh2/root

# If source activate fails:
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate /projects/illinois/eng/physics/chenyliu/Ryan_ciyouh2/root
```

Check the toolchain:

```bash
which root-config
which c++
which x86_64-conda-linux-gnu-c++
which x86_64-conda-linux-gnu-cc
which ld

root-config --version
root-config --cxx
root-config --libs
```

Expected pattern:

```text
root-config -> $CONDA_PREFIX/bin/root-config
c++         -> $CONDA_PREFIX/bin/c++
ld          -> $CONDA_PREFIX/bin/ld
root-config --cxx -> x86_64-conda-linux-gnu-c++
```

## Clean configure and build

Always delete the old build directory when switching compiler/ROOT environments. CMake caches compilers.

```bash
rm -rf build

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc" \
  -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DUCN_ENABLE_WARNINGS=ON \
  -DUCN_BUILD_TESTS=OFF

cmake --build build -j 4
```

If building inside a Slurm job:

```bash
cmake --build build -j "${SLURM_CPUS_PER_TASK:-4}"
```

## Quick local run

The executable expects exactly one config argument:

```bash
./build/run_batch_analysis config/default.json
```

For non-array local testing, keep:

```json
"shard_index": 0,
"num_shards": 1
```

## Slurm array logic

The array job should not let all tasks write to the same output files.

The runner uses these config fields:

```json
"shard_index": 0,
"num_shards": 1
```

The Slurm script generates one config per array task:

```text
config/generated/config_${SLURM_ARRAY_TASK_ID}.json
```

with:

```json
"shard_index": SLURM_ARRAY_TASK_ID,
"num_shards": SLURM_ARRAY_TASK_COUNT
```

With the shard-aware runner, each task should write to a task-specific output folder, for example:

```text
output/array/task_000/
output/array/task_001/
output/array/task_002/
```

Each task writes its own:

```text
all_pulses.csv
all_windows.csv
run_segment_summary.csv
analysis_metadata.json
```

## Submit array job

```bash
sbatch slurm/run_ucn_array.slurm
```

For 32 shards:

```bash
#SBATCH --array=0-31
```

For 64 shards:

```bash
#SBATCH --array=0-63
```

## Merge outputs after the job

From repo root:

```bash
mkdir -p output/merged

python3 - << 'PY'
from pathlib import Path
import pandas as pd

base = Path("output/array")
out = Path("output/merged")
out.mkdir(parents=True, exist_ok=True)

for name in ["all_pulses.csv", "all_windows.csv", "run_segment_summary.csv"]:
    files = sorted(base.glob(f"task_*/{name}"))
    dfs = []
    for f in files:
        if f.exists() and f.stat().st_size > 0:
            dfs.append(pd.read_csv(f))
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(out / name, index=False)
        print(f"wrote {out / name} with {len(merged)} rows")
    else:
        print(f"no files found for {name}")
PY
```

## Common failure mode

If linking fails with errors like:

```text
undefined reference to GLIBC_PRIVATE
undefined reference to GLIBCXX_3.4.31
undefined reference to CXXABI_1.3.15
```

then ROOT, compiler, linker, and libstdc++ are mixed. Fix by:

```bash
module purge
module load miniconda3
source activate /projects/illinois/eng/physics/chenyliu/Ryan_ciyouh2/root
rm -rf build
```

then configure using the explicit conda compilers and `CMAKE_PREFIX_PATH=$CONDA_PREFIX`.

Do not patch CMake with random `-lpthread`, `-ldl`, or `-lstdc++` for this issue.
