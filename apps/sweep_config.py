#!/usr/bin/env python3
"""
Sweep one config parameter without changing C++.
"""

import argparse
import json
import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path


def parse_value(raw):
    s = str(raw).strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        x = float(s)
    except ValueError:
        return s
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    return x

def set_dotted(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    parent = cfg
    for key in keys[:-1]:
        if key not in parent or not isinstance(parent[key], dict):
            raise KeyError(f"Cannot descend into '{key}' while setting '{dotted_key}'.")
        parent = parent[key]

    final_key = keys[-1]
    if final_key not in parent:
        raise KeyError(f"Final key '{final_key}' not found while setting '{dotted_key}'.")

    old_value = parent[final_key]

    # Preserve numeric int/float type when the base config already tells us the type.
    if isinstance(old_value, bool):
        new_value = bool(value)
    elif isinstance(old_value, int):
        new_value = int(float(value))
    elif isinstance(old_value, float):
        new_value = float(value)
    else:
        new_value = value

    parent[final_key] = new_value
    return old_value, new_value

def safe_text(x):
    s = str(x)
    s = s.replace("-", "m").replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_+=-]+", "_", s)
    return s.strip("_") or "value"

def task_name(shard_index):
    return f"task_{shard_index:03d}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="Base JSON config.")
    p.add_argument("--exe", default="./build/run_batch_analysis")
    p.add_argument("--sweep-name", required=True, help="Folder under output_folder.")
    p.add_argument("--param", required=True, help="Dotted config key, e.g. fit_settings.discovery_delta_nll_cut")
    p.add_argument("--label", required=True, help="Short label used in folder names, e.g. nll")
    p.add_argument("--values", nargs="+", required=True, help="Sweep values.")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.shards < 1:
        raise ValueError("--shards must be >= 1")

    if args.task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    else:
        task_id = args.task_id

    value_index = task_id // args.shards
    shard_index = task_id % args.shards

    if value_index < 0 or value_index >= len(args.values):
        raise RuntimeError(
            f"task_id={task_id} is outside sweep range. "
            f"Need 0 <= task_id < {len(args.values) * args.shards}."
        )

    raw_value = args.values[value_index]
    parsed_value = parse_value(raw_value)

    base_path = Path(args.base)
    with base_path.open() as f:
        base_cfg = json.load(f)

    cfg = deepcopy(base_cfg)
    old_value, new_value = set_dotted(cfg, args.param, parsed_value)

    cfg["num_shards"] = args.shards
    cfg["shard_index"] = shard_index

    value_dir_name = f"{args.label}_{safe_text(raw_value)}"
    cfg["array_output_subdir"] = f"{args.sweep_name}/{value_dir_name}"

    value_dir = Path(cfg["output_folder"]) / cfg["array_output_subdir"]
    task_dir = value_dir / task_name(shard_index) if args.shards > 1 else value_dir

    value_dir.mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)

    value_cfg = deepcopy(cfg)
    value_cfg["shard_index"] = 0
    with (value_dir / "config_value.json").open("w") as f:
        json.dump(value_cfg, f, indent=2)
        f.write("\n")

    config_used = task_dir / "config_used.json"
    with config_used.open("w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    command = [args.exe, str(config_used)]
    with (task_dir / "command.txt").open("w") as f:
        f.write(" ".join(command) + "\n")

    sweep_info = {
        "task_id": task_id,
        "value_index": value_index,
        "shard_index": shard_index,
        "num_shards": args.shards,
        "sweep_name": args.sweep_name,
        "param": args.param,
        "label": args.label,
        "raw_value": raw_value,
        "old_value": old_value,
        "new_value": new_value,
        "base_config": str(base_path),
        "value_dir": str(value_dir),
        "task_dir": str(task_dir),
        "config_used": str(config_used),
    }
    with (task_dir / "sweep_info.json").open("w") as f:
        json.dump(sweep_info, f, indent=2)
        f.write("\n")

    print(json.dumps(sweep_info, indent=2), flush=True)
    print("command:", " ".join(command), flush=True)

    if args.dry_run:
        return 0
    return subprocess.call(command)

if __name__ == "__main__":
    raise SystemExit(main())