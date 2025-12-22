#!/usr/bin/env python3
"""
Submit a batch of experiments described in a JSON manifest via sbatch.

Manifest example (experiments_resnet.json):

[
  {
    "model_name": "resnet-18",
    "repeats": 3,
    "concurrency": 8,
    "window_s": 0.5,
    "pattern": "steady",
    "idle_calibration_seconds": 90
  },
  {
    "model_name": "resnet-18",
    "repeats": 3,
    "concurrency": 16,
    "window_s": 0.5,
    "pattern": "burst",
    "idle_calibration_seconds": 90
  },
  {
    "model_name": "resnet-50",
    "repeats": 3,
    "concurrency": 16,
    "window_s": 0.5,
    "pattern": "steady",
    "idle_calibration_seconds": 90
  }
]
"""

import argparse
import json
import os
import subprocess
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit multiple TorchServe experiments via sbatch."
    )
    p.add_argument(
        "--manifest",
        required=True,
        help="Path to JSON manifest describing experiments.",
    )
    p.add_argument(
        "--sbatch-script",
        default="job_resnet_model.sbatch",
        help="Name of sbatch script to use for each experiment.",
    )
    return p.parse_args()


def build_export_env(exp: Dict[str, Any]) -> str:
    """
    Convert manifest dict to sbatch --export string.
    Manifest keys are mapped to uppercase env vars.
    """
    parts: List[str] = ["ALL"]
    for k, v in exp.items():
        key = k.upper()
        # Simple conversion to string; booleans become True/False, etc.
        parts.append(f"{key}={v}")
    return ",".join(parts)


def main() -> None:
    args = parse_args()
    manifest_path = os.path.abspath(args.manifest)
    sbatch_script = args.sbatch_script

    project_dir = os.path.dirname(manifest_path) or os.getcwd()

    with open(manifest_path) as f:
        experiments = json.load(f)

    if not isinstance(experiments, list):
        raise SystemExit("Manifest must contain a JSON array of experiment objects.")

    print(f"Project dir: {project_dir}")
    print(f"Using sbatch script: {sbatch_script}")
    print(f"Loaded {len(experiments)} experiments from {manifest_path}")

    for idx, exp in enumerate(experiments, start=1):
        if not isinstance(exp, dict):
            print(f"[WARN] Experiment #{idx} is not an object; skipping")
            continue

        export_env = build_export_env(exp)
        cmd = ["sbatch", f"--export={export_env}", sbatch_script]
        print(f"\nSubmitting experiment #{idx}:")
        print("  Env:", export_env)
        print("  Cmd:", " ".join(cmd))

        subprocess.run(cmd, cwd=project_dir, check=True)


if __name__ == "__main__":
    main()