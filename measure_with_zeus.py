#!/usr/bin/env python3
"""
measure_with_zeus.py

Run an arbitrary command under a Zeus energy measurement window.

Example usage (as in job.sbatch):

python measure_with_zeus.py \
  --window-name resnet18_image_burst \
  --log-csv runs/.../zeus_windows.csv \
  --json-out runs/.../zeus_summary.json \
  -- \
  python client.py ...client-args...

If Zeus is unavailable or fails, the script still runs the command and exits
with the same return code, but no measurements are written.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from zeus.monitor import ZeusMonitor
except Exception as e:  # noqa: F841
    ZeusMonitor = None  # type: ignore[assignment]


def write_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    import csv

    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["window_name", "wall_time_s", "zeus_time_s", "zeus_total_energy_j"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure GPU energy/time for a command using Zeus."
    )
    parser.add_argument(
        "--window-name",
        required=True,
        help="Name of the Zeus measurement window.",
    )
    parser.add_argument(
        "--log-csv",
        default=None,
        help="CSV file to append a single-row summary (optional).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="JSON file to write measurement summary (optional).",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run, preceded by '--'.",
    )

    args = parser.parse_args()

    # Strip the leading "--" if present
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        print("No command specified after '--'.", file=sys.stderr)
        sys.exit(1)

    print(f"[Zeus] Command: {' '.join(cmd)}")
    print(f"[Zeus] Window name: {args.window_name}")
    print(f"[Zeus] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    monitor: Optional[ZeusMonitor]
    if ZeusMonitor is None:
        print("[Zeus] Zeus is not installed or failed to import. Running without measurement.")
        monitor = None
    else:
        # Auto-detect GPUs via CUDA_VISIBLE_DEVICES as in the Zeus docs
        monitor = ZeusMonitor()  # type: ignore[call-arg]
        print("[Zeus] ZeusMonitor initialised.")

    # Start measurement window
    if monitor is not None:
        monitor.begin_window(args.window_name)

    start_time = time.time()
    try:
        proc = subprocess.run(cmd)
        return_code = proc.returncode
    finally:
        end_time = time.time()

    measurement_dict: Optional[Dict[str, Any]] = None

    if monitor is not None:
        try:
            m = monitor.end_window(args.window_name)
            measurement_dict = {
                "window_name": args.window_name,
                "wall_time_s": end_time - start_time,
                "zeus_time_s": float(m.time),
                "zeus_total_energy_j": float(m.total_energy),
            }
            print(
                f"[Zeus] Window '{args.window_name}': "
                f"wall={measurement_dict['wall_time_s']:.3f}s, "
                f"zeus_time={measurement_dict['zeus_time_s']:.3f}s, "
                f"energy={measurement_dict['zeus_total_energy_j']:.3f}J"
            )
        except Exception as e:
            print(f"[Zeus] Failed to collect measurement: {e}", file=sys.stderr)

    # Write outputs if requested and if we have measurements
    if measurement_dict is not None:
        if args.json_out:
            json_path = Path(args.json_out)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w") as f:
                json.dump(measurement_dict, f, indent=2)

        if args.log_csv:
            write_csv_row(Path(args.log_csv), measurement_dict)

    # Propagate the command's exit status
    sys.exit(return_code)


if __name__ == "__main__":
    main()
