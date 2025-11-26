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

# Try to import Zeus. If it fails (e.g. Zeus not installed on this node),
# we simply run the command without measurement.
try:
    from zeus.monitor import ZeusMonitor
except Exception as e:  # noqa: F841
    ZeusMonitor = None  # type: ignore[assignment]


def write_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    """
    Append a single summary row to a CSV log file.

    The CSV has a fixed schema with the following columns:
      - window_name
      - wall_time_s
      - zeus_time_s
      - zeus_total_energy_j

    If the file does not exist yet, a header row is written first.

    Args:
        csv_path: Path to the CSV file to append to.
        row:      Dictionary with the keys above and their values.
    """
    import csv

    file_exists = csv_path.exists()
    # Ensure the directory exists (e.g. runs/...).
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Open in "append" mode so we add a single row for each measurement.
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["window_name", "wall_time_s", "zeus_time_s", "zeus_total_energy_j"],
        )
        # Only write the header once, when the file is created.
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """
    Parse CLI arguments, optionally open a Zeus measurement window,
    run the target command, and write out measurement summaries.

    Behaviour:
      - If Zeus is installed and initialises correctly:
          * Open a measurement window named --window-name
          * Run the user command
          * Close the window and collect energy/time metrics
          * Optionally emit a JSON summary and/or append a CSV row
      - If Zeus is not installed or fails:
          * Just run the user command and propagate its exit code
          * No JSON/CSV outputs are written
    """
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
        help=(
            "Command to run, preceded by '--'. "
            "Everything after '--' is treated as the command."
        ),
    )

    args = parser.parse_args()

    # The argparse.REMAINDER will include the literal "--" separator if present.
    # Strip it so that 'cmd' is just the actual command and its arguments.
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    if not cmd:
        # Nothing to run – treat as a user error.
        print("No command specified after '--'.", file=sys.stderr)
        sys.exit(1)

    print(f"[Zeus] Command: {' '.join(cmd)}")
    print(f"[Zeus] Window name: {args.window_name}")
    print(f"[Zeus] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    # Decide whether we have a usable ZeusMonitor.
    monitor: Optional[ZeusMonitor]
    if ZeusMonitor is None:
        # Zeus not installed, import failed, or similar.
        print("[Zeus] Zeus is not installed or failed to import. Running without measurement.")
        monitor = None
    else:
        # In the typical usage, ZeusMonitor auto-detects the GPUs to measure,
        # usually based on CUDA_VISIBLE_DEVICES (see Zeus docs).
        monitor = ZeusMonitor()  # type: ignore[call-arg]
        print("[Zeus] ZeusMonitor initialised.")

    # Start the measurement window (if we have a monitor).
    if monitor is not None:
        # If this raises, we will still attempt to run the command and report.
        monitor.begin_window(args.window_name)

    # Record wall-clock start time around the entire command execution.
    start_time = time.time()
    # Default return code in case subprocess.run raises unexpectedly.
    return_code = 1
    try:
        # Run the user-provided command as a subprocess and wait for completion.
        proc = subprocess.run(cmd)
        return_code = proc.returncode
    finally:
        # Always capture the end time, even if subprocess.run throws.
        end_time = time.time()

    # Dictionary that will hold Zeus + wall-clock measurements if available.
    measurement_dict: Optional[Dict[str, Any]] = None

    # If Zeus was active, close the window and query the measurements.
    if monitor is not None:
        try:
            # Get the measurement object for this window.
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
            # If anything goes wrong during measurement collection, log it
            # and continue. The command has already run at this point.
            print(f"[Zeus] Failed to collect measurement: {e}", file=sys.stderr)

    # If we successfully collected measurements, optionally write them out.
    if measurement_dict is not None:
        # 1) JSON summary output (overwrites if file already exists).
        if args.json_out:
            json_path = Path(args.json_out)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w") as f:
                json.dump(measurement_dict, f, indent=2)

        # 2) Append a single CSV row to a log file.
        if args.log_csv:
            write_csv_row(Path(args.log_csv), measurement_dict)

    # Finally, exit with the same code as the wrapped command so that
    # batch systems / scripts see the original success/failure.
    sys.exit(return_code)


if __name__ == "__main__":
    main()
