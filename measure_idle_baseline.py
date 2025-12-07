#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from zeus.monitor import ZeusMonitor
except Exception:
    ZeusMonitor = None  # type: ignore[assignment]


def write_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    import csv

    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_name",
                "duration_s",
                "wall_time_s",
                "zeus_time_s",
                "zeus_total_energy_j",
                "avg_power_w",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure GPU idle-baseline energy using Zeus."
    )
    parser.add_argument(
        "--window-name",
        default="idle_baseline",
        help="Zeus window name (default: idle_baseline).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Measurement duration in seconds (default: 600).",
    )
    parser.add_argument(
        "--log-csv",
        default=None,
        help="Optional CSV file to append a single-row summary.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON file to write the measurement summary.",
    )

    args = parser.parse_args()

    if args.duration <= 0:
        print("Duration must be positive.", file=sys.stderr)
        sys.exit(1)

    print(f"[IdleBaseline] Window name: {args.window_name}")
    print(f"[IdleBaseline] Duration: {args.duration} s")
    print(f"[IdleBaseline] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    monitor: Optional[ZeusMonitor]
    if ZeusMonitor is None:
        print("[IdleBaseline] Zeus is not installed or failed to import. No measurements.")
        monitor = None
    else:
        monitor = ZeusMonitor()  # type: ignore[call-arg]
        print("[IdleBaseline] ZeusMonitor initialised.")

    if monitor is not None:
        monitor.begin_window(args.window_name)

    start_time = time.time()
    end_time = start_time + args.duration

    try:
        while True:
            now = time.time()
            if now >= end_time:
                break
            sleep_for = min(5.0, end_time - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        wall_end = time.time()

    measurement: Optional[Dict[str, Any]] = None

    if monitor is not None:
        try:
            m = monitor.end_window(args.window_name)
            wall_time_s = wall_end - start_time
            zeus_time_s = float(m.time)
            total_energy_j = float(m.total_energy)
            avg_power_w = total_energy_j / zeus_time_s if zeus_time_s > 0 else float("nan")

            measurement = {
                "window_name": args.window_name,
                "duration_s": args.duration,
                "wall_time_s": wall_time_s,
                "zeus_time_s": zeus_time_s,
                "zeus_total_energy_j": total_energy_j,
                "avg_power_w": avg_power_w,
            }

            print(
                "[IdleBaseline] "
                f"wall={wall_time_s:.3f}s, zeus_time={zeus_time_s:.3f}s, "
                f"energy={total_energy_j:.3f}J, avg_power={avg_power_w:.2f}W"
            )
        except Exception as e:
            print(f"[IdleBaseline] Failed to collect Zeus measurement: {e}", file=sys.stderr)

    if measurement is not None:
        if args.json_out:
            json_path = Path(args.json_out)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with json_path.open("w") as f:
                json.dump(measurement, f, indent=2)

        if args.log_csv:
            write_csv_row(Path(args.log_csv), measurement)

    sys.exit(0)


if __name__ == "__main__":
    main()
