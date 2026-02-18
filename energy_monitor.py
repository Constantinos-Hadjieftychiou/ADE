#!/usr/bin/env python3
"""
Energy monitor and window-based metric logger.

Responsibilities:
- Define fixed-duration energy windows
- Measure GPU energy via Zeus
- Sample NVML utilization
- Count completed inference requests per window
"""

import argparse
import csv
import json
import math
import os
import time

import torch
import pynvml
from zeus.monitor import ZeusMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases-json", required=True)
    parser.add_argument("--completion-log", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--window-s", type=float, default=0.05)
    parser.add_argument("--util-sample-ms", type=float, default=2.0)
    args = parser.parse_args()

    while not os.path.exists(args.completion_log):
        time.sleep(0.01)

    completion_file = open(args.completion_log, "r")

    zeus = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    util_dt = args.util_sample_ms / 1000.0

    with open(args.phases_json) as f:
        phases = json.load(f)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_index",
            "phase",
            "rps",
            "completed_requests",
            "request_idle_window",
            "energy_j",
            "avg_power_w",
            "gpu_active_flag",
        ])

        window_idx = 0

        for phase in phases:
            num_windows = int(math.ceil(phase["duration"] / args.window_s))

            for _ in range(num_windows):
                window_idx += 1
                t_start = time.perf_counter()
                t_end = t_start + args.window_s

                zeus.begin_window(f"w{window_idx}")

                util_samples = []
                while time.perf_counter() < t_end:
                    util_samples.append(
                        pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu
                    )
                    time.sleep(util_dt)

                torch.cuda.synchronize()
                meas = zeus.end_window(f"w{window_idx}")

                completed = 0
                while True:
                    pos = completion_file.tell()
                    line = completion_file.readline()
                    if not line:
                        completion_file.seek(pos)
                        break
                    ts = float(line.strip())
                    if ts < t_end:
                        completed += 1
                    else:
                        completion_file.seek(pos)
                        break

                writer.writerow([
                    window_idx,
                    phase["name"],
                    phase["rps"],
                    completed,
                    completed == 0,
                    meas.total_energy,
                    meas.total_energy / args.window_s,
                    int(any(u > 0 for u in util_samples)),
                ])
                f.flush()

    completion_file.close()


if __name__ == "__main__":
    main()
