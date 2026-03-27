#!/usr/bin/env python3
"""
Energy monitor.

This component:
- Tracks GPU energy consumption per time window
- Tracks GPU utilization statistics
- Aligns requests (sent + completed) with time windows

Key idea:
→ Correlate workload (requests) with energy usage over time
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import pynvml
from zeus.monitor import ZeusMonitor


def main():

    parser = argparse.ArgumentParser()

    # Input files
    parser.add_argument("--phases-json", required=True)
    parser.add_argument("--completion-log", required=True)
    parser.add_argument("--sent-log", required=True)

    # Output CSV
    parser.add_argument("--csv", required=True)

    # Sampling configuration
    parser.add_argument("--window-s", type=float, default=0.05)
    parser.add_argument("--util-sample-ms", type=float, default=2.0)

    args = parser.parse_args()

    # -----------------------------------------------------
    # WAIT FOR LOG FILES
    # -----------------------------------------------------
    # Ensures workload generator has started

    while not os.path.exists(args.completion_log):
        time.sleep(0.01)

    while not os.path.exists(args.sent_log):
        time.sleep(0.01)

    completion_file = open(args.completion_log, "r")
    sent_file = open(args.sent_log, "r")

    # -----------------------------------------------------
    # GPU SETUP
    # -----------------------------------------------------

    zeus = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)
    # Zeus tracks energy per window

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    util_dt = args.util_sample_ms / 1000.0  # convert ms → seconds

    # -----------------------------------------------------
    # LOAD PHASES
    # -----------------------------------------------------

    with open(args.phases_json) as f:
        phases = json.load(f)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    # -----------------------------------------------------
    # GLOBAL CLOCK
    # -----------------------------------------------------
    # Ensures drift-free scheduling across windows

    experiment_start = time.perf_counter()
    window_idx = 0

    with open(args.csv, "w", newline="") as f:

        writer = csv.writer(f)

        # CSV header
        writer.writerow([
            "window_index",
            "phase",
            "rps",
            "requests_sent",
            "completed_requests",
            "request_idle_window",
            "energy_j",
            "avg_power_w",
            "gpu_util_mean",
            "gpu_util_std",
            "gpu_util_max",
            "gpu_active_flag",
        ])

        # -------------------------------------------------
        # PHASE LOOP
        # -------------------------------------------------
        for phase in phases:

            num_windows = int(math.ceil(phase["duration"] / args.window_s))

            for w in range(num_windows):

                window_idx += 1

                # -----------------------------------------
                # DRIFT-FREE WINDOW TIMING
                # -----------------------------------------
                window_start = experiment_start + (window_idx - 1) * args.window_s
                window_end = window_start + args.window_s

                while time.perf_counter() < window_start:
                    time.sleep(0.0005)

                zeus.begin_window(f"w{window_idx}")

                util_samples = []

                # -----------------------------------------
                # SAMPLE GPU UTILIZATION
                # -----------------------------------------
                while time.perf_counter() < window_end:

                    util = pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu
                    util_samples.append(util)

                    time.sleep(util_dt)

                torch.cuda.synchronize()

                meas = zeus.end_window(f"w{window_idx}")

                # -----------------------------------------
                # COUNT SENT REQUESTS
                # -----------------------------------------
                sent = 0

                while True:
                    pos = sent_file.tell()
                    line = sent_file.readline()

                    if not line:
                        sent_file.seek(pos)
                        break

                    ts = float(line.strip())

                    if ts < window_end:
                        sent += 1
                    else:
                        sent_file.seek(pos)
                        break

                # -----------------------------------------
                # COUNT COMPLETED REQUESTS
                # -----------------------------------------
                completed = 0

                while True:
                    pos = completion_file.tell()
                    line = completion_file.readline()

                    if not line:
                        completion_file.seek(pos)
                        break

                    ts = float(line.strip())

                    if ts < window_end:
                        completed += 1
                    else:
                        completion_file.seek(pos)
                        break

                util_samples = np.array(util_samples)

                # -----------------------------------------
                # WRITE WINDOW DATA
                # -----------------------------------------
                writer.writerow([
                    window_idx,
                    phase["name"],
                    phase["rps"],
                    sent,
                    completed,
                    completed == 0,
                    meas.total_energy,
                    meas.total_energy / args.window_s,
                    float(util_samples.mean()) if len(util_samples) else 0.0,
                    float(util_samples.std()) if len(util_samples) else 0.0,
                    float(util_samples.max()) if len(util_samples) else 0.0,
                    int(np.any(util_samples > 0)),
                ])

                f.flush()

    completion_file.close()
    sent_file.close()

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()