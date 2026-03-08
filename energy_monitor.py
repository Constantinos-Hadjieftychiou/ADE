#!/usr/bin/env python3
"""
Energy monitor with:

- drift-free window scheduling
- requests_sent per window
- completed_requests per window
- GPU utilization statistics
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

    parser.add_argument("--phases-json", required=True)
    parser.add_argument("--completion-log", required=True)
    parser.add_argument("--sent-log", required=True)
    parser.add_argument("--csv", required=True)

    parser.add_argument("--window-s", type=float, default=0.05)
    parser.add_argument("--util-sample-ms", type=float, default=2.0)

    args = parser.parse_args()

    # -----------------------------------------------------
    # Wait for logs
    # -----------------------------------------------------

    while not os.path.exists(args.completion_log):
        time.sleep(0.01)

    while not os.path.exists(args.sent_log):
        time.sleep(0.01)

    completion_file = open(args.completion_log, "r")
    sent_file = open(args.sent_log, "r")

    # -----------------------------------------------------
    # GPU setup
    # -----------------------------------------------------

    zeus = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)

    pynvml.nvmlInit()

    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    util_dt = args.util_sample_ms / 1000.0

    # -----------------------------------------------------
    # Load phases
    # -----------------------------------------------------

    with open(args.phases_json) as f:
        phases = json.load(f)

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    # -----------------------------------------------------
    # Global experiment clock
    # -----------------------------------------------------

    experiment_start = time.perf_counter()

    window_idx = 0

    with open(args.csv, "w", newline="") as f:

        writer = csv.writer(f)

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
        # Iterate phases
        # -------------------------------------------------

        for phase in phases:

            num_windows = int(math.ceil(phase["duration"] / args.window_s))

            for w in range(num_windows):

                window_idx += 1

                # -----------------------------------------
                # Drift-free window schedule
                # -----------------------------------------

                window_start = experiment_start + (window_idx - 1) * args.window_s
                window_end = window_start + args.window_s

                while time.perf_counter() < window_start:
                    time.sleep(0.0005)

                zeus.begin_window(f"w{window_idx}")

                util_samples = []

                while time.perf_counter() < window_end:

                    util = pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu

                    util_samples.append(util)

                    time.sleep(util_dt)

                torch.cuda.synchronize()

                meas = zeus.end_window(f"w{window_idx}")

                # -----------------------------------------
                # Count sent requests
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
                # Count completed requests
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