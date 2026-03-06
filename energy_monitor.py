#!/usr/bin/env python3
"""
Energy monitor with:
- requests_sent per window
- completed_requests per window
- GPU utilization statistics
- robust log parsing
"""

import argparse
import csv
import json
import math
import os
import time

import torch
import numpy as np
import pynvml
from zeus.monitor import ZeusMonitor


def wait_for_logs(sent_log, completion_log):
    while True:
        if os.path.exists(sent_log) and os.path.exists(completion_log):
            return
        time.sleep(0.05)


def safe_read_timestamp(line):
    try:
        return float(line.strip())
    except Exception:
        return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--phases-json", required=True)
    parser.add_argument("--completion-log", required=True)
    parser.add_argument("--sent-log", required=True)
    parser.add_argument("--csv", required=True)

    parser.add_argument("--window-s", type=float, default=0.05)
    parser.add_argument("--util-sample-ms", type=float, default=2.0)

    args = parser.parse_args()

    wait_for_logs(args.sent_log, args.completion_log)

    sent_file = open(args.sent_log, "r")
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

                    util = pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu
                    util_samples.append(util)

                    time.sleep(util_dt)

                torch.cuda.synchronize()

                meas = zeus.end_window(f"w{window_idx}")

                # ------------------------
                # COUNT SENT REQUESTS
                # ------------------------

                sent = 0

                while True:

                    pos = sent_file.tell()

                    line = sent_file.readline()

                    if not line:
                        sent_file.seek(pos)
                        break

                    ts = safe_read_timestamp(line)

                    if ts is None:
                        continue

                    if ts < t_end:
                        sent += 1
                    else:
                        sent_file.seek(pos)
                        break

                # ------------------------
                # COUNT COMPLETED REQUESTS
                # ------------------------

                completed = 0

                while True:

                    pos = completion_file.tell()

                    line = completion_file.readline()

                    if not line:
                        completion_file.seek(pos)
                        break

                    ts = safe_read_timestamp(line)

                    if ts is None:
                        continue

                    if ts < t_end:
                        completed += 1
                    else:
                        completion_file.seek(pos)
                        break

                util_samples = np.array(util_samples)

                util_mean = float(util_samples.mean()) if len(util_samples) else 0
                util_std = float(util_samples.std()) if len(util_samples) else 0
                util_max = float(util_samples.max()) if len(util_samples) else 0

                gpu_active = int(np.any(util_samples > 0))

                writer.writerow([
                    window_idx,
                    phase["name"],
                    phase["rps"],
                    sent,
                    completed,
                    completed == 0,
                    meas.total_energy,
                    meas.total_energy / args.window_s,
                    util_mean,
                    util_std,
                    util_max,
                    gpu_active
                ])

                f.flush()

    sent_file.close()
    completion_file.close()

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()