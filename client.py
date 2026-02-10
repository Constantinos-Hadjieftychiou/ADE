#!/usr/bin/env python3
"""
TorchServe load generator with ultra-fine window-level GPU energy
and utilization measurement.

Corrected for scientifically valid interpretation on V100 GPUs.
"""

import argparse
import csv
import json
import logging
import os
import queue
import random
import statistics
import threading
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from torchvision.datasets import CIFAR10

import torch
import pynvml
from zeus.monitor import ZeusMonitor


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------
@dataclass
class RequestResult:
    latency_ms: float
    status_code: int


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def img_to_bytes(img: Image.Image) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------------------
# Request sender
# ------------------------------------------------------------------------------
def send_request(url: str, model_name: str, sample: bytes) -> RequestResult:
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{url}/predictions/{model_name}",
            data=sample,
            timeout=30.0,
        )
        return RequestResult((time.perf_counter() - t0) * 1e3, r.status_code)
    except Exception:
        return RequestResult((time.perf_counter() - t0) * 1e3, 0)


# ------------------------------------------------------------------------------
# Worker threads
# ------------------------------------------------------------------------------
def worker_loop(
    url: str,
    model_name: str,
    samples: List[bytes],
    task_queue: "queue.Queue[Optional[int]]",
    results: List[RequestResult],
    lock: threading.Lock,
):
    while True:
        token = task_queue.get()
        try:
            if token is None:
                return
            sample = random.choice(samples)
            res = send_request(url, model_name, sample)
            with lock:
                results.append(res)
        finally:
            task_queue.task_done()


# ------------------------------------------------------------------------------
# Poisson scheduler (independent thread)
# ------------------------------------------------------------------------------
def poisson_scheduler(deadline: float, rps: int, q: queue.Queue):
    if rps <= 0:
        return
    while time.perf_counter() < deadline:
        time.sleep(random.expovariate(rps))
        if time.perf_counter() < deadline:
            q.put(1)


# ------------------------------------------------------------------------------
# Main load runner
# ------------------------------------------------------------------------------
def run_load(
    url: str,
    model_name: str,
    phases: List[Dict[str, Any]],
    concurrency: int,
    csv_path: str,
    window_s: float,
    util_sample_ms: float,
):
    # --------------------------------------------------------------------------
    # Data
    # --------------------------------------------------------------------------
    ds = CIFAR10(root="./data", train=False, download=True)
    samples = [img_to_bytes(ds[i][0]) for i in range(100)]

    task_queue: queue.Queue = queue.Queue()
    results: List[RequestResult] = []
    lock = threading.Lock()

    for _ in range(concurrency):
        threading.Thread(
            target=worker_loop,
            args=(url, model_name, samples, task_queue, results, lock),
            daemon=True,
        ).start()

    # --------------------------------------------------------------------------
    # Energy + NVML
    # --------------------------------------------------------------------------
    zeus = ZeusMonitor(
        gpu_indices=[0],
        approx_instant_energy=True,  # REQUIRED for sub-100ms windows
    )

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    util_dt = util_sample_ms / 1000.0

    # --------------------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------------------
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "window_index",
            "phase",
            "rps",
            "requests",
            "is_idle_window",
            "energy_j",
            "avg_power_w",
            "energy_j_per_req",
            "gpu_active_flag",
            "gpu_util_samples",
        ])

        window_idx = 0

        for phase in phases:
            duration = float(phase["duration"])
            rps = int(phase["rps"])
            phase_name = phase["name"]

            num_windows = int(math.ceil(duration / window_s))

            for _ in range(num_windows):
                window_idx += 1
                start_count = len(results)

                t_start = time.perf_counter()
                t_end = t_start + window_s

                # ---------------- Energy window ----------------
                zeus.begin_window(f"w{window_idx}")

                # Start Poisson scheduler
                sched_thread = threading.Thread(
                    target=poisson_scheduler,
                    args=(t_end, rps, task_queue),
                    daemon=True,
                )
                sched_thread.start()

                # GPU util sampling (binary semantics)
                util_samples = []
                while time.perf_counter() < t_end:
                    util_samples.append(
                        pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu
                    )
                    time.sleep(util_dt)

                sched_thread.join()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                meas = zeus.end_window(f"w{window_idx}")

                # ---------------- Metrics ----------------
                window_results = results[start_count:]
                reqs = len(window_results)

                is_idle = (reqs == 0)

                energy_j = meas.total_energy
                avg_power = energy_j / window_s
                energy_per_req = energy_j / reqs if reqs > 0 else float("nan")

                gpu_active = int(any(u > 0 for u in util_samples))

                writer.writerow([
                    window_idx,
                    phase_name,
                    rps,
                    reqs,
                    is_idle,
                    energy_j,
                    avg_power,
                    energy_per_req,
                    gpu_active,
                    len(util_samples),
                ])
                f.flush()


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model-name", default="resnet-18")
    parser.add_argument("--phases-json", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--window-s", type=float, default=0.1)
    parser.add_argument("--util-sample-ms", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)

    with open(args.phases_json) as f:
        phases = json.load(f)

    run_load(
        url=args.url,
        model_name=args.model_name,
        phases=phases,
        concurrency=args.concurrency,
        csv_path=args.csv,
        window_s=args.window_s,
        util_sample_ms=args.util_sample_ms,
    )


if __name__ == "__main__":
    main()
