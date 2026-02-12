#!/usr/bin/env python3
"""
TorchServe load generator with ultra-fine window-level GPU energy measurement.

This script is designed to be scientifically valid for sub-100ms
energy windows on NVIDIA V100-class GPUs.

What it records per time window:
- total GPU energy (Joules)
- whether any requests completed in the window
- whether the GPU was active at least once (binary NVML signal)

Important design principle:
Energy is measured at the window level, NOT attributed to individual requests.
"""

# ------------------------------------------------------------------------------
# Standard library imports
# ------------------------------------------------------------------------------

import argparse              # Command-line argument parsing
import csv                   # Writing structured CSV output
import json                  # Loading phase definitions
import logging               # Structured logging
import os                    # Filesystem operations
import queue                 # Thread-safe task queue
import random                # Random sampling and Poisson process
import threading             # Concurrent request workers
import time                  # High-resolution timers
import math                  # Ceiling for window counts
from dataclasses import dataclass  # Lightweight data containers
from typing import Any, Dict, List, Optional  # Type hints for clarity

# ------------------------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------------------------

import requests               # HTTP client for TorchServe inference requests
from PIL import Image         # Image handling
from torchvision.datasets import CIFAR10  # Dataset for inference inputs

import torch                  # CUDA synchronization
import pynvml                 # GPU utilization sampling
from zeus.monitor import ZeusMonitor  # GPU energy measurement

# ------------------------------------------------------------------------------
# Suppress known harmless warning
# ------------------------------------------------------------------------------
# This warning is emitted by Python's multiprocessing resource tracker
# when threads exit. It does NOT affect correctness or results.
import warnings
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be .* leaked semaphore objects"
)

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
# Simple timestamped logging, useful for long HPC jobs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------

@dataclass
class RequestResult:
    """
    Stores minimal information about a completed request.

    We deliberately do NOT store latency here because:
    - latency attribution at sub-100ms windows is misleading
    - this study focuses on energy, not QoS
    """
    status_code: int


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def img_to_bytes(img: Image.Image) -> bytes:
    """
    Convert a PIL Image into JPEG-encoded bytes.

    TorchServe expects raw image bytes over HTTP.
    JPEG is chosen for compactness and realism.
    """
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def send_request(url: str, model_name: str, sample: bytes) -> RequestResult:
    """
    Send a single inference request to TorchServe.

    This function is intentionally simple:
    - no retries
    - no batching
    - minimal overhead

    Any failure is recorded as status_code = 0.
    """
    try:
        r = requests.post(
            f"{url}/predictions/{model_name}",
            data=sample,
            timeout=30.0,   # Prevent hung connections
        )
        return RequestResult(r.status_code)
    except Exception:
        # Network or server error
        return RequestResult(0)


# ------------------------------------------------------------------------------
# Worker thread loop
# ------------------------------------------------------------------------------

def worker_loop(
    url: str,
    model_name: str,
    samples: List[bytes],
    task_queue: "queue.Queue[Optional[int]]",
    results: List[RequestResult],
    lock: threading.Lock,
):
    """
    Worker thread that processes inference requests.

    Each worker:
    - waits for a token in the task queue
    - sends exactly one inference request per token
    - appends the result to a shared list

    The queue provides backpressure and decouples
    request generation from request execution.
    """
    while True:
        token = task_queue.get()
        try:
            # None is a shutdown signal (not used here, but safe design)
            if token is None:
                return

            # Randomly select an image sample
            sample = random.choice(samples)

            # Send inference request
            res = send_request(url, model_name, sample)

            # Protect shared results list
            with lock:
                results.append(res)

        finally:
            # Mark this task as completed
            task_queue.task_done()


# ------------------------------------------------------------------------------
# Poisson request scheduler
# ------------------------------------------------------------------------------

def poisson_scheduler(deadline: float, rps: int, q: queue.Queue):
    """
    Generate request tokens according to a Poisson process.

    Properties:
    - Inter-arrival times ~ Exp(rps)
    - Stops strictly at window boundary (deadline)
    - Produces *arrival intent*, not actual execution

    This models realistic microservice traffic.
    """
    if rps <= 0:
        return

    while time.perf_counter() < deadline:
        # Sleep for exponentially distributed inter-arrival time
        time.sleep(random.expovariate(rps))

        # Only enqueue if still inside the window
        if time.perf_counter() < deadline:
            q.put(1)


# ------------------------------------------------------------------------------
# Main load + measurement routine
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
    """
    Run the full experiment:
    - start worker threads
    - iterate over workload phases
    - measure energy in fixed-size windows
    - write results to CSV
    """

    # --------------------------------------------------------------------------
    # Load inference data
    # --------------------------------------------------------------------------
    # CIFAR-10 is used purely as a convenient source of images.
    # The exact dataset content is not important for energy behavior.
    ds = CIFAR10(root="./data", train=False, download=True)

    # Pre-encode a fixed set of samples to avoid runtime overhead
    samples = [img_to_bytes(ds[i][0]) for i in range(100)]

    # --------------------------------------------------------------------------
    # Shared state between threads
    # --------------------------------------------------------------------------
    task_queue: queue.Queue = queue.Queue()
    results: List[RequestResult] = []
    lock = threading.Lock()

    # Start worker threads
    for _ in range(concurrency):
        threading.Thread(
            target=worker_loop,
            args=(url, model_name, samples, task_queue, results, lock),
            daemon=True,   # Allows clean process exit
        ).start()

    # --------------------------------------------------------------------------
    # Energy and GPU monitoring setup
    # --------------------------------------------------------------------------
    # approx_instant_energy=True is REQUIRED for sub-100ms windows.
    zeus = ZeusMonitor(
        gpu_indices=[0],
        approx_instant_energy=True,
    )

    # Initialize NVML for utilization sampling
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    # Convert utilization sampling interval to seconds
    util_dt = util_sample_ms / 1000.0

    # --------------------------------------------------------------------------
    # CSV output setup
    # --------------------------------------------------------------------------
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # CSV header
        writer.writerow([
            "window_index",
            "phase",
            "rps",
            "completed_requests",
            "is_idle_window",
            "energy_j",
            "avg_power_w",
            "gpu_active_flag",
        ])

        window_idx = 0

        # ----------------------------------------------------------------------
        # Iterate over workload phases
        # ----------------------------------------------------------------------
        for phase in phases:
            duration = float(phase["duration"])
            rps = int(phase["rps"])
            phase_name = phase["name"]

            # Number of fixed-size windows in this phase
            num_windows = int(math.ceil(duration / window_s))

            for _ in range(num_windows):
                window_idx += 1

                # Snapshot number of completed requests at window start
                start_count = len(results)

                # Window boundaries
                t_start = time.perf_counter()
                t_end = t_start + window_s

                # ---------------- Energy window begins ----------------
                zeus.begin_window(f"w{window_idx}")

                # Start Poisson scheduler in parallel
                sched_thread = threading.Thread(
                    target=poisson_scheduler,
                    args=(t_end, rps, task_queue),
                    daemon=True,
                )
                sched_thread.start()

                # Sample GPU utilization during the window
                util_samples = []
                while time.perf_counter() < t_end:
                    util_samples.append(
                        pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu
                    )
                    time.sleep(util_dt)

                # Ensure scheduler finished
                sched_thread.join()

                # Ensure all CUDA work issued in this window is completed
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # ---------------- Energy window ends ----------------
                meas = zeus.end_window(f"w{window_idx}")

                # Compute number of requests completed in this window
                window_results = results[start_count:]
                completed = len(window_results)

                # Request-level idleness
                is_idle = (completed == 0)

                # Binary GPU activity flag (did NVML ever see activity?)
                gpu_active = int(any(u > 0 for u in util_samples))

                # Write one row per window
                writer.writerow([
                    window_idx,
                    phase_name,
                    rps,
                    completed,
                    is_idle,
                    meas.total_energy,
                    meas.total_energy / window_s,
                    gpu_active,
                ])

                # Flush immediately for fault tolerance on HPC
                f.flush()


# ------------------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------------------

def main():
    """
    Parse CLI arguments and launch the experiment.
    """
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

    # Deterministic behavior for reproducibility
    if args.random_seed is not None:
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)

    # Load workload phases
    with open(args.phases_json) as f:
        phases = json.load(f)

    # Run experiment
    run_load(
        url=args.url,
        model_name=args.model_name,
        phases=phases,
        concurrency=args.concurrency,
        csv_path=args.csv,
        window_s=args.window_s,
        util_sample_ms=args.util_sample_ms,
    )


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
