#!/usr/bin/env python3
"""
Workload generator for TorchServe inference.

This component is responsible for:
1. Generating request arrival times using a Poisson process
2. Sending HTTP inference requests to TorchServe
3. Logging precise timestamps for:
   - when each request is SENT
   - when each request COMPLETES

Design principles:
- CPU-only (no GPU interaction)
- High-throughput friendly (minimal overhead per request)
- Deterministic logging for post-hoc analysis (paired with energy_monitor.py)
"""

# =========================
# STANDARD LIB IMPORTS
# =========================
import argparse          # CLI argument parsing
import json              # Load phase configuration (RPS schedule)
import queue             # Thread-safe task queue for workers
import random            # Random sampling (Poisson + image selection)
import threading         # Worker thread pool
import time              # High-resolution timing
from typing import Optional, List
from dataclasses import dataclass  # Lightweight struct for completion timestamps

# =========================
# THIRD-PARTY IMPORTS
# =========================
import requests          # HTTP client for TorchServe requests
from PIL import Image    # Image handling
from torchvision.datasets import CIFAR10  # Dataset for generating inputs


# =========================
# DATA STRUCTURE
# =========================
@dataclass
class Completion:
    """
    Simple container to store request completion timestamp.

    Why:
    - Keeps return value structured
    - Makes future extension easy (e.g., latency, status codes)
    """
    timestamp: float


# =========================
# IMAGE UTIL
# =========================
def img_to_bytes(img: Image.Image) -> bytes:
    """
    Convert PIL image → raw JPEG bytes.

    Why:
    - TorchServe expects raw bytes over HTTP POST
    - We pre-encode once to avoid repeated CPU overhead in workers
    """

    import io  # local import to reduce global import overhead

    buf = io.BytesIO()          # in-memory byte buffer
    img.save(buf, format="JPEG")  # encode image as JPEG
    return buf.getvalue()       # return raw bytes


# =========================
# TORCHSERVE WAIT
# =========================
def wait_for_torchserve(url: str, timeout: float = 180.0):
    """
    Block until TorchServe is ready.

    Mechanism:
    - Poll /ping endpoint
    - Exit when HTTP 200 is returned

    Why:
    - Ensures workload starts only after server is ready
    - Prevents skewed latency / failed requests at startup
    """

    start = time.time()

    while True:
        try:
            r = requests.get(f"{url}/ping", timeout=1.0)
            if r.status_code == 200:
                return  # server is ready
        except Exception:
            pass  # ignore transient connection errors

        # Timeout protection
        if time.time() - start > timeout:
            raise RuntimeError("TorchServe did not become ready")

        time.sleep(0.5)  # avoid tight loop


# =========================
# SEND REQUEST
# =========================
def send_request(url: str, model_name: str, sample: bytes) -> Completion:
    """
    Send a single inference request to TorchServe.

    Returns:
    - Completion object with timestamp at response arrival

    Important:
    - We DO NOT measure latency here directly
    - Instead we log timestamps separately (send + completion)
      → allows offline pairing with energy windows
    """

    requests.post(
        f"{url}/predictions/{model_name}",
        data=sample,     # raw image bytes
        timeout=30.0,    # avoid hanging requests
    )

    # Record completion timestamp immediately after response
    return Completion(time.perf_counter())


# =========================
# WORKER THREAD
# =========================
def worker_loop(
    url: str,
    model_name: str,
    samples: List[bytes],
    task_queue: "queue.Queue[Optional[int]]",
    sent_log,
    completion_log,
):
    """
    Worker thread function.

    Responsibilities:
    - Pull tasks from queue (each task = one request trigger)
    - Select random image
    - Send request
    - Log timestamps

    Architecture:
    - Producer (scheduler) → pushes tokens
    - Consumers (workers) → execute requests
    """

    while True:
        token = task_queue.get()  # block until task available

        try:
            # Shutdown signal (not used currently but good design)
            if token is None:
                return

            # -------------------------
            # RANDOM INPUT SELECTION
            # -------------------------
            sample = random.choice(samples)
            # Why random:
            # - avoids caching effects
            # - simulates real-world diverse inputs

            # -------------------------
            # SEND TIMESTAMP
            # -------------------------
            send_ts = time.perf_counter()
            sent_log.write(f"{send_ts}\n")
            sent_log.flush()
            # flush ensures real-time availability for energy monitor

            # -------------------------
            # SEND REQUEST
            # -------------------------
            c = send_request(url, model_name, sample)

            # -------------------------
            # COMPLETION TIMESTAMP
            # -------------------------
            completion_log.write(f"{c.timestamp}\n")
            completion_log.flush()

        finally:
            task_queue.task_done()  # mark task complete


# =========================
# POISSON SCHEDULER
# =========================
def poisson_scheduler(deadline: float, rps: int, q: queue.Queue):
    """
    Generate arrivals using a Poisson process.

    Mechanism:
    - Inter-arrival times ~ Exp(lambda = rps)

    Why Poisson:
    - Models real-world request arrivals
    - Avoids unrealistic burst patterns

    Output:
    - Pushes tokens into queue → triggers workers
    """

    if rps <= 0:
        return

    while time.perf_counter() < deadline:

        # Sample exponential inter-arrival time
        time.sleep(random.expovariate(rps))

        if time.perf_counter() < deadline:
            q.put(1)  # enqueue request trigger


# =========================
# LOAD DATASET
# =========================
def load_cifar10_samples(max_samples: int = 1000) -> List[bytes]:
    """
    Load CIFAR-10 dataset and preprocess into byte format.

    Important optimization:
    - Preprocessing is done ONCE here
    - Workers reuse pre-encoded bytes → avoids CPU bottleneck

    Returns:
    - List of JPEG-encoded images
    """

    print("Loading CIFAR-10 dataset...")

    ds = CIFAR10(root="./data", train=False, download=True)

    samples = []

    for i in range(min(len(ds), max_samples)):
        img = ds[i][0]                # PIL image
        samples.append(img_to_bytes(img))  # convert once

    print(f"Loaded {len(samples)} images")

    return samples


# =========================
# MAIN
# =========================
def main():
    """
    Entry point of workload generator.

    High-level flow:
    1. Parse arguments
    2. Load dataset
    3. Wait for TorchServe
    4. Start worker threads
    5. Execute Poisson phases
    """

    parser = argparse.ArgumentParser()

    # TorchServe endpoint
    parser.add_argument("--url", required=True)

    # Model name served by TorchServe
    parser.add_argument("--model-name", required=True)

    # JSON describing workload phases (RPS over time)
    parser.add_argument("--phases-json", required=True)

    # Number of concurrent worker threads
    parser.add_argument("--concurrency", type=int, default=16)

    # Output logs
    parser.add_argument("--completion-log", required=True)
    parser.add_argument("--sent-log", required=True)

    args = parser.parse_args()

    # -------------------------
    # LOAD DATASET
    # -------------------------
    samples = load_cifar10_samples(max_samples=1000)

    if len(samples) == 0:
        raise RuntimeError("No samples loaded!")

    # -------------------------
    # WAIT FOR SERVER
    # -------------------------
    wait_for_torchserve(args.url)

    # -------------------------
    # TASK QUEUE
    # -------------------------
    q = queue.Queue()

    # -------------------------
    # START WORKERS
    # -------------------------
    with open(args.completion_log, "w") as completion_log, \
         open(args.sent_log, "w") as sent_log:

        for _ in range(args.concurrency):
            threading.Thread(
                target=worker_loop,
                args=(
                    args.url,
                    args.model_name,
                    samples,
                    q,
                    sent_log,
                    completion_log,
                ),
                daemon=True,  # auto-exit when main exits
            ).start()

        # -------------------------
        # LOAD PHASE CONFIG
        # -------------------------
        with open(args.phases_json) as f:
            phases = json.load(f)

        # -------------------------
        # EXECUTE PHASES
        # -------------------------
        for phase in phases:
            print(f"Starting phase: {phase['name']}")

            # Compute phase end time
            deadline = time.perf_counter() + float(phase["duration"])

            # Generate arrivals
            poisson_scheduler(deadline, int(phase["rps"]), q)


if __name__ == "__main__":
    main()