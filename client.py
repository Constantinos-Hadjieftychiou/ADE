#!/usr/bin/env python3
"""
client.py — TorchServe load generator inspired by official benchmarks.

Features:
  - Randomised burst/idle traffic (microservice-like).
  - Configurable concurrency (threads) & requests/sec.
  - Optional steady / poisson patterns like synthetic benchmarks.
  - Per-request CSV metrics for offline analysis.
  - Multi-phase patterns via --phases-json.
"""

import argparse
import csv
import io
import json
import logging
import os
import queue
import random
import statistics
import threading
import time
from dataclasses import dataclass
from typing import List, Any, Dict, Optional, Literal

import requests
from PIL import Image
from torchvision.datasets import CIFAR10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

TrafficPattern = Literal["burst", "steady", "poisson"]


@dataclass
class RequestResult:
    ts_start: float
    ts_end: float
    latency_ms: float
    status_code: int
    error: Optional[str]


def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def send_request(
    url: str,
    model_name: str,
    sample: Any,
    mode: str,
    headers: Dict[str, str],
    timeout: float = 30.0,
) -> RequestResult:
    """
    Send one request to TorchServe and return a RequestResult.
    """
    ts_start = time.time()
    try:
        if mode == "image":
            img_bytes = img_to_bytes(sample)
            r = requests.post(
                f"{url}/predictions/{model_name}",
                data=img_bytes,
                headers=headers,
                timeout=timeout,
            )
        else:
            payload = json.dumps({"text": sample})
            r = requests.post(
                f"{url}/predictions/{model_name}",
                data=payload,
                headers={**headers, "Content-Type": "application/json"},
                timeout=timeout,
            )
        ts_end = time.time()
        latency_ms = (ts_end - ts_start) * 1000.0
        return RequestResult(
            ts_start=ts_start,
            ts_end=ts_end,
            latency_ms=latency_ms,
            status_code=r.status_code,
            error=None if r.status_code == 200 else r.text[:200],
        )
    except Exception as e:
        ts_end = time.time()
        latency_ms = (ts_end - ts_start) * 1000.0
        logging.warning(f"Request failed: {e}")
        return RequestResult(
            ts_start=ts_start,
            ts_end=ts_end,
            latency_ms=latency_ms,
            status_code=0,
            error=str(e),
        )


def load_samples(mode: str, num_samples: int = 100) -> List[Any]:
    if mode == "image":
        logging.info("Loading CIFAR-10 test subset for image classification…")
        dataset = CIFAR10(root="./data", train=False, download=True)
        return [dataset[i][0] for i in range(min(num_samples, len(dataset)))]
    elif mode == "text":
        logging.info("Using hard-coded sentences for text mode…")
        sentences = [
            "I absolutely loved this movie!",
            "This film was a waste of time.",
            "The food at the restaurant was delicious.",
            "I wouldn't recommend this product to anyone.",
            "The book captivated me from beginning to end.",
            "It was the worst experience of my life.",
            "The concert was fantastic and full of energy.",
            "I regret buying this item.",
            "The game kept me entertained for hours.",
            "The service was slow and unsatisfactory.",
            "I feel happy and relaxed after the yoga session.",
            "The car broke down just after a week.",
            "What an incredible performance!",
            "The hotel room was dirty and uncomfortable.",
            "I enjoyed every moment of our vacation.",
            "The software keeps crashing unexpectedly.",
            "The customer support was very helpful and friendly.",
            "It's overpriced and not worth the money.",
            "The lecture was informative and engaging.",
            "I've never been so disappointed with a purchase.",
        ]
        if num_samples > len(sentences):
            repeats = (num_samples + len(sentences) - 1) // len(sentences)
            return (sentences * repeats)[:num_samples]
        return sentences[:num_samples]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def worker_loop(
    name: str,
    url: str,
    model_name: str,
    mode: str,
    headers: Dict[str, str],
    sample_pool: List[Any],
    task_queue: "queue.Queue[Optional[int]]",
    result_list: List[RequestResult],
):
    """
    Worker thread: consumes "tokens" from task_queue and performs that many requests.
    Each token represents 1 request to send now.
    A token of None is a sentinel telling the worker to exit cleanly.
    """
    while True:
        token = task_queue.get()
        try:
            if token is None:
                # Sentinel: shut down this worker
                return
            if token <= 0:
                continue

            sample = random.choice(sample_pool)
            result = send_request(url, model_name, sample, mode, headers)
            result_list.append(result)
        finally:
            task_queue.task_done()


def schedule_burst_pattern(
    duration: int,
    burst: int,
    idle: int,
    base_rps: int,
    task_queue: "queue.Queue[Optional[int]]",
):
    """
    Bursty pattern: alternate randomised burst + idle periods, enqueueing tokens.
    """
    start = time.time()
    while time.time() - start < duration:
        burst_dur = max(5, int(random.gauss(burst, max(1.0, burst * 0.2))))
        idle_dur = max(2, int(random.gauss(idle, max(1.0, idle * 0.3))))
        req_per_sec = max(1, int(random.gauss(base_rps, max(1.0, base_rps * 0.25))))
        logging.info(f"💥 Burst for {burst_dur}s at {req_per_sec} req/s")
        burst_start = time.time()

        while time.time() - burst_start < burst_dur and time.time() - start < duration:
            for _ in range(req_per_sec):
                task_queue.put(1)
            time.sleep(1.0)

        if time.time() - start >= duration:
            break
        logging.info(f"😴 Idle for {idle_dur}s")
        time.sleep(idle_dur)


def schedule_steady_pattern(
    duration: int,
    rps: int,
    task_queue: "queue.Queue[Optional[int]]",
):
    """
    Steady "open-loop" RPS: every 1/rps seconds enqueue 1 request.
    """
    logging.info(f"📈 Steady load: {rps} req/s for {duration}s")
    start = time.time()
    inter_arrival = 1.0 / max(1, rps)
    next_time = start
    while time.time() - start < duration:
        now = time.time()
        if now >= next_time:
            task_queue.put(1)
            next_time += inter_arrival
        else:
            sleep_for = next_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)


def schedule_poisson_pattern(
    duration: int,
    rps: int,
    task_queue: "queue.Queue[Optional[int]]",
):
    """
    Poisson arrivals: inter-arrival ~ Exp(lambda = rps).
    """
    logging.info(f"🎲 Poisson load: avg {rps} req/s for {duration}s")
    start = time.time()
    lam = float(max(1, rps))
    while time.time() - start < duration:
        wait = random.expovariate(lam)
        time.sleep(wait)
        if time.time() - start >= duration:
            break
        task_queue.put(1)


def schedule_pattern(
    pattern: str,
    duration: int,
    rps: int,
    burst: int,
    idle: int,
    task_queue: "queue.Queue[Optional[int]]",
) -> None:
    """
    Helper: choose the right scheduler based on pattern name.
    Used both for single-pattern and multi-phase runs.
    """
    pattern = pattern.lower()
    if pattern == "burst":
        schedule_burst_pattern(duration, burst, idle, rps, task_queue)
    elif pattern == "steady":
        schedule_steady_pattern(duration, rps, task_queue)
    elif pattern == "poisson":
        schedule_poisson_pattern(duration, rps, task_queue)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def run_load(
    url: str,
    mode: str,
    model_name: str,
    duration: int,
    burst: int,
    idle: int,
    rps: int,
    headers: Dict[str, str],
    concurrency: int,
    pattern: TrafficPattern,
    warmup_requests: int,
    csv_path: Optional[str],
    phases: Optional[List[Dict[str, Any]]] = None,
) -> None:
    samples = load_samples(mode)

    # Warmup (not recorded)
    if warmup_requests > 0:
        logging.info(f"🔥 Warmup: sending {warmup_requests} requests (not recorded)")
        for _ in range(warmup_requests):
            sample = random.choice(samples)
            _ = send_request(url, model_name, sample, mode, headers)

    task_queue: "queue.Queue[Optional[int]]" = queue.Queue()
    results: List[RequestResult] = []

    # Start worker threads
    threads: List[threading.Thread] = []
    for i in range(concurrency):
        t = threading.Thread(
            target=worker_loop,
            args=(f"worker-{i}", url, model_name, mode, headers, samples, task_queue, results),
        )
        t.start()
        threads.append(t)

    # Schedule tasks according to pattern or phases
    if phases:
        total_phase_duration = sum(int(p.get("duration", 0)) for p in phases)
        logging.info(
            f"🚦 Running {len(phases)} phases (total duration ~{total_phase_duration}s). "
            "Per-phase settings can override --pattern/--duration/--rps/--burst/--idle."
        )
        for idx, phase in enumerate(phases, start=1):
            phase_pattern = (phase.get("pattern", pattern) or pattern).lower()
            phase_duration = int(phase.get("duration", duration))
            phase_rps = int(phase.get("rps", rps))
            phase_burst = int(phase.get("burst", burst))
            phase_idle = int(phase.get("idle", idle))
            name = phase.get("name", f"phase-{idx}")
            logging.info(
                f"=== Phase {idx}/{len(phases)}: {name} | "
                f"pattern={phase_pattern}, duration={phase_duration}s, "
                f"rps={phase_rps}, burst={phase_burst}, idle={phase_idle} ==="
            )
            schedule_pattern(
                pattern=phase_pattern,
                duration=phase_duration,
                rps=phase_rps,
                burst=phase_burst,
                idle=phase_idle,
                task_queue=task_queue,
            )
    else:
        schedule_pattern(
            pattern=pattern,
            duration=duration,
            rps=rps,
            burst=burst,
            idle=idle,
            task_queue=task_queue,
        )

    # Send sentinel None to tell workers to exit once queue is empty
    for _ in range(concurrency):
        task_queue.put(None)

    # Wait for all work (including sentinels) to be processed
    task_queue.join()

    # Join threads
    for t in threads:
        t.join()

    if not results:
        logging.warning("No successful requests recorded (or results list empty).")
        return

    # Compute stats like JMeter/AB style
    latencies = [r.latency_ms for r in results if r.status_code == 200]
    errors = [r for r in results if r.status_code != 200]
    total = len(results)
    window = max(r.ts_end for r in results) - min(r.ts_start for r in results)
    throughput = total / window if window > 0 else float("nan")

    if latencies:
        avg_latency = statistics.mean(latencies)
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=100)[94]
        p99 = statistics.quantiles(latencies, n=100)[98]
    else:
        avg_latency = p50 = p95 = p99 = float("nan")

    error_rate = len(errors) * 100.0 / total

    logging.info(
        "Completed load test: "
        f"requests={total}, throughput={throughput:.2f} req/s, "
        f"avg={avg_latency:.2f}ms, p50={p50:.2f}ms, p95={p95:.2f}ms, "
        f"p99={p99:.2f}ms, error%={error_rate:.2f}"
    )

    # Optional CSV output in a style similar-ish to ab_report/jmeter
    if csv_path:
        logging.info(f"📝 Writing per-request metrics to {csv_path}")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "ts_start",
                    "ts_end",
                    "latency_ms",
                    "status_code",
                    "error",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        f"{r.ts_start:.6f}",
                        f"{r.ts_end:.6f}",
                        f"{r.latency_ms:.3f}",
                        r.status_code,
                        r.error or "",
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TorchServe load generator (randomised + benchmark-style)."
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Base URL of TorchServe (default: http://$HOSTNAME:8080)",
    )
    parser.add_argument(
        "--mode",
        choices=["image", "text"],
        default="image",
        help="Service type to test",
    )
    parser.add_argument(
        "--model-name",
        default="resnet-18",
        help="TorchServe model name (default: resnet-18)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Test duration (s) if not using --phases-json",
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=30,
        help="Mean burst duration (s) for burst pattern",
    )
    parser.add_argument(
        "--idle",
        type=int,
        default=15,
        help="Mean idle duration (s) for burst pattern",
    )
    parser.add_argument(
        "--rps",
        type=int,
        default=10,
        help="Target requests per second (pattern-dependent)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of worker threads (like JMeter/AB concurrency)",
    )
    parser.add_argument(
        "--pattern",
        choices=["burst", "steady", "poisson"],
        default="burst",
        help="Traffic pattern: burst (microservice-like), steady, or poisson.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=50,
        help="Warmup requests before measurement",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to write per-request CSV metrics (optional)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Auth token (optional)",
    )
    parser.add_argument(
        "--phases-json",
        default=None,
        help=(
            "Optional path to a JSON file describing multiple load phases. "
            "If set, runs all phases sequentially and overrides "
            "--pattern/--duration/--rps/--burst/--idle."
        ),
    )

    args = parser.parse_args()

    url = args.url or f"http://{os.environ.get('HOSTNAME', '127.0.0.1')}:8080"
    headers: Dict[str, str] = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    phases = None
    if args.phases_json:
        with open(args.phases_json) as f:
            phases = json.load(f)

    run_load(
        url=url,
        mode=args.mode,
        model_name=args.model_name,
        duration=args.duration,
        burst=args.burst,
        idle=args.idle,
        rps=args.rps,
        headers=headers,
        concurrency=args.concurrency,
        pattern=args.pattern,  # type: ignore[arg-type]
        warmup_requests=args.warmup_requests,
        csv_path=args.csv,
        phases=phases,
    )


if __name__ == "__main__":
    main()
