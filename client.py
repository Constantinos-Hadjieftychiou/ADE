#!/usr/bin/env python3
"""
client.py — Randomised load generator for TorchServe on GPU microservices.

Sends bursty traffic to a TorchServe model:
  - image mode: POST /predictions/resnet-18 with JPEG bytes
"""

import argparse
import io
import json
import logging
import os
import random
import statistics
import time
from typing import List, Any, Dict, Optional

import requests
from PIL import Image
from torchvision.datasets import CIFAR10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


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
) -> Optional[float]:
    """
    Send one request to TorchServe and return latency in ms (or None on error).
    """
    try:
        start = time.perf_counter()
        if mode == "image":
            img_bytes = img_to_bytes(sample)
            r = requests.post(
                f"{url}/predictions/{model_name}",
                data=img_bytes,
                headers=headers,
                timeout=30,
            )
        else:
            # placeholder for future text model, if you add one
            payload = json.dumps({"text": sample})
            r = requests.post(
                f"{url}/predictions/{model_name}",
                data=payload,
                headers={**headers, "Content-Type": "application/json"},
                timeout=30,
            )
        if r.status_code == 200:
            return (time.perf_counter() - start) * 1000.0
        else:
            logging.warning(f"Bad response: {r.status_code} | {r.text[:120]}")
    except Exception as e:
        logging.warning(f"Request failed: {e}")
    return None


def load_samples(mode: str, num_samples: int = 100) -> List[Any]:
    if mode == "image":
        logging.info("Loading CIFAR-10 test subset for image classification…")
        dataset = CIFAR10(root="./data", train=False, download=True)
        return [dataset[i][0] for i in range(min(num_samples, len(dataset)))]
    elif mode == "text":
        # You can plug in a TorchServe text model later
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


def run_random_load(
    url: str,
    mode: str,
    model_name: str,
    duration: int,
    burst: int,
    idle: int,
    rps: int,
    headers: Dict[str, str],
) -> None:
    samples = load_samples(mode)
    start = time.time()
    latencies: List[float] = []
    num_requests = 0

    while time.time() - start < duration:
        burst_dur = max(5, int(random.gauss(burst, max(1.0, burst * 0.2))))
        idle_dur = max(2, int(random.gauss(idle, max(1.0, idle * 0.3))))
        req_per_sec = max(1, int(random.gauss(rps, max(1.0, rps * 0.25))))
        logging.info(f"💥 Burst for {burst_dur}s at {req_per_sec} req/s")
        burst_start = time.time()

        while time.time() - burst_start < burst_dur and time.time() - start < duration:
            batch = random.choices(samples, k=req_per_sec)
            for sample in batch:
                latency = send_request(url, model_name, sample, mode, headers)
                if latency is not None:
                    latencies.append(latency)
                num_requests += 1
            time.sleep(1)

        if time.time() - start >= duration:
            break
        logging.info(f"😴 Idle for {idle_dur}s")
        time.sleep(idle_dur)

    if latencies:
        avg_latency = statistics.mean(latencies)
        p95 = statistics.quantiles(latencies, n=100)[94]
        p99 = statistics.quantiles(latencies, n=100)[98]
        logging.info(
            f"Completed load test: requests={num_requests}, "
            f"avg_latency={avg_latency:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms"
        )
    else:
        logging.warning("No successful requests recorded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomised load generator for TorchServe.")
    parser.add_argument("--url", default=None, help="Base URL of TorchServe (default: http://$HOSTNAME:8080)")
    parser.add_argument("--mode", choices=["image", "text"], default="image", help="Service type to test")
    parser.add_argument("--model-name", default="resnet-18", help="TorchServe model name (default: resnet-18)")
    parser.add_argument("--duration", type=int, default=300, help="Test duration (s)")
    parser.add_argument("--burst", type=int, default=30, help="Mean burst duration (s)")
    parser.add_argument("--idle", type=int, default=15, help="Mean idle duration (s)")
    parser.add_argument("--rps", type=int, default=10, help="Mean requests per second during bursts")
    parser.add_argument("--token", default=None, help="Auth token (optional)")
    args = parser.parse_args()

    url = args.url or f"http://{os.environ.get('HOSTNAME', '127.0.0.1')}:8080"
    headers: Dict[str, str] = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    run_random_load(url, args.mode, args.model_name, args.duration, args.burst, args.idle, args.rps, headers)


if __name__ == "__main__":
    main()
