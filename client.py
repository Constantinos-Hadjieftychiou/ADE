#!/usr/bin/env python3

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
import zipfile
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


UCI_SENTIMENT_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip"
)
UCI_SENTENCE_CACHE = os.path.join("data", "uci_sentiment_sentences.txt")


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
    ts_start = time.time()
    try:
        if mode == "image":
            # Allow samples to be either pre-encoded bytes or PIL Images
            if isinstance(sample, (bytes, bytearray)):
                img_bytes = sample
            else:
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


def _download_uci_sentences(cache_path: str = UCI_SENTENCE_CACHE) -> List[str]:
    try:
        logging.info("Downloading UCI Sentiment Labelled Sentences dataset")
        resp = requests.get(UCI_SENTIMENT_ZIP_URL, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logging.warning(f"Failed to download UCI sentiment dataset: {e}")
        return []

    try:
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
    except Exception as e:
        logging.warning(f"Failed to open UCI sentiment ZIP: {e}")
        return []

    wanted_files = (
        "imdb_labelled.txt",
        "amazon_cells_labelled.txt",
        "yelp_labelled.txt",
    )

    sentences: List[str] = []

    for fname in wanted_files:
        if fname not in zf.namelist():
            logging.warning(f"File {fname} not found in UCI sentiment ZIP")
            continue

        try:
            with zf.open(fname) as f:
                for raw_line in f:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if not parts:
                        continue
                    sentence = parts[0].strip()
                    if sentence:
                        sentences.append(sentence)
        except Exception as e:
            logging.warning(f"Failed to parse {fname} from UCI sentiment ZIP: {e}")

    if not sentences:
        logging.warning("No sentences extracted from UCI sentiment dataset")
        return []

    seen = set()
    unique_sentences: List[str] = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    try:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            for s in unique_sentences:
                f.write(s + "\n")
        logging.info(f"Cached {len(unique_sentences)} text samples to {cache_path}")
    except Exception as e:
        logging.warning(f"Failed to write UCI sentence cache file {cache_path}: {e}")

    return unique_sentences


def load_text_samples(num_samples: int = 100) -> List[str]:
    sentences: List[str] = []

    if os.path.exists(UCI_SENTENCE_CACHE):
        logging.info(f"Loading text samples from cache: {UCI_SENTENCE_CACHE}")
        try:
            with open(UCI_SENTENCE_CACHE, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.warning(f"Failed to read cached text samples: {e}")
            sentences = []

    if not sentences:
        sentences = _download_uci_sentences(cache_path=UCI_SENTENCE_CACHE)

    if not sentences:
        logging.warning("Using small built-in list of text samples")
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

    if num_samples <= 0:
        return []

    if num_samples > len(sentences):
        repeats = (num_samples + len(sentences) - 1) // len(sentences)
        return (sentences * repeats)[:num_samples]

    return sentences[:num_samples]


def load_samples(mode: str, num_samples: int = 100) -> List[Any]:
    if mode == "image":
        logging.info("Loading CIFAR-10 test subset")
        dataset = CIFAR10(root="./data", train=False, download=True)

        # Pre-encode to JPEG bytes so the hot path doesn't spend CPU time
        # repeatedly encoding images.
        max_samples = min(num_samples, len(dataset))
        samples: List[bytes] = []
        for i in range(max_samples):
            img = dataset[i][0]
            samples.append(img_to_bytes(img))
        return samples
    elif mode == "text":
        return load_text_samples(num_samples=num_samples)
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
    result_lock: threading.Lock,
):
    while True:
        token = task_queue.get()
        try:
            if token is None:
                return
            if token <= 0:
                continue

            sample = random.choice(sample_pool)
            result = send_request(url, model_name, sample, mode, headers)
            # Protect list appends so we don't corrupt the results under
            # very high concurrency.
            with result_lock:
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
    if base_rps <= 0:
        logging.info(f"Burst pattern with base_rps={base_rps}: idle for {duration}s")
        time.sleep(duration)
        return

    start = time.time()
    while time.time() - start < duration:
        burst_dur = max(5, int(random.gauss(burst, max(1.0, burst * 0.2))))
        idle_dur = max(2, int(random.gauss(idle, max(1.0, idle * 0.3))))
        req_per_sec = max(1, int(random.gauss(base_rps, max(1.0, base_rps * 0.25))))

        logging.info(f"Burst for {burst_dur}s at ~{req_per_sec} req/s")
        burst_start = time.time()

        while time.time() - burst_start < burst_dur and time.time() - start < duration:
            for _ in range(req_per_sec):
                task_queue.put(1)
            time.sleep(1.0)

        if time.time() - start >= duration:
            break

        logging.info(f"Idle for {idle_dur}s")
        time.sleep(idle_dur)


def schedule_steady_pattern(
    duration: int,
    rps: int,
    task_queue: "queue.Queue[Optional[int]]",
):
    if rps <= 0:
        logging.info(f"Steady load with rps={rps}: idle for {duration}s")
        time.sleep(duration)
        return

    logging.info(f"Steady load: {rps} req/s for {duration}s")
    start = time.time()
    inter_arrival = 1.0 / float(rps)
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
    if rps <= 0:
        logging.info(f"Poisson load with rps={rps}: idle for {duration}s")
        time.sleep(duration)
        return

    logging.info(f"Poisson load: avg {rps} req/s for {duration}s")
    start = time.time()
    lam = float(rps)

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
    pattern = pattern.lower()
    if pattern == "burst":
        schedule_burst_pattern(duration, burst, idle, rps, task_queue)
    elif pattern == "steady":
        schedule_steady_pattern(duration, rps, task_queue)
    elif pattern == "poisson":
        schedule_poisson_pattern(duration, rps, task_queue)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def aggregate_window_metrics(csv_path: str, window_s: float = 1.0) -> Optional[str]:
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        logging.warning("pandas/numpy not available; skipping window aggregation")
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.warning(f"Failed to read CSV {csv_path}: {e}")
        return None

    if df.empty:
        logging.warning(f"No rows in {csv_path}; skipping window aggregation")
        return None

    required_cols = {"ts_start", "ts_end", "latency_ms", "status_code"}
    if not required_cols.issubset(df.columns):
        logging.warning(
            f"{csv_path} missing required columns {required_cols}; "
            "skipping window aggregation"
        )
        return None

    t_min = df["ts_start"].min()
    t_max = df["ts_start"].max()
    if t_min is None or (isinstance(t_min, float) and (t_min != t_min)):
        logging.warning("ts_start has no valid values; skipping window aggregation")
        return None

    df["window_index"] = np.floor((df["ts_start"] - t_min) / window_s).astype(int)
    grouped = df.groupby("window_index")

    window_stats = grouped.agg(
        requests_started=("ts_start", "count"),
        avg_latency_ms=("latency_ms", "mean"),
        p50_latency_ms=("latency_ms", "median"),
        error_rate=("status_code", lambda s: (s != 200).mean() * 100.0),
    )

    max_index = int(np.floor((t_max - t_min) / window_s))
    full_index = np.arange(0, max_index + 1, dtype=int)
    window_stats = window_stats.reindex(full_index)

    window_stats["requests_started"] = window_stats["requests_started"].fillna(0).astype(int)
    window_stats["error_rate"] = window_stats["error_rate"].fillna(0.0)

    window_stats = window_stats.reset_index().rename(columns={"index": "window_index"})
    window_stats["window_start_ts"] = t_min + window_stats["window_index"] * window_s

    try:
        window_stats["window_start_dt"] = pd.to_datetime(
            window_stats["window_start_ts"], unit="s"
        )
    except Exception:
        pass

    window_stats["is_idle"] = window_stats["requests_started"] == 0
    window_stats["idle_label"] = window_stats["is_idle"].astype(int)

    out_path = csv_path.replace(".csv", f"_windows_{int(window_s)}s.csv")
    try:
        window_stats.to_csv(out_path, index=False)
        logging.info(f"Wrote window-level metrics to {out_path}")
        return out_path
    except Exception as e:
        logging.warning(f"Failed to write window stats CSV {out_path}: {e}")
        return None


def attach_energy_to_windows(
    windows_csv_path: str,
    energy_csv_path: str,
    energy_ts_col: str = "timestamp",
    energy_col: str = "energy_j",
    idle_power_threshold_w: Optional[float] = None,
) -> None:
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        logging.warning("pandas/numpy not available; skipping energy/window join")
        return

    if not os.path.exists(windows_csv_path):
        logging.warning(
            f"Window metrics CSV {windows_csv_path} not found; cannot attach energy"
        )
        return

    if not os.path.exists(energy_csv_path):
        logging.warning(
            f"Energy CSV {energy_csv_path} not found; skipping energy/window join"
        )
        return

    try:
        windows = pd.read_csv(windows_csv_path)
        energy = pd.read_csv(energy_csv_path)
    except Exception as e:
        logging.warning(f"Failed to read windows/energy CSVs: {e}")
        return

    if windows.empty or energy.empty:
        logging.warning("Windows or energy CSV is empty; skipping join")
        return

    if "window_start_ts" not in windows.columns:
        logging.warning(
            f"{windows_csv_path} missing window_start_ts; skipping energy/window join"
        )
        return

    if energy_ts_col not in energy.columns or energy_col not in energy.columns:
        logging.warning(
            f"{energy_csv_path} missing required columns "
            f"({energy_ts_col}, {energy_col}); skipping energy/window join"
        )
        return

    windows["second"] = windows["window_start_ts"].round().astype("int64")
    energy["second"] = energy[energy_ts_col].round().astype("int64")

    energy_agg = (
        energy.groupby("second", as_index=False)[energy_col]
        .sum()
        .rename(columns={energy_col: "energy_j_per_window"})
    )

    merged = windows.merge(
        energy_agg,
        on="second",
        how="left",
        validate="m:1",
    )

    merged["energy_per_request"] = (
        merged["energy_j_per_window"] /
        merged["requests_started"].clip(lower=1)
    )
    merged.loc[merged["requests_started"] == 0, "energy_per_request"] = np.nan

    if "energy_j_per_window" in merged.columns:
        try:
            corr_df = merged[["requests_started", "energy_j_per_window"]].dropna()
            if not corr_df.empty:
                corr = corr_df.corr().iloc[0, 1]
                logging.info(
                    f"Corr(requests_started, energy_j_per_window) = {corr:.4f}"
                )
        except Exception as e:
            logging.warning(f"Failed to compute load/energy correlation: {e}")

    if "idle_label" in merged.columns and "energy_j_per_window" in merged.columns:
        try:
            idle_energy = merged.loc[merged["idle_label"] == 1, "energy_j_per_window"].dropna()
            busy_energy = merged.loc[merged["idle_label"] == 0, "energy_j_per_window"].dropna()
            if not idle_energy.empty and not busy_energy.empty:
                logging.info(
                    "Idle vs busy energy: "
                    f"idle_mean={idle_energy.mean():.2f}J, "
                    f"busy_mean={busy_energy.mean():.2f}J"
                )
        except Exception as e:
            logging.warning(f"Failed to compute idle/busy energy stats: {e}")

    if idle_power_threshold_w is not None and "energy_j_per_window" in merged.columns:
        try:
            mask_valid = merged["energy_j_per_window"].notna()
            energy_idle = merged["energy_j_per_window"] <= idle_power_threshold_w
            merged["energy_idle_label"] = np.where(
                mask_valid,
                energy_idle.astype(int),
                np.nan,
            )
            n_idle = int((merged["energy_idle_label"] == 1).sum())
            n_total = int(mask_valid.sum())
            logging.info(
                "Energy-idle threshold %.2fW: %d/%d windows flagged idle",
                idle_power_threshold_w,
                n_idle,
                n_total,
            )
        except Exception as e:
            logging.warning(f"Failed to compute energy_idle_label: {e}")

    try:
        merged.drop(columns=["second"], inplace=True)
    except KeyError:
        pass

    try:
        merged.to_csv(windows_csv_path, index=False)
        logging.info(
            f"Attached energy features to window metrics in {windows_csv_path}"
        )
    except Exception as e:
        logging.warning(f"Failed to write energy-augmented CSV {windows_csv_path}: {e}")


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
    energy_csv_path: Optional[str] = None,
    phases_total_seconds: Optional[int] = None,
    idle_power_threshold_w: Optional[float] = None,
) -> None:
    samples = load_samples(mode)

    if warmup_requests > 0:
        logging.info(f"Warmup: sending {warmup_requests} requests (not recorded)")
        for _ in range(warmup_requests):
            sample = random.choice(samples)
            _ = send_request(url, model_name, sample, mode, headers)

    task_queue: "queue.Queue[Optional[int]]" = queue.Queue()
    results: List[RequestResult] = []
    results_lock = threading.Lock()

    threads: List[threading.Thread] = []
    for i in range(concurrency):
        t = threading.Thread(
            target=worker_loop,
            args=(
                f"worker-{i}",
                url,
                model_name,
                mode,
                headers,
                samples,
                task_queue,
                results,
                results_lock,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    if phases:
        phases_list = list(phases)
        base_total_phase_duration = sum(int(p.get("duration", 0)) for p in phases_list)

        if phases_total_seconds is not None and phases_total_seconds > 0:
            logging.info(
                f"Running phases up to ~{phases_total_seconds}s "
                f"(one pass ~{base_total_phase_duration}s)"
            )

            total_scheduled = 0
            round_idx = 0

            while total_scheduled < phases_total_seconds:
                round_idx += 1
                random.shuffle(phases_list)
                for phase in phases_list:
                    if total_scheduled >= phases_total_seconds:
                        break

                    phase_pattern = (phase.get("pattern", pattern) or pattern).lower()
                    orig_phase_duration = int(phase.get("duration", duration))
                    remaining = phases_total_seconds - total_scheduled
                    phase_duration = max(1, min(orig_phase_duration, remaining))
                    phase_rps = int(phase.get("rps", rps))
                    phase_burst = int(phase.get("burst", burst))
                    phase_idle = int(phase.get("idle", idle))
                    name = phase.get("name", f"phase-round{round_idx}")

                    logging.info(
                        f"Phase round {round_idx}: {name}, "
                        f"pattern={phase_pattern}, duration={phase_duration}s, "
                        f"rps={phase_rps}, burst={phase_burst}, idle={phase_idle}"
                    )
                    schedule_pattern(
                        pattern=phase_pattern,
                        duration=phase_duration,
                        rps=phase_rps,
                        burst=phase_burst,
                        idle=phase_idle,
                        task_queue=task_queue,
                    )
                    total_scheduled += phase_duration
        else:
            total_phase_duration = base_total_phase_duration
            random.shuffle(phases_list)
            logging.info(
                f"Running {len(phases_list)} phases once in random order "
                f"(total duration ~{total_phase_duration}s)"
            )

            for idx, phase in enumerate(phases_list, start=1):
                phase_pattern = (phase.get("pattern", pattern) or pattern).lower()
                phase_duration = int(phase.get("duration", duration))
                phase_rps = int(phase.get("rps", rps))
                phase_burst = int(phase.get("burst", burst))
                phase_idle = int(phase.get("idle", idle))
                name = phase.get("name", f"phase-{idx}")
                logging.info(
                    f"Phase {idx}/{len(phases_list)}: {name}, "
                    f"pattern={phase_pattern}, duration={phase_duration}s, "
                    f"rps={phase_rps}, burst={phase_burst}, idle={phase_idle}"
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

    for _ in range(concurrency):
        task_queue.put(None)

    task_queue.join()

    for t in threads:
        t.join()

    if not results:
        logging.warning("No successful requests recorded (results list is empty)")
        return

    latencies = [r.latency_ms for r in results if r.status_code == 200]
    errors = [r for r in results if r.status_code != 200]
    total = len(results)
    window = max(r.ts_end for r in results) - min(r.ts_start for r in results)
    throughput = total / window if window > 0 else float("nan")

    if latencies:
        latencies_sorted = sorted(latencies)
        avg_latency = statistics.mean(latencies_sorted)
        p50 = statistics.median(latencies_sorted)
        n = len(latencies_sorted)
        p95 = latencies_sorted[int(0.95 * (n - 1))]
        p99 = latencies_sorted[int(0.99 * (n - 1))]
    else:
        avg_latency = p50 = p95 = p99 = float("nan")

    error_rate = len(errors) * 100.0 / total

    logging.info(
        "Completed load test: "
        f"requests={total}, throughput={throughput:.2f} req/s, "
        f"avg={avg_latency:.2f}ms, p50={p50:.2f}ms, p95={p95:.2f}ms, "
        f"p99={p99:.2f}ms, error%={error_rate:.2f}"
    )

    if csv_path:
        logging.info(f"Writing per-request metrics to {csv_path}")
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        try:
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
        except Exception as e:
            logging.warning(f"Failed to write per-request CSV {csv_path}: {e}")
            return

        windows_csv_path = aggregate_window_metrics(csv_path, window_s=1.0)

        if windows_csv_path and energy_csv_path:
            attach_energy_to_windows(
                windows_csv_path=windows_csv_path,
                energy_csv_path=energy_csv_path,
                idle_power_threshold_w=idle_power_threshold_w,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TorchServe load generator"
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
        help="Service type to test: 'image' or 'text'",
    )
    parser.add_argument(
        "--model-name",
        default="resnet-18",
        help="TorchServe model name",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Test duration in seconds if not using --phases-json",
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=30,
        help="Mean burst duration (s) for 'burst' pattern",
    )
    parser.add_argument(
        "--idle",
        type=int,
        default=15,
        help="Mean idle duration (s) for 'burst' pattern",
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
        help="Number of worker threads",
    )
    parser.add_argument(
        "--pattern",
        choices=["burst", "steady", "poisson"],
        default="burst",
        help="Traffic pattern: burst, steady, or poisson",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=50,
        help="Warmup requests to send before measurement starts",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to write per-request CSV metrics (optional)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Auth token for TorchServe (optional)",
    )
    parser.add_argument(
        "--phases-json",
        default=None,
        help="Path to a JSON file describing multiple load phases",
    )
    parser.add_argument(
        "--phase-name",
        default=None,
        help=(
            "Optional: when used with --phases-json, select a single phase "
            "by its 'name'. If not set, all phases are used."
        ),
    )
    parser.add_argument(
        "--energy-csv",
        default=None,
        help=(
            "Optional path to a per-second energy CSV with columns "
            "'timestamp' and 'energy_j'."
        ),
    )
    parser.add_argument(
        "--phases-total-seconds",
        type=int,
        default=None,
        help=(
            "When used with --phases-json, sets an approximate total number "
            "of seconds of scheduled phase time. Phases will be shuffled and "
            "repeated until this budget is exhausted."
        ),
    )
    parser.add_argument(
        "--idle-power-threshold-w",
        type=float,
        default=None,
        help=(
            "Optional energy-based idle threshold in watts. When set and "
            "--energy-csv is provided, windows with energy_j_per_window "
            "less than or equal to this are marked with energy_idle_label=1."
        ),
    )
    parser.add_argument(
        "--windows-csv",
        default=None,
        help=(
            "Path to an existing window-level metrics CSV to post-process "
            "with energy data (used with --attach-energy-only)."
        ),
    )
    parser.add_argument(
        "--attach-energy-only",
        action="store_true",
        help=(
            "Skip load generation and only attach energy data to an existing "
            "windows CSV. Requires --windows-csv and --energy-csv."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible load patterns.",
    )

    args = parser.parse_args()

    # Make load patterns (and sampling) reproducible when requested.
    if args.random_seed is not None:
        random.seed(args.random_seed)
        try:
            import numpy as _np
            _np.random.seed(args.random_seed)
        except Exception:
            pass
        try:
            import torch as _torch  # type: ignore[import]
            _torch.manual_seed(args.random_seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(args.random_seed)
        except Exception:
            # Torch not available or CUDA not usable; ignore
            pass

    # Post-processing-only mode: just attach energy to an existing windows CSV
    if args.attach_energy_only:
        if not args.windows_csv:
            raise SystemExit("--attach-energy-only requires --windows-csv")
        if not args.energy_csv:
            raise SystemExit("--attach-energy-only requires --energy-csv")

        attach_energy_to_windows(
            windows_csv_path=args.windows_csv,
            energy_csv_path=args.energy_csv,
            idle_power_threshold_w=args.idle_power_threshold_w,
        )
        return

    url = args.url or f"http://{os.environ.get('HOSTNAME', '127.0.0.1')}:8080"

    headers: Dict[str, str] = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    phases = None
    if args.phases_json:
        with open(args.phases_json) as f:
            phases = json.load(f)

        if args.phase_name:
            filtered = [p for p in phases if p.get("name") == args.phase_name]
            if not filtered:
                raise SystemExit(
                    f"No phase named '{args.phase_name}' found in {args.phases_json}"
                )
            phases = filtered

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
        energy_csv_path=args.energy_csv,
        phases_total_seconds=args.phases_total_seconds,
        idle_power_threshold_w=args.idle_power_threshold_w,
    )


if __name__ == "__main__":
    main()
