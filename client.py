#!/usr/bin/env python3
"""
TorchServe load generator with NVML power sampling and window-level metrics aggregation.

This script:
1. Loads sample data (CIFAR-10 images or UCI sentiment text)
2. Generates traffic against a TorchServe inference endpoint using configurable patterns
3. Samples GPU power via NVML in a background thread
4. Aggregates per-request metrics into fixed time windows
5. Optionally attaches energy data to windows for idle detection
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
import zipfile
from dataclasses import dataclass
from typing import List, Any, Dict, Optional, Literal

import requests
from PIL import Image
from torchvision.datasets import CIFAR10

# Configure logging for all output (timestamp + level + message)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Type hint for traffic pattern selection
TrafficPattern = Literal["burst", "steady", "poisson"]


@dataclass
class RequestResult:
    """Container for a single request's metrics: start/end time, latency, status, error."""
    ts_start: float
    ts_end: float
    latency_ms: float
    status_code: int
    error: Optional[str]


# URL and cache path for UCI sentiment dataset (for text mode)
UCI_SENTIMENT_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip"
)
UCI_SENTENCE_CACHE = os.path.join("data", "uci_sentiment_sentences.txt")


def img_to_bytes(img: Image.Image) -> bytes:
    """Convert PIL Image to JPEG bytes for HTTP POST."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_window_suffix(window_s: float) -> str:
    """
    Create window suffix for CSV filenames (e.g., 0.5s -> "0p5s").
    Used to distinguish window CSVs by their aggregation size.
    """
    suffix = f"{float(window_s):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return suffix


def send_request(
    url: str,
    model_name: str,
    sample: Any,
    mode: str,
    headers: Dict[str, str],
    timeout: float = 30.0,
) -> RequestResult:
    """
    Send a single request to TorchServe and measure latency.
    
    Handles both image (POST binary JPEG) and text (POST JSON) modes.
    Returns RequestResult with timing and status info.
    """
    ts_start = time.time()
    try:
        if mode == "image":
            # Convert sample to JPEG bytes if needed
            if isinstance(sample, (bytes, bytearray)):
                img_bytes = sample
            else:
                img_bytes = img_to_bytes(sample)

            # POST to /predictions/<model_name> with image data
            r = requests.post(
                f"{url}/predictions/{model_name}",
                data=img_bytes,
                headers=headers,
                timeout=timeout,
            )
        else:  # text mode
            # Wrap text sample in JSON payload
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
        # Capture network/timeout errors as failed requests
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
    """Download UCI sentiment dataset ZIP and extract unique sentences."""
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

    # Extract sentences from multiple sources in the ZIP
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
                # Each line: "<sentence>\t<label>" (tab-separated)
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

    # Remove duplicates while preserving insertion order
    seen = set()
    unique_sentences: List[str] = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    # Cache to disk for future runs
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
    """Load text samples: try cache first, then download, then fallback to built-in."""
    sentences: List[str] = []

    # Try to load from cache
    if os.path.exists(UCI_SENTENCE_CACHE):
        logging.info(f"Loading text samples from cache: {UCI_SENTENCE_CACHE}")
        try:
            with open(UCI_SENTENCE_CACHE, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.warning(f"Failed to read cached text samples: {e}")
            sentences = []

    # If cache miss, download
    if not sentences:
        sentences = _download_uci_sentences(cache_path=UCI_SENTENCE_CACHE)

    # If download fails, use built-in fallback
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

    # Handle edge cases
    if num_samples <= 0:
        return []

    # Repeat sentences if needed to reach num_samples
    if num_samples > len(sentences):
        repeats = (num_samples + len(sentences) - 1) // len(sentences)
        return (sentences * repeats)[:num_samples]

    return sentences[:num_samples]


def load_samples(mode: str, num_samples: int = 100) -> List[Any]:
    """Load dataset samples (images or text) based on mode."""
    if mode == "image":
        logging.info("Loading CIFAR-10 test subset")
        dataset = CIFAR10(root="./data", train=False, download=True)

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
    """
    Worker thread that pulls requests from task_queue and sends them to TorchServe.
    
    Stops when receiving None token. Appends results to shared result_list (thread-safe).
    """
    while True:
        token = task_queue.get()
        try:
            if token is None:  # Poison pill to stop
                return
            if token <= 0:  # Skip non-positive tokens
                continue

            sample = random.choice(sample_pool)
            result = send_request(url, model_name, sample, mode, headers)
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
    """
    Schedule burst pattern: alternating active bursts and quiet periods.
    
    Durations are randomized around mean values using Gaussian distribution.
    """
    if base_rps <= 0:
        logging.info(f"Burst pattern with base_rps={base_rps}: idle for {duration}s")
        time.sleep(duration)
        return

    start = time.time()
    while time.time() - start < duration:
        # Randomize burst and idle durations around means
        burst_dur = max(5, int(random.gauss(burst, max(1.0, burst * 0.2))))
        idle_dur = max(2, int(random.gauss(idle, max(1.0, idle * 0.3))))
        req_per_sec = max(1, int(random.gauss(base_rps, max(1.0, base_rps * 0.25))))

        logging.info(f"Burst for {burst_dur}s at ~{req_per_sec} req/s")
        burst_start = time.time()

        # Send requests during burst period
        while time.time() - burst_start < burst_dur and time.time() - start < duration:
            for _ in range(req_per_sec):
                task_queue.put(1)  # 1 token = 1 request
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
    """
    Schedule steady pattern: deterministic fixed requests-per-second.
    
    Uses precise inter-arrival timing to maintain constant RPS.
    """
    if rps <= 0:
        logging.info(f"Steady load with rps={rps}: idle for {duration}s")
        time.sleep(duration)
        return

    logging.info(f"Steady load: {rps} req/s for {duration}s")
    start = time.time()
    inter_arrival = 1.0 / float(rps)  # Time between requests
    next_time = start

    while time.time() - start < duration:
        now = time.time()
        if now >= next_time:
            task_queue.put(1)
            next_time += inter_arrival
        else:
            # Sleep until next scheduled time
            sleep_for = next_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)


def schedule_poisson_pattern(
    duration: int,
    rps: int,
    task_queue: "queue.Queue[Optional[int]]",
):
    """
    Schedule Poisson pattern: exponential inter-arrival times (realistic traffic).
    
    Average RPS is maintained but timing is random (exponential distribution).
    """
    if rps <= 0:
        logging.info(f"Poisson load with rps={rps}: idle for {duration}s")
        time.sleep(duration)
        return

    logging.info(f"Poisson load: avg {rps} req/s for {duration}s")
    start = time.time()
    lam = float(rps)  # Lambda parameter for exponential distribution

    while time.time() - start < duration:
        wait = random.expovariate(lam)  # Random wait time (exponential)
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
    """Dispatcher to schedule requests based on chosen pattern."""
    pattern = pattern.lower()
    if pattern == "burst":
        schedule_burst_pattern(duration, burst, idle, rps, task_queue)
    elif pattern == "steady":
        schedule_steady_pattern(duration, rps, task_queue)
    elif pattern == "poisson":
        schedule_poisson_pattern(duration, rps, task_queue)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def aggregate_window_metrics(
    csv_path: str,
    window_s: float = 1.0,
    meta: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Aggregate per-request CSV into fixed-size windows.
    
    Computes window-level metrics:
    - requests_started/finished: request counts
    - latency percentiles (avg, p50)
    - error_rate: % of non-200 responses
    - idle_label: 1 if no requests in window
    
    Outputs: <csv_path>_windows_<window_s>s.csv
    """
    try:
        import pandas as pd
        import numpy as np  # noqa: F401
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

    # Verify required columns exist
    required_cols = {"ts_start", "ts_end", "latency_ms", "status_code"}
    if not required_cols.issubset(df.columns):
        logging.warning(
            f"{csv_path} missing required columns {required_cols}; "
            "skipping window aggregation"
        )
        return None

    # Convert to numeric (coerce errors to NaN)
    for c in ("ts_start", "ts_end", "latency_ms", "status_code"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remove rows with missing ts_start
    df = df.dropna(subset=["ts_start"])
    if df.empty:
        logging.warning("ts_start has no valid values after cleaning; skipping")
        return None

    if window_s <= 0:
        logging.warning(f"Invalid window_s={window_s}; using default 1.0s")
        window_s = 1.0

    t_min = float(df["ts_start"].min())
    t_max = float(df["ts_start"].max())

    # Map each request to its window index
    df["window_index"] = (
        ((df["ts_start"] - t_min) / float(window_s))
        .astype("float64")
        .floordiv(1)
        .astype(int)
    )

    # Aggregate per window
    grouped = df.groupby("window_index")
    window_stats = grouped.agg(
        requests_started=("ts_start", "count"),
        avg_latency_ms=("latency_ms", "mean"),
        p50_latency_ms=("latency_ms", "median"),
        error_rate=("status_code", lambda s: (s != 200).mean() * 100.0),
    )

    # Also count requests that finished in each window
    df_end = df.dropna(subset=["ts_end"]).copy()
    if not df_end.empty:
        df_end["window_index_end"] = (
            ((df_end["ts_end"] - t_min) / float(window_s))
            .astype("float64")
            .floordiv(1)
            .astype(int)
        )
        finished = df_end.groupby("window_index_end").agg(requests_finished=("ts_end", "count"))
        finished.index.name = "window_index"
        window_stats = window_stats.merge(
            finished, left_index=True, right_index=True, how="left"
        )
    else:
        window_stats["requests_finished"] = 0

    # Fill missing windows (no requests that period) with zeros
    max_index = int(((t_max - t_min) / float(window_s)) // 1)
    full_index = range(0, max_index + 1)
    window_stats = window_stats.reindex(full_index)

    window_stats["requests_started"] = window_stats["requests_started"].fillna(0).astype(int)
    window_stats["requests_finished"] = window_stats["requests_finished"].fillna(0).astype(int)
    window_stats["error_rate"] = window_stats["error_rate"].fillna(0.0)

    # Reset index and compute derived columns
    window_stats = window_stats.reset_index(names="window_index")
    window_stats["window_start_ts"] = t_min + window_stats["window_index"] * float(window_s)

    # Optional: convert to datetime
    try:
        window_stats["window_start_dt"] = pd.to_datetime(
            window_stats["window_start_ts"], unit="s"
        )
    except Exception:
        pass

    # Label windows as idle if no requests
    window_stats["is_idle"] = window_stats["requests_started"] == 0
    window_stats["idle_label"] = window_stats["is_idle"].astype(int)
    window_stats["label_idle_gt"] = window_stats["idle_label"]

    window_stats["window_s"] = float(window_s)
    window_stats["rps"] = window_stats["requests_started"] / float(window_s)

    # Attach metadata (e.g., model_name, pattern, concurrency)
    if meta is not None:
        for k, v in meta.items():
            window_stats[str(k)] = v

    # Write output
    suffix = make_window_suffix(window_s)
    out_path = csv_path.replace(".csv", f"_windows_{suffix}s.csv")

    try:
        window_stats.to_csv(out_path, index=False)
        logging.info(f"Wrote window-level metrics to {out_path}")
        return out_path
    except Exception as e:
        logging.warning(f"Failed to write window stats CSV {out_path}: {e}")
        return None


def power_sampler_loop(
    csv_path: str,
    sample_period_s: float,
    device_index: int,
    stop_event: threading.Event,
) -> None:
    """
    Background thread that periodically samples GPU power via NVML.
    
    Writes: timestamp, power_w, energy_j (≈ power * sample_period_s)
    Stops when stop_event is set.
    """
    try:
        import pynvml  # type: ignore[import]
    except Exception as e:
        logging.warning("pynvml/NVML not available; power sampling disabled: %s", e)
        return

    try:
        pynvml.nvmlInit()
    except Exception as e:
        logging.warning("Failed to initialize NVML; power sampling disabled: %s", e)
        return

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception as e:
        logging.warning(
            "Failed to get NVML handle for device index %d: %s", device_index, e
        )
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return

    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    try:
        f = open(csv_path, "w", newline="")
    except Exception as e:
        logging.warning("Failed to open power CSV %s: %s", csv_path, e)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        return

    # Write CSV header
    writer = csv.DictWriter(f, fieldnames=["timestamp", "power_w", "energy_j"])
    writer.writeheader()
    f.flush()

    logging.info(
        "Starting power sampler: device_index=%d, period=%.3fs, csv=%s",
        device_index,
        sample_period_s,
        csv_path,
    )

    try:
        while not stop_event.is_set():
            t0 = time.time()
            try:
                # Query GPU power in milliwatts, convert to watts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = float(power_mw) / 1000.0
                energy_j = power_w * float(sample_period_s)
            except Exception as e:
                logging.warning("Failed to query NVML power: %s", e)
                power_w = float("nan")
                energy_j = float("nan")

            writer.writerow(
                {
                    "timestamp": f"{t0:.6f}",
                    "power_w": f"{power_w:.6f}",
                    "energy_j": f"{energy_j:.6f}",
                }
            )
            f.flush()

            # Sleep to maintain target sample period
            elapsed = time.time() - t0
            sleep_for = float(sample_period_s) - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        try:
            f.close()
        except Exception:
            pass
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    logging.info("Power sampler stopped")


def start_power_sampler(
    csv_path: Optional[str],
    sample_period_s: float,
    device_index: int,
) -> (Optional[threading.Thread], Optional[threading.Event]):
    """Start a background thread for power sampling; return (thread, stop_event)."""
    if not csv_path:
        return None, None

    stop_event: threading.Event = threading.Event()
    thread = threading.Thread(
        target=power_sampler_loop,
        args=(csv_path, sample_period_s, device_index, stop_event),
        daemon=True,
    )
    try:
        thread.start()
        return thread, stop_event
    except Exception as e:
        logging.warning("Failed to start power sampler: %s", e)
        return None, None


def attach_energy_to_windows(
    windows_csv_path: str,
    energy_csv_path: str,
    energy_ts_col: str = "timestamp",
    energy_col: str = "energy_j",
    idle_power_threshold_w: Optional[float] = None,
    idle_calibration_seconds: Optional[float] = None,
) -> None:
    """
    Join per-sample power/energy data into window CSV.
    
    Adds:
    - energy_j_per_window: sum of energy_j in that window
    - avg_power_w: energy / window_s
    - energy_idle_label: 1 if avg_power_w <= threshold
    
    Modifies windows_csv_path in-place.
    """
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

    # Verify required columns
    if "window_start_ts" not in windows.columns or "window_index" not in windows.columns:
        logging.warning(
            f"{windows_csv_path} missing window_start_ts/window_index; "
            "skipping energy/window join"
        )
        return

    if "label_idle_gt" not in windows.columns and "idle_label" in windows.columns:
        windows["label_idle_gt"] = windows["idle_label"]

    # Extract window size from first window
    window_s = 1.0
    if "window_s" in windows.columns:
        try:
            ws = float(windows["window_s"].iloc[0])
            if ws > 0:
                window_s = ws
        except Exception:
            pass

    t0 = windows["window_start_ts"].min()
    if not isinstance(t0, (int, float)) or not (t0 == t0):
        logging.warning("window_start_ts has no valid values; skipping energy join")
        return

    # Auto-calibrate idle power threshold from early samples if not provided
    auto_idle_thr_w: Optional[float] = None
    if (
        (idle_power_threshold_w is None or idle_power_threshold_w <= 0.0)
        and idle_calibration_seconds is not None
        and idle_calibration_seconds > 0.0
    ):
        power_col = None
        for cand in ("power_w", "gpu_power_w", "power", "power_draw_w"):
            if cand in energy.columns:
                power_col = cand
                break

        if power_col is not None:
            try:
                ts_series = pd.to_numeric(
                    energy.get(energy_ts_col, energy.index),
                    errors="coerce",
                )
                t0_energy = ts_series.min()
                if t0_energy == t0_energy:
                    # Use 99th percentile of early power samples as idle threshold
                    cutoff = float(t0_energy) + float(idle_calibration_seconds)
                    idle_mask = ts_series <= cutoff
                    idle_power = pd.to_numeric(
                        energy.loc[idle_mask, power_col],
                        errors="coerce",
                    ).dropna()
                    if not idle_power.empty:
                        p99_idle = float(idle_power.quantile(0.99))
                        auto_idle_thr_w = p99_idle
                        logging.info(
                            "Auto-calibrated idle power threshold from first %.1fs "
                            "of power samples -> threshold=%.2fW",
                            idle_calibration_seconds,
                            auto_idle_thr_w,
                        )
            except Exception as e:
                logging.warning("Failed to auto-calibrate idle power threshold: %s", e)

    def _apply_energy_idle_label(df, idle_thr_w: Optional[float]) -> None:
        """Compute energy_idle_label based on avg_power_w and threshold."""
        if idle_thr_w is None:
            logging.info("No idle power threshold available; skipping energy_idle_label.")
            return
        if "avg_power_w" not in df.columns:
            logging.info("avg_power_w not present; cannot compute energy_idle_label.")
            return
        try:
            mask_valid = df["avg_power_w"].notna()
            df["energy_idle_label"] = np.where(
                mask_valid,
                (df["avg_power_w"] <= float(idle_thr_w)).astype(int),
                np.nan,
            )
            df["idle_power_threshold_w"] = float(idle_thr_w)
        except Exception as e:
            logging.warning("Failed to compute energy_idle_label: %s", e)

    # Join energy into windows
    if energy_ts_col in energy.columns and energy_col in energy.columns:
        logging.info(
            f"Energy CSV {energy_csv_path} has per-sample columns "
            f"({energy_ts_col}, {energy_col}); using window-index-based join."
        )

        energy = energy[energy[energy_ts_col].notna()].copy()
        energy[energy_ts_col] = energy[energy_ts_col].astype("float64")

        # Map each energy sample to its window
        energy["window_index"] = (
            ((energy[energy_ts_col] - float(t0)) / float(window_s))
        ).floordiv(1).astype(int)

        # Sum energy within each window
        energy_agg = (
            energy.groupby("window_index", as_index=False)[energy_col]
            .sum()
            .rename(columns={energy_col: "energy_j_per_window"})
        )

        # Left join: keep all windows, add energy where available
        merged = windows.merge(
            energy_agg,
            on="window_index",
            how="left",
            validate="1:1",
        )

        # Compute average power from energy and window size
        if "energy_j_per_window" in merged.columns:
            try:
                merged["avg_power_w"] = (
                    merged["energy_j_per_window"].astype("float64")
                    / merged["window_s"].astype("float64").replace(0.0, float("nan"))
                )
            except Exception as e:
                logging.warning("Failed to compute avg_power_w: %s", e)

        # Compute energy per request (for load-aware analysis)
        merged["energy_per_request"] = (
            merged["energy_j_per_window"] / merged["requests_started"].clip(lower=1)
        )
        merged.loc[merged["requests_started"] == 0, "energy_per_request"] = float("nan")

        # Determine effective idle threshold and label windows
        effective_idle_thr_w: Optional[float]
        if idle_power_threshold_w is not None and idle_power_threshold_w > 0.0:
            effective_idle_thr_w = float(idle_power_threshold_w)
        else:
            effective_idle_thr_w = auto_idle_thr_w

        _apply_energy_idle_label(merged, effective_idle_thr_w)

        # Write augmented windows CSV
        try:
            merged.to_csv(windows_csv_path, index=False)
            logging.info(f"Attached energy features to window metrics in {windows_csv_path}")
        except Exception as e:
            logging.warning(f"Failed to write energy-augmented CSV {windows_csv_path}: {e}")
        return

    logging.warning(
        f"{energy_csv_path} missing required columns for energy join. "
        f"Tried per-sample ({energy_ts_col}, {energy_col}). "
        f"Available columns: {list(energy.columns)}"
    )


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
    window_s: float,
    phases: Optional[List[Dict[str, Any]]] = None,
    energy_csv_path: Optional[str] = None,
    power_csv_path: Optional[str] = None,
    power_sample_period_s: float = 0.1,
    power_device_index: int = 0,
    phases_total_seconds: Optional[int] = None,
    idle_power_threshold_w: Optional[float] = None,
    idle_calibration_seconds: float = 0.0,
    random_seed: Optional[int] = None,
) -> None:
    """
    Main load generation orchestrator.
    
    Coordinates:
    1. Sample loading
    2. Power sampling thread
    3. Worker threads
    4. Traffic scheduling
    5. Result aggregation and energy attachment
    """
    # ... existing code ...
    samples = load_samples(mode)

    power_thread: Optional[threading.Thread] = None
    power_stop_event: Optional[threading.Event] = None
    if power_csv_path:
        power_thread, power_stop_event = start_power_sampler(
            csv_path=power_csv_path,
            sample_period_s=power_sample_period_s,
            device_index=power_device_index,
        )

    # Idle calibration: measure baseline power before sending traffic
    if idle_calibration_seconds and idle_calibration_seconds > 0.0:
        logging.info(
            "Idle calibration: sleeping for %.1fs with no requests to measure "
            "GPU baseline power",
            idle_calibration_seconds,
        )
        time.sleep(idle_calibration_seconds)

    # Send warmup requests to prime the model (not recorded)
    if warmup_requests > 0:
        logging.info(f"Warmup: sending {warmup_requests} requests (not recorded)")
        for _ in range(warmup_requests):
            sample = random.choice(samples)
            _ = send_request(url, model_name, sample, mode, headers)

    # Create task queue and worker threads
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

    # ... existing phase/pattern scheduling code ...
    phases_used_names: List[str] = []
    if phases:
        for p in phases:
            nm = p.get("name")
            if nm:
                phases_used_names.append(nm)

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

    # Signal workers to stop after scheduling completes
    for _ in range(concurrency):
        task_queue.put(None)

    # Wait for all tasks to complete
    task_queue.join()

    # Wait for worker threads to finish
    for t in threads:
        t.join()

    # Stop power sampler
    if power_stop_event is not None and power_thread is not None:
        power_stop_event.set()
        power_thread.join()

    # Compute statistics from collected results
    if not results:
        logging.warning("No successful requests recorded (results list is empty)")
        return

    latencies = [r.latency_ms for r in results if r.status_code == 200]
    errors = [r for r in results if r.status_code != 200]
    total = len(results)
    window = max(r.ts_end for r in results) - min(r.ts_start for r in results)
    throughput = total / window if window > 0 else float("nan")

    # Compute latency percentiles
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

    # Write per-request CSV
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

        # Aggregate into windows
        meta: Dict[str, Any] = {
            "model_name": model_name,
            "mode": mode,
            "pattern": pattern,
            "concurrency": concurrency,
            "window_s": window_s,
        }
        if random_seed is not None:
            meta["random_seed"] = random_seed
        if phases_total_seconds is not None:
            meta["phases_total_seconds"] = phases_total_seconds
        if phases_used_names:
            try:
                meta["phases_used"] = json.dumps(phases_used_names)
            except Exception:
                meta["phases_used"] = ",".join(phases_used_names)

        windows_csv_path = aggregate_window_metrics(
            csv_path, window_s=window_s, meta=meta
        )

        # Attach energy to windows if available
        effective_energy_csv = energy_csv_path or power_csv_path

        if windows_csv_path and effective_energy_csv:
            attach_energy_to_windows(
                windows_csv_path=windows_csv_path,
                energy_csv_path=effective_energy_csv,
                idle_power_threshold_w=idle_power_threshold_w,
                idle_calibration_seconds=idle_calibration_seconds,
            )


def main() -> None:
    """Parse command-line arguments and launch load test."""
    parser = argparse.ArgumentParser(description="TorchServe load generator")
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
        "--window-s",
        type=float,
        default=1.0,
        help="Window size in seconds for aggregated metrics (default: 1.0)",
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
        "--phase-duration-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to all phase durations when using "
            "--phases-json (e.g., 0.5 halves all durations)."
        ),
    )
    parser.add_argument(
        "--energy-csv",
        default=None,
        help=(
            "Optional path to an energy CSV. Supports per-sample "
            "columns 'timestamp'/'energy_j'."
        ),
    )
    parser.add_argument(
        "--power-csv",
        default=None,
        help=(
            "Optional path to a power sampling CSV collected via NVML "
            "(timestamp, power_w, energy_j). When provided, a background "
            "sampler will run during the load test."
        ),
    )
    parser.add_argument(
        "--power-sample-period",
        type=float,
        default=0.1,
        help="Sampling period in seconds for --power-csv (default: 0.1)",
    )
    parser.add_argument(
        "--power-device-index",
        type=int,
        default=0,
        help="NVML GPU device index to sample with --power-csv (default: 0)",
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
            "--energy-csv is provided, windows with avg_power_w "
            "less than or equal to this are marked with energy_idle_label=1. "
            "If not set and --idle-calibration-seconds is provided, an idle "
            "threshold is derived from early power samples."
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
    parser.add_argument(
        "--idle-calibration-seconds",
        type=float,
        default=0.0,
        help=(
            "Optional: seconds to spend at the beginning of the run with "
            "no requests, only GPU power sampling. When provided and "
            "--idle-power-threshold-w is not set, this period is used to "
            "automatically calibrate an idle power threshold from NVML "
            "power samples."
        ),
    )

    args = parser.parse_args()

    # Set global random seeds for reproducibility
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
            pass

    # Handle attach-energy-only mode (post-processing only)
    if args.attach_energy_only:
        if not args.windows_csv:
            raise SystemExit("--attach-energy-only requires --windows-csv")
        if not args.energy_csv:
            raise SystemExit("--attach-energy-only requires --energy-csv")

        attach_energy_to_windows(
            windows_csv_path=args.windows_csv,
            energy_csv_path=args.energy_csv,
            idle_power_threshold_w=args.idle_power_threshold_w,
            idle_calibration_seconds=args.idle_calibration_seconds,
        )
        return

    # Determine TorchServe URL
    url = args.url or f"http://{os.environ.get('HOSTNAME', '127.0.0.1')}:8080"

    # Set up auth headers if token provided
    headers: Dict[str, str] = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    # Load phases if provided
    phases = None
    if args.phases_json:
        with open(args.phases_json) as f:
            phases = json.load(f)

        # Filter to single phase by name if requested
        if args.phase_name:
            filtered = [p for p in phases if p.get("name") == args.phase_name]
            if not filtered:
                raise SystemExit(
                    f"No phase named '{args.phase_name}' found in {args.phases_json}"
                )
            phases = filtered

        # Apply global duration scale to all phases
        if args.phase_duration_scale != 1.0:
            factor = float(args.phase_duration_scale)
            for p in phases:
                if "duration" in p:
                    p["duration"] = max(1, int(round(p["duration"] * factor)))
            logging.info(
                "Scaled phase durations by factor %.3f based on --phase-duration-scale",
                factor,
            )

    # Launch the load test
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
        window_s=args.window_s,
        phases=phases,
        energy_csv_path=args.energy_csv,
        power_csv_path=args.power_csv,
        power_sample_period_s=args.power_sample_period,
        power_device_index=args.power_device_index,
        phases_total_seconds=args.phases_total_seconds,
        idle_power_threshold_w=args.idle_power_threshold_w,
        idle_calibration_seconds=args.idle_calibration_seconds,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()