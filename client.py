#!/usr/bin/env python3
"""
client.py — TorchServe load generator inspired by official benchmarks.

High-level idea
---------------
This script generates synthetic load (HTTP requests) against a TorchServe model
server and records detailed metrics:

  - When each request started and finished.
  - Latency per request.
  - HTTP status codes and errors.
  - Aggregated metrics per 1-second window (requests per second, latency stats).
  - An "idle vs busy" label per window (based on request count).
  - Optional correlation between per-window load and per-window GPU energy
    (if we provide a Zeus energy CSV).

It supports:
  - Randomised burst/idle traffic (microservice-like).
  - Configurable concurrency (number of worker threads).
  - Different traffic patterns: steady, burst, Poisson.
  - Multi-phase patterns via a JSON file (phases.json).
  - Randomised order / repetition of phases for long-running experiments.
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

# Configure logging for the entire script. This controls how log messages
# from logging.info(), logging.warning(), etc. will appear on stdout/stderr.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Type alias for the allowed traffic patterns. This is mainly for type-checkers
# and readability when we pass around the "pattern" parameter.
TrafficPattern = Literal["burst", "steady", "poisson"]


@dataclass
class RequestResult:
    """
    Container for per-request metrics that we can later aggregate
    or dump to CSV.

    Each HTTP request sent to TorchServe results in one RequestResult.

    Attributes:
        ts_start:    Wall-clock timestamp (seconds since epoch) just before
                     sending the HTTP request.
        ts_end:      Wall-clock timestamp when we receive the response or
                     encounter an error.
        latency_ms:  Simple latency measurement = (ts_end - ts_start) * 1000.
        status_code: HTTP status code from TorchServe (e.g. 200), or 0 if the
                     request never reached the server (e.g. timeout, network error).
        error:       Short error description or snippet of response body if
                     the response is not 200 OK; otherwise None.
    """
    ts_start: float
    ts_end: float
    latency_ms: float
    status_code: int
    error: Optional[str]


# ---------------------------------------------------------------------------
# Helpers for building request payloads
# ---------------------------------------------------------------------------

def img_to_bytes(img: Image.Image) -> bytes:
    """
    Encode a PIL image as JPEG and return the raw bytes.

    TorchServe image handlers typically expect binary image data in the request
    body (e.g., for classifiers). We keep everything in-memory using BytesIO.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG")  # Encode image as JPEG into the buffer.
    return buf.getvalue()         # Extract raw bytes from buffer.


def send_request(
    url: str,
    model_name: str,
    sample: Any,
    mode: str,
    headers: Dict[str, str],
    timeout: float = 30.0,
) -> RequestResult:
    """
    Send a single HTTP request to TorchServe and capture basic timing/response stats.

    This is the lowest-level function that actually talks to the TorchServe
    inference endpoint. Everything else (workers, traffic patterns, etc.) ends
    up calling this.

    Args:
        url:        Base TorchServe URL (e.g. http://localhost:8080).
        model_name: Name of the model as registered in TorchServe.
        sample:     Input data for the model:
                        - PIL.Image.Image when mode == "image"
                        - str (sentence) when mode == "text"
        mode:       "image" or "text" selects how we build the payload.
        headers:    Extra HTTP headers to include (e.g. Authorization).
        timeout:    HTTP timeout in seconds for this request.

    Returns:
        RequestResult with timestamps, latency, status code and error info.
    """
    ts_start = time.time()  # Start timer right before the HTTP call.
    try:
        if mode == "image":
            # For image models we send raw JPEG bytes.
            img_bytes = img_to_bytes(sample)
            r = requests.post(
                f"{url}/predictions/{model_name}",
                data=img_bytes,
                headers=headers,
                timeout=timeout,
            )
        else:
            # For text models we send a small JSON object: {"text": "<sentence>"}.
            payload = json.dumps({"text": sample})
            r = requests.post(
                f"{url}/predictions/{model_name}",
                data=payload,
                headers={**headers, "Content-Type": "application/json"},
                timeout=timeout,
            )

        ts_end = time.time()
        latency_ms = (ts_end - ts_start) * 1000.0

        # For successful responses we keep error=None; otherwise we store up to
        # 200 characters of the body to help debug failures.
        return RequestResult(
            ts_start=ts_start,
            ts_end=ts_end,
            latency_ms=latency_ms,
            status_code=r.status_code,
            error=None if r.status_code == 200 else r.text[:200],
        )
    except Exception as e:
        # Any exception here means we never got a valid response from the server.
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


# ---------------------------------------------------------------------------
# Text dataset handling (for --mode text)
# ---------------------------------------------------------------------------

# Public dataset with short English sentences + sentiment labels.
# We only use the sentence text as generic payloads (we ignore labels).
UCI_SENTIMENT_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip"
)

# Local cache where we store the extracted sentences so we don't re-download
# the dataset on every run. This lets us run even without network after the
# first successful download.
UCI_SENTENCE_CACHE = os.path.join("data", "uci_sentiment_sentences.txt")


def _download_uci_sentences(cache_path: str = UCI_SENTENCE_CACHE) -> List[str]:
    """
    Download and parse the UCI 'Sentiment Labelled Sentences' dataset.

    The ZIP file contains three text files (IMDB, Amazon, Yelp). Each line has
    the format:

        <sentence>\t<label>

    We ignore the label and keep the sentence text.

    Args:
        cache_path: Where to persist the extracted sentences locally.

    Returns:
        List of unique, non-empty sentences. Returns an empty list if anything
        goes wrong (HTTP error, parse error, etc.).
    """
    try:
        logging.info(
            "Downloading UCI Sentiment Labelled Sentences dataset "
            "for text-mode load generation…"
        )
        # Download ZIP file from the UCI repository.
        resp = requests.get(UCI_SENTIMENT_ZIP_URL, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logging.warning(f"Failed to download UCI sentiment dataset: {e}")
        return []

    try:
        # Wrap downloaded bytes in a buffer so zipfile can open it.
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
    except Exception as e:
        logging.warning(f"Failed to open UCI sentiment ZIP: {e}")
        return []

    # Expected text files in the ZIP archive.
    wanted_files = (
        "imdb_labelled.txt",
        "amazon_cells_labelled.txt",
        "yelp_labelled.txt",
    )

    sentences: List[str] = []

    for fname in wanted_files:
        if fname not in zf.namelist():
            logging.warning(f"File {fname} not found inside UCI sentiment ZIP.")
            continue

        try:
            with zf.open(fname) as f:
                for raw_line in f:
                    # Lines are bytes -> decode to string.
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    # Split "<sentence>\t<label>" and keep the sentence part.
                    parts = line.split("\t")
                    if not parts:
                        continue

                    sentence = parts[0].strip()
                    if sentence:
                        sentences.append(sentence)
        except Exception as e:
            logging.warning(f"Failed to parse {fname} from UCI sentiment ZIP: {e}")

    if not sentences:
        logging.warning("No sentences extracted from UCI sentiment dataset.")
        return []

    # Deduplicate while preserving original order.
    seen = set()
    unique_sentences: List[str] = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)

    # Cache to a local file so we don't re-download on every run.
    try:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            for s in unique_sentences:
                f.write(s + "\n")
        logging.info(
            f"Cached {len(unique_sentences)} text samples to {cache_path} "
            "for future runs."
        )
    except Exception as e:
        logging.warning(f"Failed to write UCI sentence cache file {cache_path}: {e}")

    return unique_sentences


def load_text_samples(num_samples: int = 100) -> List[str]:
    """
    Load a pool of text samples to send to TorchServe when --mode text.

    Strategy:
      1. Try to read sentences from the local cache file (fast, no network).
      2. If cache is missing/empty, download the UCI dataset and build it.
      3. If that fails, fall back to a small built-in list of example sentences.

    Args:
        num_samples: Target number of sentences to return.

    Returns:
        A list of sentences of length >= num_samples. If the dataset has fewer
        sentences than requested, we repeat them to reach num_samples.
    """
    sentences: List[str] = []

    # 1) Try local cache first (no network dependency).
    if os.path.exists(UCI_SENTENCE_CACHE):
        logging.info(f"Loading text samples from cache: {UCI_SENTENCE_CACHE}")
        try:
            with open(UCI_SENTENCE_CACHE, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.warning(f"Failed to read cached text samples: {e}")
            sentences = []

    # 2) If cache is missing/empty, download and build it.
    if not sentences:
        sentences = _download_uci_sentences(cache_path=UCI_SENTENCE_CACHE)

    # 3) Last-resort fallback so the load generator still works even offline.
    if not sentences:
        logging.warning(
            "Falling back to a small built-in list of text samples "
            "(UCI dataset unavailable)."
        )
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

    # If we requested more samples than we have, repeat them until we reach
    # num_samples. This avoids running out of text samples for long tests.
    if num_samples > len(sentences):
        repeats = (num_samples + len(sentences) - 1) // len(sentences)
        return (sentences * repeats)[:num_samples]

    return sentences[:num_samples]


# ---------------------------------------------------------------------------
# Sample loading (image / text)
# ---------------------------------------------------------------------------

def load_samples(mode: str, num_samples: int = 100) -> List[Any]:
    """
    Build a pool of input samples depending on the selected mode.

    For image mode:
        - Use a subset of the CIFAR-10 test set (downloaded via torchvision).
    For text mode:
        - Use sentences from the UCI dataset (or fallback list).

    All worker threads will randomly draw from this shared pool. We don't care
    about labels, only inputs.
    """
    if mode == "image":
        logging.info("Loading CIFAR-10 test subset for image classification…")
        # torchvision will automatically download CIFAR-10 to ./data on first use.
        dataset = CIFAR10(root="./data", train=False, download=True)
        # dataset[i] returns (PIL.Image, label). We keep only the PIL.Image.
        return [dataset[i][0] for i in range(min(num_samples, len(dataset)))]
    elif mode == "text":
        return load_text_samples(num_samples=num_samples)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Worker threads and traffic scheduling
# ---------------------------------------------------------------------------

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
    Worker thread body.

    Each worker continuously consumes items ("tokens") from task_queue.
    The semantics of the tokens:

      - An integer token (e.g. 1) means "send a request now".
      - A token of None is a sentinel that tells the worker to exit cleanly.

    For each "real" token, the worker:
      1. Picks a random sample from sample_pool.
      2. Calls send_request(...) to talk to TorchServe.
      3. Appends the RequestResult to result_list.

    Note:
      - list.append() is atomic in CPython, so appending to result_list from
        multiple threads is safe in practice.
      - task_queue.task_done() is called in a finally block so that the queue
        accounting is correct even if something goes wrong.
    """
    while True:
        token = task_queue.get()  # Block until a token is available.
        try:
            if token is None:
                # Sentinel received: stop this worker and exit the loop.
                return
            if token <= 0:
                # Ignore invalid tokens but still mark them as done.
                continue

            # Pick a random input sample (image or sentence).
            sample = random.choice(sample_pool)
            result = send_request(url, model_name, sample, mode, headers)
            result_list.append(result)
        finally:
            # Notify that we finished processing this queue item.
            task_queue.task_done()


def schedule_burst_pattern(
    duration: int,
    burst: int,
    idle: int,
    base_rps: int,
    task_queue: "queue.Queue[Optional[int]]",
):
    """
    Bursty pattern: alternate randomised burst + idle periods.

    Behaviour:
      - During "burst" periods, we enqueue roughly base_rps tokens per second.
      - During "idle" periods, we enqueue nothing and just sleep.

    Rationale:
      - This pattern better mimics microservice traffic, which typically
        arrives in spikes and then quiet periods rather than being constant.

    Args:
        duration:   Total duration (seconds) to run this pattern.
        burst:      Mean burst duration (seconds). Actual burst duration is
                    sampled from a Gaussian around this mean.
        idle:       Mean idle duration (seconds). Same idea as burst.
        base_rps:   Average requests per second during bursts.
        task_queue: Shared queue to which we will push "request tokens".
    """
    if base_rps <= 0:
        # Degenerate case: treat the entire duration as idle.
        logging.info(f"😴 Burst pattern with base_rps={base_rps} -> pure idle for {duration}s")
        time.sleep(duration)
        return

    start = time.time()
    while time.time() - start < duration:
        # Randomise the exact durations and RPS to avoid perfectly regular patterns.
        burst_dur = max(5, int(random.gauss(burst, max(1.0, burst * 0.2))))
        idle_dur = max(2, int(random.gauss(idle, max(1.0, idle * 0.3))))
        req_per_sec = max(1, int(random.gauss(base_rps, max(1.0, base_rps * 0.25))))

        logging.info(f"💥 Burst for {burst_dur}s at ~{req_per_sec} req/s")
        burst_start = time.time()

        # Emit tokens for each second of the burst, but don't run past the
        # total pattern duration.
        while time.time() - burst_start < burst_dur and time.time() - start < duration:
            for _ in range(req_per_sec):
                task_queue.put(1)  # Each token triggers exactly one request.
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
    Steady open-loop rate: every 1/rps seconds, enqueue one request token.

    This gives a roughly constant arrival rate independent of per-request
    latency (open-loop). If the server can't keep up, latency will grow,
    but the rate of new requests is not throttled by response time.

    Args:
        duration:   Total duration (seconds) for this pattern.
        rps:        Target requests per second. If <= 0, treat as idle.
        task_queue: Shared queue for request tokens.
    """
    if rps <= 0:
        logging.info(f"📉 Steady load with rps={rps} -> pure idle for {duration}s")
        time.sleep(duration)
        return

    logging.info(f"📈 Steady load: {rps} req/s for {duration}s")
    start = time.time()
    inter_arrival = 1.0 / float(rps)
    next_time = start

    while time.time() - start < duration:
        now = time.time()
        if now >= next_time:
            # Time to enqueue the next request.
            task_queue.put(1)
            next_time += inter_arrival
        else:
            # Sleep until the next scheduled arrival.
            sleep_for = next_time - now
            if sleep_for > 0:
                time.sleep(sleep_for)


def schedule_poisson_pattern(
    duration: int,
    rps: int,
    task_queue: "queue.Queue[Optional[int]]",
):
    """
    Poisson arrivals: exponential inter-arrival times with rate λ = rps.

    This approximates independent users sending requests at random. The
    inter-arrival times are drawn from an exponential distribution, which is
    typical in queueing theory.

    Args:
        duration:   Total duration (seconds).
        rps:        Average requests per second (λ). If <= 0, treat as idle.
        task_queue: Shared queue for request tokens.
    """
    if rps <= 0:
        logging.info(f"🎲 Poisson load with rps={rps} -> pure idle for {duration}s")
        time.sleep(duration)
        return

    logging.info(f"🎲 Poisson load: avg {rps} req/s for {duration}s")
    start = time.time()
    lam = float(rps)

    while time.time() - start < duration:
        # Exponential inter-arrival time with mean 1/λ.
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
    Dispatch helper: choose the right scheduler based on pattern name.

    This function is used both for:
      - A single-pattern run (no phases).
      - Each phase when using a --phases-json file.
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


# ---------------------------------------------------------------------------
# CSV post-processing: fixed-size windows + idle labels (+ optional energy)
# ---------------------------------------------------------------------------

def aggregate_window_metrics(csv_path: str, window_s: float = 1.0) -> Optional[str]:
    """
    Post-process a per-request CSV into fixed-size time windows.

    Input:
      - Per-request CSV with columns: ts_start, ts_end, latency_ms, status_code, error

    Output:
      - A file like "<csv_path>_windows_1s.csv" with columns:

          window_index       (0, 1, 2, … based on ts_start)
          window_start_ts    (float seconds since epoch for start of window)
          window_start_dt    (human-readable timestamp)
          requests_started   (#requests whose ts_start fell in this window)
          avg_latency_ms     (mean latency for this window)
          p50_latency_ms     (median latency for this window)
          error_rate         (% of non-200 status codes)
          is_idle            (True if requests_started == 0)
          idle_label         (1 if idle, 0 if busy)

    The purpose is to convert per-request data into per-second "time series"
    samples that we can use to train and evaluate "idle vs busy" predictors.

    If pandas/numpy are not available, this step is skipped.

    Returns:
        Path to the window-level CSV if successfully written, otherwise None.
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        logging.warning("pandas/numpy not available; skipping window aggregation.")
        return None

    # Read the per-request CSV.
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.warning(f"Failed to read CSV {csv_path}: {e}")
        return None

    if df.empty:
        logging.warning(f"No rows in {csv_path}; skipping window aggregation.")
        return None

    # Make sure required columns exist.
    required_cols = {"ts_start", "ts_end", "latency_ms", "status_code"}
    if not required_cols.issubset(df.columns):
        logging.warning(
            f"{csv_path} missing required columns {required_cols}; "
            "skipping window aggregation."
        )
        return None

    # Determine the earliest and latest request start time.
    t_min = df["ts_start"].min()
    t_max = df["ts_start"].max()
    if t_min is None or (isinstance(t_min, float) and (t_min != t_min)):
        logging.warning("ts_start has no valid values; skipping window aggregation.")
        return None

    # Assign each request to an integer window index based on ts_start and window size.
    df["window_index"] = np.floor((df["ts_start"] - t_min) / window_s).astype(int)

    # Group all requests that fall into the same window index.
    grouped = df.groupby("window_index")

    # Compute per-window aggregates for windows that have at least one request.
    window_stats = grouped.agg(
        requests_started=("ts_start", "count"),
        avg_latency_ms=("latency_ms", "mean"),
        p50_latency_ms=("latency_ms", "median"),
        error_rate=("status_code", lambda s: (s != 200).mean() * 100.0),
    )

    # Build the full range of window indices, including those with zero requests.
    max_index = int(np.floor((t_max - t_min) / window_s))
    full_index = np.arange(0, max_index + 1, dtype=int)

    # Reindex so we have a row for every window (0..max_index).
    window_stats = window_stats.reindex(full_index)

    # For windows with no requests, requests_started is 0 and error_rate is 0.
    window_stats["requests_started"] = window_stats["requests_started"].fillna(0).astype(int)
    # Latencies remain NaN for empty windows.
    window_stats["error_rate"] = window_stats["error_rate"].fillna(0.0)

    # Move index into a column called "window_index".
    window_stats = window_stats.reset_index().rename(columns={"index": "window_index"})

    # Compute the start timestamp of each window.
    window_stats["window_start_ts"] = t_min + window_stats["window_index"] * window_s

    # Optional: also include a human-readable datetime column.
    try:
        window_stats["window_start_dt"] = pd.to_datetime(
            window_stats["window_start_ts"], unit="s"
        )
    except Exception:
        # Not critical if this fails.
        pass

    # Define idle windows as those with zero requests.
    window_stats["is_idle"] = window_stats["requests_started"] == 0
    window_stats["idle_label"] = window_stats["is_idle"].astype(int)  # 1 = idle, 0 = busy

    # Write the result to disk next to the per-request CSV.
    out_path = csv_path.replace(".csv", f"_windows_{int(window_s)}s.csv")
    try:
        window_stats.to_csv(out_path, index=False)
        logging.info(f"🧮 Wrote window-level metrics to {out_path}")
        return out_path
    except Exception as e:
        logging.warning(f"Failed to write window stats CSV {out_path}: {e}")
        return None


def attach_energy_to_windows(
    windows_csv_path: str,
    energy_csv_path: str,
    energy_ts_col: str = "timestamp",
    energy_col: str = "energy_j",
) -> None:
    """
    Optional post-processing step: correlate per-second energy with load.

    Inputs:
      - windows_csv_path: CSV produced by aggregate_window_metrics(...), e.g.
            "<csv_path>_windows_1s.csv"
      - energy_csv_path: CSV with at least:
            * <energy_ts_col>: timestamp in seconds since epoch
            * <energy_col>:    energy (e.g. Joules) for that second

    This function:
      - Aligns per-second energy to the window_start_ts of each window.
      - Adds two new columns:
            * energy_j_per_window: total energy in that 1-second window.
            * energy_per_request:  energy per request in that window.
      - Logs:
            * Correlation between requests_started and energy_j_per_window.
            * Mean energy in idle windows vs busy windows.

    This is useful to:
      - Validate that our "idle" windows indeed consume less energy.
      - Provide extra features for ML models (e.g. energy per request).
    """
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        logging.warning("pandas/numpy not available; skipping energy/window join.")
        return

    if not os.path.exists(windows_csv_path):
        logging.warning(
            f"Window metrics CSV {windows_csv_path} not found; "
            "cannot attach energy."
        )
        return

    if not os.path.exists(energy_csv_path):
        logging.warning(
            f"Energy CSV {energy_csv_path} not found; "
            "skipping energy/window join."
        )
        return

    # Read both CSVs.
    try:
        windows = pd.read_csv(windows_csv_path)
        energy = pd.read_csv(energy_csv_path)
    except Exception as e:
        logging.warning(f"Failed to read windows/energy CSVs: {e}")
        return

    if windows.empty or energy.empty:
        logging.warning("Windows or energy CSV is empty; skipping join.")
        return

    if "window_start_ts" not in windows.columns:
        logging.warning(
            f"{windows_csv_path} missing window_start_ts; "
            "skipping energy/window join."
        )
        return

    if energy_ts_col not in energy.columns or energy_col not in energy.columns:
        logging.warning(
            f"{energy_csv_path} missing required columns "
            f"({energy_ts_col}, {energy_col}); skipping energy/window join."
        )
        return

    # Create an integer "second" key in both dataframes by rounding the timestamps.
    # This allows us to join windows and energy on the same integer second.
    windows["second"] = windows["window_start_ts"].round().astype("int64")
    energy["second"] = energy[energy_ts_col].round().astype("int64")

    # If there are multiple energy rows for the same second, aggregate them
    # by summing up the energy for that second.
    energy_agg = (
        energy
        .groupby("second", as_index=False)[energy_col]
        .sum()
        .rename(columns={energy_col: "energy_j_per_window"})
    )

    # Merge per-window metrics with per-second energy.
    merged = windows.merge(
        energy_agg,
        on="second",
        how="left",          # Keep all windows even if some seconds lack energy data.
        validate="m:1",      # Many windows to 1 energy value per second.
    )

    # Compute energy per request.
    # We clip requests_started at 1 to avoid division by zero, but then explicitly
    # set energy_per_request to NaN for windows with 0 requests (idle).
    merged["energy_per_request"] = (
        merged["energy_j_per_window"] /
        merged["requests_started"].clip(lower=1)
    )
    merged.loc[merged["requests_started"] == 0, "energy_per_request"] = np.nan

    # Simple load–energy correlation for sanity checking.
    if "energy_j_per_window" in merged.columns:
        try:
            corr_df = merged[["requests_started", "energy_j_per_window"]].dropna()
            if not corr_df.empty:
                corr = corr_df.corr().iloc[0, 1]
                logging.info(
                    "🔌 Corr(requests_started, energy_j_per_window) = "
                    f"{corr:.4f}"
                )
        except Exception as e:
            logging.warning(f"Failed to compute load/energy correlation: {e}")

    # Basic idle vs busy energy stats to help validate idle_label.
    if "idle_label" in merged.columns and "energy_j_per_window" in merged.columns:
        try:
            idle_energy = merged.loc[merged["idle_label"] == 1, "energy_j_per_window"].dropna()
            busy_energy = merged.loc[merged["idle_label"] == 0, "energy_j_per_window"].dropna()
            if not idle_energy.empty and not busy_energy.empty:
                logging.info(
                    "🔍 Idle vs busy energy: "
                    f"idle_mean={idle_energy.mean():.2f}J, "
                    f"busy_mean={busy_energy.mean():.2f}J"
                )
        except Exception as e:
            logging.warning(f"Failed to compute idle/busy energy stats: {e}")

    # Helper column is not needed in the final CSV.
    try:
        merged.drop(columns=["second"], inplace=True)
    except KeyError:
        pass

    # Overwrite the windows CSV with the energy-augmented version.
    try:
        merged.to_csv(windows_csv_path, index=False)
        logging.info(
            f"🔋 Attached energy features to window metrics in {windows_csv_path}"
        )
    except Exception as e:
        logging.warning(f"Failed to write energy-augmented CSV {windows_csv_path}: {e}")


# ---------------------------------------------------------------------------
# Main orchestration for a full load-test run
# ---------------------------------------------------------------------------

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
) -> None:
    """
    Orchestrate the entire load test from start to finish.

    High-level steps:
      1. Prepare input samples (images or text).
      2. Optional warmup phase (requests not recorded).
      3. Start worker threads.
      4. Schedule request tokens according to either:
           - a single global pattern (burst/steady/poisson), OR
           - a multi-phase configuration from phases.json, possibly repeated
             and randomised until phases_total_seconds is used up.
      5. Wait for all work to finish and stop workers.
      6. Compute headline stats (throughput, latency percentiles).
      7. Write per-request CSV (optional).
      8. Generate per-second window metrics and idle labels.
      9. Optionally join with per-second energy CSV and log correlations.
    """
    # Pre-load a small pool of samples that all worker threads will reuse.
    samples = load_samples(mode)

    # --- Warmup (not recorded in the metrics) ------------------------------
    # Warmup helps the model and GPU "stabilize" (e.g. load weights into memory)
    # before we start recording performance data.
    if warmup_requests > 0:
        logging.info(f"🔥 Warmup: sending {warmup_requests} requests (not recorded)")
        for _ in range(warmup_requests):
            sample = random.choice(samples)
            _ = send_request(url, model_name, sample, mode, headers)

    # Shared queue for work items (tokens). The scheduler pushes here; workers
    # pop from here and send requests.
    task_queue: "queue.Queue[Optional[int]]" = queue.Queue()

    # Shared list of RequestResult objects; only appended to by workers.
    results: List[RequestResult] = []

    # --- Start worker threads ---------------------------------------------
    threads: List[threading.Thread] = []
    for i in range(concurrency):
        t = threading.Thread(
            target=worker_loop,
            args=(f"worker-{i}", url, model_name, mode, headers, samples, task_queue, results),
            daemon=True,  # Daemon threads won't block interpreter exit.
        )
        t.start()
        threads.append(t)

    # --- Schedule tasks according to pattern or phases --------------------
    if phases:
        # Clone the list so we can shuffle it without mutating the original.
        phases_list = list(phases)

        # Compute how long one sequential pass through phases.json would take.
        base_total_phase_duration = sum(int(p.get("duration", 0)) for p in phases_list)

        if phases_total_seconds is not None and phases_total_seconds > 0:
            # Case 1: Run phases repeatedly, in random order, until we have
            # scheduled approximately phases_total_seconds seconds of load.
            logging.info(
                f"🚦 Running randomized phases up to ~{phases_total_seconds}s "
                "of scheduled duration."
            )
            logging.info(
                f"(One sequential pass through phases.json would be "
                f"~{base_total_phase_duration}s.)"
            )

            total_scheduled = 0  # How many seconds we have scheduled so far.
            round_idx = 0        # How many full "rounds" through the phase list.

            while total_scheduled < phases_total_seconds:
                round_idx += 1
                random.shuffle(phases_list)  # Randomize order for this round.
                for phase in phases_list:
                    if total_scheduled >= phases_total_seconds:
                        break

                    phase_pattern = (phase.get("pattern", pattern) or pattern).lower()
                    # Original configured duration for this phase.
                    orig_phase_duration = int(phase.get("duration", duration))
                    remaining = phases_total_seconds - total_scheduled
                    # Clip to remaining budget, but ensure at least 1 second.
                    phase_duration = max(1, min(orig_phase_duration, remaining))
                    phase_rps = int(phase.get("rps", rps))
                    phase_burst = int(phase.get("burst", burst))
                    phase_idle = int(phase.get("idle", idle))
                    name = phase.get("name", f"phase-round{round_idx}")

                    logging.info(
                        f"=== Phase round {round_idx}: {name} | "
                        f"pattern={phase_pattern}, duration={phase_duration}s "
                        f"(configured {orig_phase_duration}s), "
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
                    total_scheduled += phase_duration
        else:
            # Case 2: Run each phase exactly once but in random order.
            total_phase_duration = base_total_phase_duration
            random.shuffle(phases_list)
            logging.info(
                f"🚦 Running {len(phases_list)} phases ONCE in random order "
                f"(total duration ~{total_phase_duration}s). "
                "Per-phase settings can override --pattern/--duration/--rps/--burst/--idle."
            )

            for idx, phase in enumerate(phases_list, start=1):
                phase_pattern = (phase.get("pattern", pattern) or pattern).lower()
                phase_duration = int(phase.get("duration", duration))
                phase_rps = int(phase.get("rps", rps))
                phase_burst = int(phase.get("burst", burst))
                phase_idle = int(phase.get("idle", idle))
                name = phase.get("name", f"phase-{idx}")
                logging.info(
                    f"=== Phase {idx}/{len(phases_list)}: {name} | "
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
        # No phases.json provided: run a single global pattern.
        schedule_pattern(
            pattern=pattern,
            duration=duration,
            rps=rps,
            burst=burst,
            idle=idle,
            task_queue=task_queue,
        )

    # --- Signal workers to shut down --------------------------------------
    # Push one sentinel (None) per worker so they exit after the queue empties.
    for _ in range(concurrency):
        task_queue.put(None)

    # Wait for all queued items (including sentinels) to be processed.
    task_queue.join()

    # Ensure all worker threads have finished.
    for t in threads:
        t.join()

    if not results:
        logging.warning("No successful requests recorded (results list is empty).")
        return

    # --- Compute headline stats (similar to tools like ab/JMeter) ---------
    latencies = [r.latency_ms for r in results if r.status_code == 200]
    errors = [r for r in results if r.status_code != 200]
    total = len(results)
    window = max(r.ts_end for r in results) - min(r.ts_start for r in results)
    throughput = total / window if window > 0 else float("nan")

    if latencies:
        # Sort once and compute percentile estimates.
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

    # --- Optional CSV output (per-request) --------------------------------
    if csv_path:
        logging.info(f"📝 Writing per-request metrics to {csv_path}")
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                # CSV header row.
                writer.writerow(
                    [
                        "ts_start",
                        "ts_end",
                        "latency_ms",
                        "status_code",
                        "error",
                    ]
                )
                # One row per request result.
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

        # Aggregate per-request CSV into per-second windows with idle labels.
        windows_csv_path = aggregate_window_metrics(csv_path, window_s=1.0)

        # If we also have per-second energy, join that with the window metrics
        # to compute correlations and energy features.
        if windows_csv_path and energy_csv_path:
            attach_energy_to_windows(
                windows_csv_path=windows_csv_path,
                energy_csv_path=energy_csv_path,
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse CLI arguments and kick off a load-test run.

    Example usages:

        # Simple image classification test (ResNet-18 on CIFAR-like images)
        python client.py --mode image --model-name resnet-18 --duration 60

        # Text model test using UCI sentence dataset, Poisson arrivals
        python client.py --mode text --model-name bert-text \
                         --pattern poisson --rps 20 --duration 120

        # Multi-phase scenario described in a JSON file
        python client.py --phases-json phases.json --csv results/run1.csv

        # Multi-phase scenario with repeated, randomised phases for ~2h
        python client.py --phases-json phases.json \
                         --phases-total-seconds 7200 \
                         --csv results/run_long.csv
    """
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
        help="Service type to test: 'image' (CIFAR-10) or 'text' (UCI sentences).",
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
        help="Test duration in seconds if not using --phases-json.",
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=30,
        help="Mean burst duration (s) for 'burst' pattern.",
    )
    parser.add_argument(
        "--idle",
        type=int,
        default=15,
        help="Mean idle duration (s) for 'burst' pattern.",
    )
    parser.add_argument(
        "--rps",
        type=int,
        default=10,
        help="Target requests per second (pattern-dependent).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of worker threads (similar to JMeter/ab concurrency).",
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
        help="Warmup requests to send before measurement starts.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to write per-request CSV metrics (optional).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Auth token for TorchServe (optional, for secured endpoints).",
    )
    parser.add_argument(
        "--phases-json",
        default=None,
        help=(
            "Optional path to a JSON file describing multiple load phases. "
            "If set, runs phases instead of a single global pattern."
        ),
    )
    parser.add_argument(
        "--energy-csv",
        default=None,
        help=(
            "Optional path to a per-second energy CSV (e.g. Zeus output) "
            "with columns 'timestamp' and 'energy_j'. If set, energy will be "
            "joined with 1-second window metrics to correlate energy with load."
        ),
    )
    parser.add_argument(
        "--phases-total-seconds",
        type=int,
        default=None,
        help=(
            "When used together with --phases-json, sets an approximate total "
            "number of seconds of scheduled phase time. Phases from the JSON "
            "will be shuffled and repeatedly sampled in random order until "
            "this budget is exhausted. If not set, each phase is run exactly "
            "once in random order."
        ),
    )

    args = parser.parse_args()

    # Derive default URL from HOSTNAME if not explicitly provided.
    url = args.url or f"http://{os.environ.get('HOSTNAME', '127.0.0.1')}:8080"

    # Optional Authorization header if user supplied a token.
    headers: Dict[str, str] = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    # Optional multi-phase configuration (read once here).
    phases = None
    if args.phases_json:
        with open(args.phases_json) as f:
            phases = json.load(f)

    # Kick off the run with the parsed arguments.
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
    )


if __name__ == "__main__":
    # Standard Python entry point: only run main() when executed as a script.
    main()
