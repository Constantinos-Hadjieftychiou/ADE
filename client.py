#!/usr/bin/env python3
# Shebang: tells OS to run this script with python3 when executed as ./client.py
"""
TorchServe load generator with NVML power sampling and window-level metrics aggregation.

This script:
1. Loads sample data (CIFAR-10 images or UCI sentiment text)
2. Generates traffic against a TorchServe inference endpoint using configurable patterns
3. Samples GPU power via NVML in a background thread
4. Aggregates per-request metrics into fixed time windows
5. Optionally attaches energy data to windows for idle detection
"""
# Module docstring: describes what this entire script does

import argparse  # Standard library: parse command-line arguments like --url, --model-name
import csv  # Standard library: read/write CSV files for metrics output
import io  # Standard library: in-memory byte streams for image conversion
import json  # Standard library: parse JSON config files and create JSON payloads
import logging  # Standard library: structured logging with timestamps and levels
import os  # Standard library: file/directory operations, environment variables
import queue  # Standard library: thread-safe FIFO queue for task distribution
import random  # Standard library: random number generation for sampling and timing
import statistics  # Standard library: compute mean, median for latency stats
import threading  # Standard library: multi-threading for workers and power sampler
import time  # Standard library: timestamps (time.time()) and sleep operations
import zipfile  # Standard library: extract UCI sentiment dataset from ZIP
from dataclasses import dataclass  # Decorator: auto-generate __init__, __repr__ for classes
from typing import List, Any, Dict, Optional, Literal  # Type hints for documentation and IDE support

import requests  # Third-party: HTTP library for sending requests to TorchServe
from PIL import Image  # Third-party: Python Imaging Library for image manipulation
from torchvision.datasets import CIFAR10  # PyTorch: dataset class for CIFAR-10 images

# Configure logging for all output (timestamp + level + message)
logging.basicConfig(  # Set up the logging module
    level=logging.INFO,  # Show INFO level and above (INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Format: "2024-01-02 12:34:56 [INFO] message"
)

# Type hint for traffic pattern selection
TrafficPattern = Literal["burst", "steady", "poisson"]  # Restricts valid pattern values to these three strings


@dataclass  # Decorator: auto-generates __init__, __repr__, __eq__ methods
class RequestResult:  # Data class to hold results from a single HTTP request
    """Container for a single request's metrics: start/end time, latency, status, error."""
    ts_start: float  # Unix timestamp when request was sent (seconds since epoch)
    ts_end: float  # Unix timestamp when response was received
    latency_ms: float  # Round-trip time in milliseconds: (ts_end - ts_start) * 1000
    status_code: int  # HTTP status: 200=success, 4xx/5xx=error, 0=network failure
    error: Optional[str]  # Error message if failed, None if successful


# URL and cache path for UCI sentiment dataset (for text mode)
UCI_SENTIMENT_ZIP_URL = (  # URL to download UCI sentiment labeled sentences dataset
    "https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip"
)
UCI_SENTENCE_CACHE = os.path.join("data", "uci_sentiment_sentences.txt")  # Local cache path: "data/uci_sentiment_sentences.txt"


def _load_phases_json(path: str) -> List[Dict[str, Any]]:  # Function to load and validate phases.json
    """
    Load and validate the phases JSON file.

    Fixes the common failure:
      JSONDecodeError: Expecting value: line 1 column 1 (char 0)
    which typically means the file is empty (0 bytes) or truncated / not JSON.

    Accepts either:
      - a list: [ {phase}, {phase}, ... ]
      - a dict wrapper: {"phases": [ ... ]}
    """
    if not path:  # Check if path is empty string or None
        raise SystemExit("ERROR: --phases-json path is empty")  # Exit program with error

    if not os.path.exists(path):  # Check if file exists on filesystem
        raise SystemExit(f"ERROR: phases JSON not found: {path}")  # Exit if file missing

    if os.path.getsize(path) == 0:  # Check if file is empty (0 bytes)
        raise SystemExit(f"ERROR: phases JSON is empty (0 bytes): {path}")  # Exit if empty

    try:  # Try to read the file
        with open(path, "r", encoding="utf-8") as f:  # Open with UTF-8 encoding
            raw = f.read()  # Read entire file as string
    except Exception as e:  # Catch any read errors
        raise SystemExit(f"ERROR: cannot read phases JSON {path}: {e}")  # Exit with error

    prefix = raw[:200].replace("\n", "\\n")  # Save first 200 chars for debugging

    try:  # Try to parse JSON
        data = json.loads(raw)  # Parse JSON string to Python object
    except json.JSONDecodeError as e:  # Catch JSON syntax errors
        raise SystemExit(  # Exit with helpful error message
            f"ERROR: invalid JSON in phases file: {path}\n"
            f"JSON error: {e}\n"
            f"File starts with: {prefix!r}"
        )

    if isinstance(data, dict) and "phases" in data:  # Handle {"phases": [...]} wrapper
        data = data["phases"]  # Unwrap the list from dict

    if not isinstance(data, list):  # Validate result is a list
        raise SystemExit(  # Exit if not a list
            f"ERROR: phases JSON must be a list (or {{'phases': [...]}}). Got {type(data).__name__}"
        )

    # Basic validation: ensure each phase is a dict
    for i, p in enumerate(data):  # Iterate with index for error messages
        if not isinstance(p, dict):  # Each phase must be a dictionary
            raise SystemExit(  # Exit if phase is not a dict
                f"ERROR: phases[{i}] must be an object/dict. Got {type(p).__name__}"
            )

    return data  # Return list of phase dictionaries


def img_to_bytes(img: Image.Image) -> bytes:  # Convert PIL Image to JPEG bytes
    """Convert PIL Image to JPEG bytes for HTTP POST."""
    buf = io.BytesIO()  # Create in-memory byte buffer
    img.save(buf, format="JPEG")  # Encode image as JPEG into buffer
    return buf.getvalue()  # Extract and return bytes from buffer


def make_window_suffix(window_s: float) -> str:  # Create filename suffix from window size
    """
    Create window suffix for CSV filenames (e.g., 0.5s -> "0p5s").
    Used to distinguish window CSVs by their aggregation size.
    """
    suffix = f"{float(window_s):.3f}".rstrip("0").rstrip(".").replace(".", "p")  # Format, strip zeros, replace dot with 'p'
    return suffix  # Return the suffix string


def send_request(  # Function to send a single HTTP request to TorchServe
    url: str,  # Base URL of TorchServe (e.g., "http://127.0.0.1:8080")
    model_name: str,  # Model name to invoke (e.g., "resnet-18")
    sample: Any,  # Sample data: bytes for images, string for text
    mode: str,  # "image" or "text" mode
    headers: Dict[str, str],  # HTTP headers (e.g., Authorization)
    timeout: float = 30.0,  # Request timeout in seconds
) -> RequestResult:  # Returns RequestResult with timing info
    """
    Send a single request to TorchServe and measure latency.

    Handles both image (POST binary JPEG) and text (POST JSON) modes.
    Returns RequestResult with timing and status info.
    """
    ts_start = time.time()  # Record start timestamp (Unix time with microseconds)
    try:  # Try to send request
        if mode == "image":  # Image mode: send raw JPEG bytes
            # Convert sample to JPEG bytes if needed
            if isinstance(sample, (bytes, bytearray)):  # Already bytes
                img_bytes = sample  # Use as-is
            else:  # PIL Image
                img_bytes = img_to_bytes(sample)  # Convert to bytes

            # POST to /predictions/<model_name> with image data
            r = requests.post(  # Send HTTP POST request
                f"{url}/predictions/{model_name}",  # TorchServe inference endpoint
                data=img_bytes,  # Raw binary body
                headers=headers,  # Auth headers if any
                timeout=timeout,  # Fail if no response within timeout
            )
        else:  # text mode
            # Wrap text sample in JSON payload
            payload = json.dumps({"text": sample})  # Create JSON: {"text": "..."}
            r = requests.post(  # Send HTTP POST request
                f"{url}/predictions/{model_name}",  # TorchServe inference endpoint
                data=payload,  # JSON string body
                headers={**headers, "Content-Type": "application/json"},  # Add JSON content type
                timeout=timeout,  # Timeout in seconds
            )

        ts_end = time.time()  # Record end timestamp
        latency_ms = (ts_end - ts_start) * 1000.0  # Convert seconds to milliseconds

        return RequestResult(  # Return success result
            ts_start=ts_start,  # When request was sent
            ts_end=ts_end,  # When response was received
            latency_ms=latency_ms,  # Round-trip time in ms
            status_code=r.status_code,  # HTTP status (200=success)
            error=None if r.status_code == 200 else r.text[:200],  # Error text if not 200
        )
    except Exception as e:  # Catch network errors, timeouts, etc.
        # Capture network/timeout errors as failed requests
        ts_end = time.time()  # Record end time even on failure
        latency_ms = (ts_end - ts_start) * 1000.0  # Calculate latency
        logging.warning(f"Request failed: {e}")  # Log warning
        return RequestResult(  # Return failure result
            ts_start=ts_start,  # When request was sent
            ts_end=ts_end,  # When error occurred
            latency_ms=latency_ms,  # Time until failure
            status_code=0,  # 0 indicates network-level failure
            error=str(e),  # Store exception message
        )


def _download_uci_sentences(cache_path: str = UCI_SENTENCE_CACHE) -> List[str]:  # Download and extract UCI dataset
    """Download UCI sentiment dataset ZIP and extract unique sentences."""
    try:  # Try to download
        logging.info("Downloading UCI Sentiment Labelled Sentences dataset")  # Log info
        resp = requests.get(UCI_SENTIMENT_ZIP_URL, timeout=30)  # Download ZIP file
        resp.raise_for_status()  # Raise exception for 4xx/5xx errors
    except Exception as e:  # Catch download errors
        logging.warning(f"Failed to download UCI sentiment dataset: {e}")  # Log warning
        return []  # Return empty list on failure

    try:  # Try to open ZIP
        zf = zipfile.ZipFile(io.BytesIO(resp.content))  # Open ZIP from in-memory bytes
    except Exception as e:  # Catch ZIP errors
        logging.warning(f"Failed to open UCI sentiment ZIP: {e}")  # Log warning
        return []  # Return empty list

    # Extract sentences from multiple sources in the ZIP
    wanted_files = (  # Files to extract from ZIP
        "imdb_labelled.txt",  # IMDB movie reviews
        "amazon_cells_labelled.txt",  # Amazon product reviews
        "yelp_labelled.txt",  # Yelp business reviews
    )

    sentences: List[str] = []  # Accumulator for extracted sentences

    for fname in wanted_files:  # Process each file
        if fname not in zf.namelist():  # Check if file exists in ZIP
            logging.warning(f"File {fname} not found in UCI sentiment ZIP")  # Log warning
            continue  # Skip missing files

        try:  # Try to extract sentences
            with zf.open(fname) as f:  # Open file within ZIP
                # Each line: "<sentence>\t<label>" (tab-separated)
                for raw_line in f:  # Iterate over lines (bytes)
                    line = raw_line.decode("utf-8", errors="ignore").strip()  # Decode to string
                    if not line:  # Skip empty lines
                        continue  # Continue to next line
                    parts = line.split("\t")  # Split on tab
                    if not parts:  # Skip malformed lines
                        continue  # Continue to next line
                    sentence = parts[0].strip()  # Take sentence part (ignore label)
                    if sentence:  # Skip empty sentences
                        sentences.append(sentence)  # Add to list
        except Exception as e:  # Catch parsing errors
            logging.warning(f"Failed to parse {fname} from UCI sentiment ZIP: {e}")  # Log warning

    if not sentences:  # No sentences extracted
        logging.warning("No sentences extracted from UCI sentiment dataset")  # Log warning
        return []  # Return empty list

    # Remove duplicates while preserving insertion order
    seen = set()  # Set for O(1) duplicate checking
    unique_sentences: List[str] = []  # List for unique sentences
    for s in sentences:  # Iterate through all sentences
        if s not in seen:  # Only add if not seen before
            seen.add(s)  # Mark as seen
            unique_sentences.append(s)  # Add to unique list

    # Cache to disk for future runs
    try:  # Try to write cache
        cache_dir = os.path.dirname(cache_path)  # Get directory part
        if cache_dir:  # If not empty string
            os.makedirs(cache_dir, exist_ok=True)  # Create directory
        with open(cache_path, "w", encoding="utf-8") as f:  # Open for writing
            for s in unique_sentences:  # Write each sentence
                f.write(s + "\n")  # One per line
        logging.info(f"Cached {len(unique_sentences)} text samples to {cache_path}")  # Log success
    except Exception as e:  # Catch write errors
        logging.warning(f"Failed to write UCI sentence cache file {cache_path}: {e}")  # Log warning

    return unique_sentences  # Return unique sentences


def load_text_samples(num_samples: int = 100) -> List[str]:  # Load text samples for testing
    """Load text samples: try cache first, then download, then fallback to built-in."""
    sentences: List[str] = []  # Initialize empty list

    # Try to load from cache
    if os.path.exists(UCI_SENTENCE_CACHE):  # Check if cache exists
        logging.info(f"Loading text samples from cache: {UCI_SENTENCE_CACHE}")  # Log info
        try:  # Try to read cache
            with open(UCI_SENTENCE_CACHE, "r", encoding="utf-8") as f:  # Open cache file
                sentences = [line.strip() for line in f if line.strip()]  # Read non-empty lines
        except Exception as e:  # Catch read errors
            logging.warning(f"Failed to read cached text samples: {e}")  # Log warning
            sentences = []  # Reset on error

    # If cache miss, download
    if not sentences:  # No sentences loaded yet
        sentences = _download_uci_sentences(cache_path=UCI_SENTENCE_CACHE)  # Download

    # If download fails, use built-in fallback
    if not sentences:  # Still no sentences
        logging.warning("Using small built-in list of text samples")  # Log warning
        sentences = [  # Built-in fallback sentences
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
    if num_samples <= 0:  # Invalid num_samples
        return []  # Return empty list

    # Repeat sentences if needed to reach num_samples
    if num_samples > len(sentences):  # Need more than available
        repeats = (num_samples + len(sentences) - 1) // len(sentences)  # Ceiling division
        return (sentences * repeats)[:num_samples]  # Repeat and truncate

    return sentences[:num_samples]  # Return first N sentences


def load_samples(mode: str, num_samples: int = 100) -> List[Any]:  # Load samples based on mode
    """Load dataset samples (images or text) based on mode."""
    if mode == "image":  # Image mode
        logging.info("Loading CIFAR-10 test subset")  # Log info
        dataset = CIFAR10(root="./data", train=False, download=True)  # Download/load CIFAR-10

        max_samples = min(num_samples, len(dataset))  # Don't exceed dataset size
        samples: List[bytes] = []  # List for image bytes
        for i in range(max_samples):  # Load each image
            img = dataset[i][0]  # Get PIL Image (ignore label)
            samples.append(img_to_bytes(img))  # Convert to JPEG bytes
        return samples  # Return image bytes list
    elif mode == "text":  # Text mode
        return load_text_samples(num_samples=num_samples)  # Load text samples
    else:  # Invalid mode
        raise ValueError(f"Unknown mode: {mode}")  # Raise error


def worker_loop(  # Worker thread function
    name: str,  # Worker name for logging
    url: str,  # TorchServe URL
    model_name: str,  # Model to test
    mode: str,  # "image" or "text"
    headers: Dict[str, str],  # HTTP headers
    sample_pool: List[Any],  # Pool of samples to choose from
    task_queue: "queue.Queue[Optional[int]]",  # Queue to pull tasks from
    result_list: List[RequestResult],  # Shared results list
    result_lock: threading.Lock,  # Lock for thread-safe access
):
    """
    Worker thread that pulls requests from task_queue and sends them to TorchServe.

    Stops when receiving None token. Appends results to shared result_list (thread-safe).
    """
    while True:  # Infinite loop until poison pill
        token = task_queue.get()  # Block until task available
        try:  # Process task
            if token is None:  # Poison pill to stop
                return  # Exit thread
            if token <= 0:  # Skip non-positive tokens
                continue  # Continue to next task

            sample = random.choice(sample_pool)  # Randomly select sample
            result = send_request(url, model_name, sample, mode, headers)  # Send request
            with result_lock:  # Acquire lock for thread safety
                result_list.append(result)  # Append result to shared list
        finally:  # Always execute
            task_queue.task_done()  # Mark task as done


def schedule_burst_pattern(  # Schedule burst traffic pattern
    duration: int,  # Total duration in seconds
    burst: int,  # Mean burst duration
    idle: int,  # Mean idle duration
    base_rps: int,  # Target RPS during burst
    task_queue: "queue.Queue[Optional[int]]",  # Queue to put tokens
):
    """
    Schedule burst pattern: alternating active bursts and quiet periods.

    Durations are randomized around mean values using Gaussian distribution.
    """
    if base_rps <= 0:  # Edge case: 0 RPS
        logging.info(f"Burst pattern with base_rps={base_rps}: idle for {duration}s")  # Log info
        time.sleep(duration)  # Just sleep
        return  # Exit function

    start = time.time()  # Record start time
    while time.time() - start < duration:  # Loop until duration elapsed
        # Randomize burst and idle durations around means
        burst_dur = max(5, int(random.gauss(burst, max(1.0, burst * 0.2))))  # Random burst duration
        idle_dur = max(2, int(random.gauss(idle, max(1.0, idle * 0.3))))  # Random idle duration
        req_per_sec = max(1, int(random.gauss(base_rps, max(1.0, base_rps * 0.25))))  # Random RPS

        logging.info(f"Burst for {burst_dur}s at ~{req_per_sec} req/s")  # Log burst info
        burst_start = time.time()  # Record burst start

        # Send requests during burst period
        while time.time() - burst_start < burst_dur and time.time() - start < duration:  # During burst
            for _ in range(req_per_sec):  # Send req_per_sec tokens
                task_queue.put(1)  # 1 token = 1 request
            time.sleep(1.0)  # Wait 1 second

        if time.time() - start >= duration:  # Check if total duration exceeded
            break  # Exit loop

        logging.info(f"Idle for {idle_dur}s")  # Log idle info
        time.sleep(idle_dur)  # Sleep during idle period


def schedule_steady_pattern(  # Schedule steady traffic pattern
    duration: int,  # Total duration in seconds
    rps: int,  # Target requests per second
    task_queue: "queue.Queue[Optional[int]]",  # Queue to put tokens
):
    """
    Schedule steady pattern: deterministic fixed requests-per-second.

    Uses precise inter-arrival timing to maintain constant RPS.
    """
    if rps <= 0:  # Edge case: 0 RPS
        logging.info(f"Steady load with rps={rps}: idle for {duration}s")  # Log info
        time.sleep(duration)  # Just sleep
        return  # Exit function

    logging.info(f"Steady load: {rps} req/s for {duration}s")  # Log load info
    start = time.time()  # Record start time
    inter_arrival = 1.0 / float(rps)  # Time between requests
    next_time = start  # Next scheduled time

    while time.time() - start < duration:  # Loop until duration elapsed
        now = time.time()  # Current time
        if now >= next_time:  # Time to send request
            task_queue.put(1)  # Send token
            next_time += inter_arrival  # Schedule next
        else:  # Not yet time
            # Sleep until next scheduled time
            sleep_for = next_time - now  # Calculate sleep duration
            if sleep_for > 0:  # If positive
                time.sleep(sleep_for)  # Sleep


def schedule_poisson_pattern(  # Schedule Poisson traffic pattern
    duration: int,  # Total duration in seconds
    rps: int,  # Average requests per second
    task_queue: "queue.Queue[Optional[int]]",  # Queue to put tokens
):
    """
    Schedule Poisson pattern: exponential inter-arrival times (realistic traffic).

    Average RPS is maintained but timing is random (exponential distribution).
    """
    if rps <= 0:  # Edge case: 0 RPS
        logging.info(f"Poisson load with rps={rps}: idle for {duration}s")  # Log info
        time.sleep(duration)  # Just sleep
        return  # Exit function

    logging.info(f"Poisson load: avg {rps} req/s for {duration}s")  # Log load info
    start = time.time()  # Record start time
    lam = float(rps)  # Lambda = rate parameter

    while time.time() - start < duration:  # Loop until duration elapsed
        wait = random.expovariate(lam)  # Random wait (exponential distribution)
        time.sleep(wait)  # Sleep for random interval

        if time.time() - start >= duration:  # Check if exceeded duration
            break  # Exit loop

        task_queue.put(1)  # Send token


def schedule_pattern(  # Dispatcher for traffic patterns
    pattern: str,  # Pattern name
    duration: int,  # Duration in seconds
    rps: int,  # Requests per second
    burst: int,  # Burst duration
    idle: int,  # Idle duration
    task_queue: "queue.Queue[Optional[int]]",  # Queue for tokens
) -> None:  # Returns nothing
    """Dispatcher to schedule requests based on chosen pattern."""
    pattern = pattern.lower()  # Normalize to lowercase
    if pattern == "burst":  # Burst pattern
        schedule_burst_pattern(duration, burst, idle, rps, task_queue)  # Call burst scheduler
    elif pattern == "steady":  # Steady pattern
        schedule_steady_pattern(duration, rps, task_queue)  # Call steady scheduler
    elif pattern == "poisson":  # Poisson pattern
        schedule_poisson_pattern(duration, rps, task_queue)  # Call poisson scheduler
    else:  # Invalid pattern
        raise ValueError(f"Unknown pattern: {pattern}")  # Raise error


def aggregate_window_metrics(  # Aggregate per-request metrics into windows
    csv_path: str,  # Path to per-request CSV
    window_s: float = 1.0,  # Window size in seconds
    meta: Optional[Dict[str, Any]] = None,  # Metadata to add
) -> Optional[str]:  # Returns output path or None
    """
    Aggregate per-request CSV into fixed-size windows.

    Computes window-level metrics:
    - requests_started/finished: request counts
    - latency percentiles (avg, p50)
    - error_rate: % of non-200 responses
    - idle_label: 1 if no requests in window

    Outputs: <csv_path>_windows_<window_s>s.csv
    """
    try:  # Try to import pandas/numpy
        import pandas as pd  # Data manipulation library
        import numpy as np  # noqa: F401  # Numerical library
    except ImportError:  # Not available
        logging.warning("pandas/numpy not available; skipping window aggregation")  # Log warning
        return None  # Return None

    try:  # Try to read CSV
        df = pd.read_csv(csv_path)  # Load CSV into DataFrame
    except Exception as e:  # Read error
        logging.warning(f"Failed to read CSV {csv_path}: {e}")  # Log warning
        return None  # Return None

    if df.empty:  # Empty DataFrame
        logging.warning(f"No rows in {csv_path}; skipping window aggregation")  # Log warning
        return None  # Return None

    # Verify required columns exist
    required_cols = {"ts_start", "ts_end", "latency_ms", "status_code"}  # Required columns
    if not required_cols.issubset(df.columns):  # Check if all present
        logging.warning(  # Log warning
            f"{csv_path} missing required columns {required_cols}; "
            "skipping window aggregation"
        )
        return None  # Return None

    # Convert to numeric (coerce errors to NaN)
    for c in ("ts_start", "ts_end", "latency_ms", "status_code"):  # For each column
        df[c] = pd.to_numeric(df[c], errors="coerce")  # Convert to numeric

    # Remove rows with missing ts_start
    df = df.dropna(subset=["ts_start"])  # Drop rows with NaN ts_start
    if df.empty:  # All rows dropped
        logging.warning("ts_start has no valid values after cleaning; skipping")  # Log warning
        return None  # Return None

    if window_s <= 0:  # Invalid window size
        logging.warning(f"Invalid window_s={window_s}; using default 1.0s")  # Log warning
        window_s = 1.0  # Use default

    t_min = float(df["ts_start"].min())  # Earliest timestamp
    t_max = float(df["ts_start"].max())  # Latest timestamp

    # Map each request to its window index
    df["window_index"] = (  # Calculate window index
        ((df["ts_start"] - t_min) / float(window_s))  # Relative time / window size
        .astype("float64")  # Convert to float
        .floordiv(1)  # Floor division
        .astype(int)  # Convert to int
    )

    # Aggregate per window
    grouped = df.groupby("window_index")  # Group by window
    window_stats = grouped.agg(  # Aggregate metrics
        requests_started=("ts_start", "count"),  # Count requests
        avg_latency_ms=("latency_ms", "mean"),  # Mean latency
        p50_latency_ms=("latency_ms", "median"),  # Median latency
        error_rate=("status_code", lambda s: (s != 200).mean() * 100.0),  # Error percentage
    )

    # Also count requests that finished in each window
    df_end = df.dropna(subset=["ts_end"]).copy()  # Rows with valid ts_end
    if not df_end.empty:  # Has data
        df_end["window_index_end"] = (  # Calculate end window index
            ((df_end["ts_end"] - t_min) / float(window_s))  # Relative time / window size
            .astype("float64")  # Convert to float
            .floordiv(1)  # Floor division
            .astype(int)  # Convert to int
        )
        finished = df_end.groupby("window_index_end").agg(requests_finished=("ts_end", "count"))  # Count finished
        finished.index.name = "window_index"  # Rename index
        window_stats = window_stats.merge(  # Merge with window_stats
            finished, left_index=True, right_index=True, how="left"  # Left join
        )
    else:  # No data
        window_stats["requests_finished"] = 0  # Set to 0

    # Fill missing windows (no requests that period) with zeros
    max_index = int(((t_max - t_min) / float(window_s)) // 1)  # Last window index
    full_index = range(0, max_index + 1)  # Complete range
    window_stats = window_stats.reindex(full_index)  # Reindex to include all windows

    window_stats["requests_started"] = window_stats["requests_started"].fillna(0).astype(int)  # Fill NaN with 0
    window_stats["requests_finished"] = window_stats["requests_finished"].fillna(0).astype(int)  # Fill NaN with 0
    window_stats["error_rate"] = window_stats["error_rate"].fillna(0.0)  # Fill NaN with 0.0

    # Reset index and compute derived columns
    window_stats = window_stats.reset_index(names="window_index")  # Move index to column
    window_stats["window_start_ts"] = t_min + window_stats["window_index"] * float(window_s)  # Compute start timestamp

    # Optional: convert to datetime
    try:  # Try to convert
        window_stats["window_start_dt"] = pd.to_datetime(  # Convert to datetime
            window_stats["window_start_ts"], unit="s"  # Unix timestamp
        )
    except Exception:  # Conversion failed
        pass  # Ignore

    # Label windows as idle if no requests
    window_stats["is_idle"] = window_stats["requests_started"] == 0  # Boolean: True if no requests
    window_stats["idle_label"] = window_stats["is_idle"].astype(int)  # Integer: 1 if idle, 0 if busy
    window_stats["label_idle_gt"] = window_stats["idle_label"]  # Ground-truth label

    window_stats["window_s"] = float(window_s)  # Record window size
    window_stats["rps"] = window_stats["requests_started"] / float(window_s)  # Compute RPS

    # Attach metadata (e.g., model_name, pattern, concurrency)
    if meta is not None:  # Has metadata
        for k, v in meta.items():  # For each key-value
            window_stats[str(k)] = v  # Add as column

    # Write output
    suffix = make_window_suffix(window_s)  # Create suffix
    out_path = csv_path.replace(".csv", f"_windows_{suffix}s.csv")  # Create output path

    try:  # Try to write
        window_stats.to_csv(out_path, index=False)  # Write CSV
        logging.info(f"Wrote window-level metrics to {out_path}")  # Log success
        return out_path  # Return path
    except Exception as e:  # Write error
        logging.warning(f"Failed to write window stats CSV {out_path}: {e}")  # Log warning
        return None  # Return None


def power_sampler_loop(  # Background thread for power sampling
    csv_path: str,  # Output CSV path
    sample_period_s: float,  # Sampling interval
    device_index: int,  # GPU device index
    stop_event: threading.Event,  # Event to signal stop
) -> None:  # Returns nothing
    """
    Background thread that periodically samples GPU power via NVML.

    Writes: timestamp, power_w, energy_j (≈ power * sample_period_s)
    Stops when stop_event is set.
    """
    try:  # Try to import pynvml
        import pynvml  # type: ignore[import]  # NVIDIA Management Library
    except Exception as e:  # Import failed
        logging.warning("pynvml/NVML not available; power sampling disabled: %s", e)  # Log warning
        return  # Exit function

    try:  # Try to initialize NVML
        pynvml.nvmlInit()  # Initialize NVML library
    except Exception as e:  # Initialization failed
        logging.warning("Failed to initialize NVML; power sampling disabled: %s", e)  # Log warning
        return  # Exit function

    try:  # Try to get GPU handle
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)  # Get handle to GPU
    except Exception as e:  # Failed to get handle
        logging.warning(  # Log warning
            "Failed to get NVML handle for device index %d: %s", device_index, e
        )
        try:  # Try to shutdown NVML
            pynvml.nvmlShutdown()  # Shutdown NVML
        except Exception:  # Shutdown failed
            pass  # Ignore
        return  # Exit function

    csv_dir = os.path.dirname(csv_path)  # Get directory part
    if csv_dir:  # If not empty
        os.makedirs(csv_dir, exist_ok=True)  # Create directory

    try:  # Try to open file
        f = open(csv_path, "w", newline="")  # Open for writing
    except Exception as e:  # Open failed
        logging.warning("Failed to open power CSV %s: %s", csv_path, e)  # Log warning
        try:  # Try to shutdown NVML
            pynvml.nvmlShutdown()  # Shutdown NVML
        except Exception:  # Shutdown failed
            pass  # Ignore
        return  # Exit function

    # Write CSV header
    writer = csv.DictWriter(f, fieldnames=["timestamp", "power_w", "energy_j"])  # Create CSV writer
    writer.writeheader()  # Write header row
    f.flush()  # Flush to disk

    logging.info(  # Log info
        "Starting power sampler: device_index=%d, period=%.3fs, csv=%s",
        device_index,  # GPU index
        sample_period_s,  # Sampling period
        csv_path,  # Output path
    )

    try:  # Main sampling loop
        while not stop_event.is_set():  # Loop until stop signaled
            t0 = time.time()  # Record sample time
            try:  # Try to query power
                # Query GPU power in milliwatts, convert to watts
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # Get power in mW
                power_w = float(power_mw) / 1000.0  # Convert to watts
                energy_j = power_w * float(sample_period_s)  # Energy = Power × Time
            except Exception as e:  # Query failed
                logging.warning("Failed to query NVML power: %s", e)  # Log warning
                power_w = float("nan")  # Use NaN
                energy_j = float("nan")  # Use NaN

            writer.writerow(  # Write sample to CSV
                {
                    "timestamp": f"{t0:.6f}",  # 6 decimal places
                    "power_w": f"{power_w:.6f}",  # 6 decimal places
                    "energy_j": f"{energy_j:.6f}",  # 6 decimal places
                }
            )
            f.flush()  # Flush to disk immediately

            # Sleep to maintain target sample period
            elapsed = time.time() - t0  # Time spent sampling
            sleep_for = float(sample_period_s) - elapsed  # Remaining time
            if sleep_for > 0:  # If positive
                time.sleep(sleep_for)  # Sleep
    finally:  # Cleanup
        try:  # Try to close file
            f.close()  # Close file
        except Exception:  # Close failed
            pass  # Ignore
        try:  # Try to shutdown NVML
            pynvml.nvmlShutdown()  # Shutdown NVML
        except Exception:  # Shutdown failed
            pass  # Ignore

    logging.info("Power sampler stopped")  # Log info


def start_power_sampler(  # Start power sampling thread
    csv_path: Optional[str],  # Output path (None to disable)
    sample_period_s: float,  # Sampling interval
    device_index: int,  # GPU index
) -> (Optional[threading.Thread], Optional[threading.Event]):  # Returns (thread, stop_event)
    """Start a background thread for power sampling; return (thread, stop_event)."""
    if not csv_path:  # Disabled
        return None, None  # Return None, None

    stop_event: threading.Event = threading.Event()  # Create stop event
    thread = threading.Thread(  # Create thread
        target=power_sampler_loop,  # Function to run
        args=(csv_path, sample_period_s, device_index, stop_event),  # Arguments
        daemon=True,  # Daemon thread (dies with main)
    )
    try:  # Try to start
        thread.start()  # Start thread
        return thread, stop_event  # Return both
    except Exception as e:  # Start failed
        logging.warning("Failed to start power sampler: %s", e)  # Log warning
        return None, None  # Return None, None


def attach_energy_to_windows(  # Attach energy data to window CSV
    windows_csv_path: str,  # Path to window CSV
    energy_csv_path: str,  # Path to power samples CSV
    energy_ts_col: str = "timestamp",  # Timestamp column name
    energy_col: str = "energy_j",  # Energy column name
    idle_power_threshold_w: Optional[float] = None,  # Manual threshold
    idle_calibration_seconds: Optional[float] = None,  # Auto-calibration period
) -> None:  # Returns nothing
    """
    Join per-sample power/energy data into window CSV.

    Adds:
    - energy_j_per_window: sum of energy_j in that window
    - avg_power_w: energy / window_s
    - energy_idle_label: 1 if avg_power_w <= threshold

    Modifies windows_csv_path in-place.
    """
    try:  # Try to import
        import pandas as pd  # Data manipulation
        import numpy as np  # Numerical operations
    except ImportError:  # Import failed
        logging.warning("pandas/numpy not available; skipping energy/window join")  # Log warning
        return  # Exit function

    if not os.path.exists(windows_csv_path):  # Check windows CSV exists
        logging.warning(  # Log warning
            f"Window metrics CSV {windows_csv_path} not found; cannot attach energy"
        )
        return  # Exit function

    if not os.path.exists(energy_csv_path):  # Check energy CSV exists
        logging.warning(  # Log warning
            f"Energy CSV {energy_csv_path} not found; skipping energy/window join"
        )
        return  # Exit function

    try:  # Try to read CSVs
        windows = pd.read_csv(windows_csv_path)  # Read window CSV
        energy = pd.read_csv(energy_csv_path)  # Read energy CSV
    except Exception as e:  # Read failed
        logging.warning(f"Failed to read windows/energy CSVs: {e}")  # Log warning
        return  # Exit function

    if windows.empty or energy.empty:  # Empty data
        logging.warning("Windows or energy CSV is empty; skipping join")  # Log warning
        return  # Exit function

    # Verify required columns
    if "window_start_ts" not in windows.columns or "window_index" not in windows.columns:  # Missing columns
        logging.warning(  # Log warning
            f"{windows_csv_path} missing window_start_ts/window_index; "
            "skipping energy/window join"
        )
        return  # Exit function

    if "label_idle_gt" not in windows.columns and "idle_label" in windows.columns:  # Copy label
        windows["label_idle_gt"] = windows["idle_label"]  # Copy column

    # Extract window size from first window
    window_s = 1.0  # Default
    if "window_s" in windows.columns:  # Has window_s column
        try:  # Try to extract
            ws = float(windows["window_s"].iloc[0])  # Get first value
            if ws > 0:  # Valid
                window_s = ws  # Use it
        except Exception:  # Failed
            pass  # Use default

    t0 = windows["window_start_ts"].min()  # Get reference timestamp
    if not isinstance(t0, (int, float)) or not (t0 == t0):  # Check for NaN
        logging.warning("window_start_ts has no valid values; skipping energy join")  # Log warning
        return  # Exit function

    # Auto-calibrate idle power threshold from early samples if not provided
    auto_idle_thr_w: Optional[float] = None  # Initialize
    if (  # Conditions for auto-calibration
        (idle_power_threshold_w is None or idle_power_threshold_w <= 0.0)  # No manual threshold
        and idle_calibration_seconds is not None  # Calibration period specified
        and idle_calibration_seconds > 0.0  # Positive duration
    ):
        power_col = None  # Initialize
        for cand in ("power_w", "gpu_power_w", "power", "power_draw_w"):  # Try different names
            if cand in energy.columns:  # Found
                power_col = cand  # Use this
                break  # Stop searching

        if power_col is not None:  # Found power column
            try:  # Try to calibrate
                ts_series = pd.to_numeric(  # Convert to numeric
                    energy.get(energy_ts_col, energy.index),  # Get timestamps
                    errors="coerce",  # Invalid -> NaN
                )
                t0_energy = ts_series.min()  # Get earliest timestamp
                if t0_energy == t0_energy:  # Not NaN
                    # Use 99th percentile of early power samples as idle threshold
                    cutoff = float(t0_energy) + float(idle_calibration_seconds)  # Cutoff time
                    idle_mask = ts_series <= cutoff  # Select early samples
                    idle_power = pd.to_numeric(  # Get power values
                        energy.loc[idle_mask, power_col],  # Filter rows
                        errors="coerce",  # Invalid -> NaN
                    ).dropna()  # Remove NaN
                    if not idle_power.empty:  # Has data
                        p99_idle = float(idle_power.quantile(0.99))  # 99th percentile
                        auto_idle_thr_w = p99_idle  # Set threshold
                        logging.info(  # Log info
                            "Auto-calibrated idle power threshold from first %.1fs "
                            "of power samples -> threshold=%.2fW",
                            idle_calibration_seconds,  # Duration
                            auto_idle_thr_w,  # Threshold
                        )
            except Exception as e:  # Calibration failed
                logging.warning("Failed to auto-calibrate idle power threshold: %s", e)  # Log warning

    def _apply_energy_idle_label(df, idle_thr_w: Optional[float]) -> None:  # Helper function
        """Compute energy_idle_label based on avg_power_w and threshold."""
        if idle_thr_w is None:  # No threshold
            logging.info("No idle power threshold available; skipping energy_idle_label.")  # Log info
            return  # Exit function
        if "avg_power_w" not in df.columns:  # Missing column
            logging.info("avg_power_w not present; cannot compute energy_idle_label.")  # Log info
            return  # Exit function
        try:  # Try to compute
            mask_valid = df["avg_power_w"].notna()  # Valid power values
            df["energy_idle_label"] = np.where(  # Compute label
                mask_valid,  # Where power is valid
                (df["avg_power_w"] <= float(idle_thr_w)).astype(int),  # 1 if below threshold
                np.nan,  # NaN if no power data
            )
            df["idle_power_threshold_w"] = float(idle_thr_w)  # Record threshold
        except Exception as e:  # Failed
            logging.warning("Failed to compute energy_idle_label: %s", e)  # Log warning

    # Join energy into windows
    if energy_ts_col in energy.columns and energy_col in energy.columns:  # Has required columns
        logging.info(  # Log info
            f"Energy CSV {energy_csv_path} has per-sample columns "
            f"({energy_ts_col}, {energy_col}); using window-index-based join."
        )

        energy = energy[energy[energy_ts_col].notna()].copy()  # Remove rows without timestamp
        energy[energy_ts_col] = energy[energy_ts_col].astype("float64")  # Convert to float

        # Map each energy sample to its window
        energy["window_index"] = (  # Calculate window index
            ((energy[energy_ts_col] - float(t0)) / float(window_s))  # Relative time / window
        ).floordiv(1).astype(int)  # Floor and convert to int

        # Sum energy within each window
        energy_agg = (  # Aggregate energy
            energy.groupby("window_index", as_index=False)[energy_col]  # Group by window
            .sum()  # Sum energy
            .rename(columns={energy_col: "energy_j_per_window"})  # Rename column
        )

        # Left join: keep all windows, add energy where available
        merged = windows.merge(  # Merge dataframes
            energy_agg,  # Energy aggregates
            on="window_index",  # Join key
            how="left",  # Keep all windows
            validate="1:1",  # One-to-one relationship
        )

        # Compute average power from energy and window size
        if "energy_j_per_window" in merged.columns:  # Has energy column
            try:  # Try to compute
                merged["avg_power_w"] = (  # Compute average power
                    merged["energy_j_per_window"].astype("float64")  # Energy as float
                    / merged["window_s"].astype("float64").replace(0.0, float("nan"))  # Divide by window size
                )
            except Exception as e:  # Failed
                logging.warning("Failed to compute avg_power_w: %s", e)  # Log warning

        # Compute energy per request (for load-aware analysis)
        merged["energy_per_request"] = (  # Compute energy per request
            merged["energy_j_per_window"] / merged["requests_started"].clip(lower=1)  # Avoid division by zero
        )
        merged.loc[merged["requests_started"] == 0, "energy_per_request"] = float("nan")  # NaN for idle windows

        # Determine effective idle threshold and label windows
        effective_idle_thr_w: Optional[float]  # Declare type
        if idle_power_threshold_w is not None and idle_power_threshold_w > 0.0:  # Manual threshold
            effective_idle_thr_w = float(idle_power_threshold_w)  # Use manual
        else:  # No manual threshold
            effective_idle_thr_w = auto_idle_thr_w  # Use auto-calibrated

        _apply_energy_idle_label(merged, effective_idle_thr_w)  # Apply labels

        # Write augmented windows CSV
        try:  # Try to write
            merged.to_csv(windows_csv_path, index=False)  # Write CSV (overwrites)
            logging.info(f"Attached energy features to window metrics in {windows_csv_path}")  # Log success
        except Exception as e:  # Write failed
            logging.warning(f"Failed to write energy-augmented CSV {windows_csv_path}: {e}")  # Log warning
        return  # Exit function

    logging.warning(  # Log warning if columns missing
        f"{energy_csv_path} missing required columns for energy join. "
        f"Tried per-sample ({energy_ts_col}, {energy_col}). "
        f"Available columns: {list(energy.columns)}"
    )


def run_load(  # Main load generation orchestrator
    url: str,  # TorchServe URL
    mode: str,  # "image" or "text"
    model_name: str,  # Model name
    duration: int,  # Default duration
    burst: int,  # Default burst duration
    idle: int,  # Default idle duration
    rps: int,  # Default RPS
    headers: Dict[str, str],  # HTTP headers
    concurrency: int,  # Worker threads
    pattern: TrafficPattern,  # Default pattern
    warmup_requests: int,  # Warmup requests
    csv_path: Optional[str],  # Output CSV path
    window_s: float,  # Window size
    phases: Optional[List[Dict[str, Any]]] = None,  # Phase definitions
    energy_csv_path: Optional[str] = None,  # External energy CSV
    power_csv_path: Optional[str] = None,  # Power sampling output
    power_sample_period_s: float = 0.1,  # Power sampling interval
    power_device_index: int = 0,  # GPU index
    phases_total_seconds: Optional[int] = None,  # Time budget
    idle_power_threshold_w: Optional[float] = None,  # Manual threshold
    idle_calibration_seconds: float = 0.0,  # Calibration period
    random_seed: Optional[int] = None,  # Random seed
) -> None:  # Returns nothing
    """
    Main load generation orchestrator.

    Coordinates:
    1. Sample loading
    2. Power sampling thread
    3. Worker threads
    4. Traffic scheduling
    5. Result aggregation and energy attachment
    """
    samples = load_samples(mode)  # Load samples (images or text)

    power_thread: Optional[threading.Thread] = None  # Initialize thread
    power_stop_event: Optional[threading.Event] = None  # Initialize event
    if power_csv_path:  # Power sampling enabled
        power_thread, power_stop_event = start_power_sampler(  # Start sampler
            csv_path=power_csv_path,  # Output path
            sample_period_s=power_sample_period_s,  # Sampling interval
            device_index=power_device_index,  # GPU index
        )

    # Idle calibration: measure baseline power before sending traffic
    if idle_calibration_seconds and idle_calibration_seconds > 0.0:  # Calibration enabled
        logging.info(  # Log info
            "Idle calibration: sleeping for %.1fs with no requests to measure "
            "GPU baseline power",
            idle_calibration_seconds,  # Duration
        )
        time.sleep(idle_calibration_seconds)  # Sleep (power sampler records baseline)

    # Send warmup requests to prime the model (not recorded)
    if warmup_requests > 0:  # Warmup enabled
        logging.info(f"Warmup: sending {warmup_requests} requests (not recorded)")  # Log info
        for _ in range(warmup_requests):  # Send warmup requests
            sample = random.choice(samples)  # Random sample
            _ = send_request(url, model_name, sample, mode, headers)  # Discard result

    # Create task queue and worker threads
    task_queue: "queue.Queue[Optional[int]]" = queue.Queue()  # Thread-safe queue
    results: List[RequestResult] = []  # Shared results list
    results_lock = threading.Lock()  # Lock for thread safety

    threads: List[threading.Thread] = []  # List of worker threads
    for i in range(concurrency):  # Create N workers
        t = threading.Thread(  # Create thread
            target=worker_loop,  # Worker function
            args=(  # Arguments
                f"worker-{i}",  # Worker name
                url,  # TorchServe URL
                model_name,  # Model name
                mode,  # "image" or "text"
                headers,  # HTTP headers
                samples,  # Sample pool
                task_queue,  # Task queue
                results,  # Shared results
                results_lock,  # Lock
            ),
            daemon=True,  # Daemon thread
        )
        t.start()  # Start worker
        threads.append(t)  # Add to list

    phases_used_names: List[str] = []  # Track phases used
    if phases:  # Has phases
        for p in phases:  # For each phase
            nm = p.get("name")  # Get name
            if nm:  # Has name
                phases_used_names.append(nm)  # Add to list

    if phases:  # Multi-phase mode
        phases_list = list(phases)  # Copy list
        base_total_phase_duration = sum(int(p.get("duration", 0)) for p in phases_list)  # Total duration

        if phases_total_seconds is not None and phases_total_seconds > 0:  # Budget mode
            logging.info(  # Log info
                f"Running phases up to ~{phases_total_seconds}s "
                f"(one pass ~{base_total_phase_duration}s)"
            )

            total_scheduled = 0  # Track scheduled time
            round_idx = 0  # Track rounds

            while total_scheduled < phases_total_seconds:  # Until budget exhausted
                round_idx += 1  # Increment round
                random.shuffle(phases_list)  # Shuffle phases
                for phase in phases_list:  # For each phase
                    if total_scheduled >= phases_total_seconds:  # Budget exhausted
                        break  # Exit loop

                    phase_pattern = (phase.get("pattern", pattern) or pattern).lower()  # Get pattern
                    orig_phase_duration = int(phase.get("duration", duration))  # Get duration
                    remaining = phases_total_seconds - total_scheduled  # Remaining budget
                    phase_duration = max(1, min(orig_phase_duration, remaining))  # Clamp duration
                    phase_rps = int(phase.get("rps", rps))  # Get RPS
                    phase_burst = int(phase.get("burst", burst))  # Get burst
                    phase_idle = int(phase.get("idle", idle))  # Get idle
                    name = phase.get("name", f"phase-round{round_idx}")  # Get name

                    logging.info(  # Log phase info
                        f"Phase round {round_idx}: {name}, "
                        f"pattern={phase_pattern}, duration={phase_duration}s, "
                        f"rps={phase_rps}, burst={phase_burst}, idle={phase_idle}"
                    )
                    schedule_pattern(  # Schedule phase
                        pattern=phase_pattern,  # Pattern
                        duration=phase_duration,  # Duration
                        rps=phase_rps,  # RPS
                        burst=phase_burst,  # Burst
                        idle=phase_idle,  # Idle
                        task_queue=task_queue,  # Queue
                    )
                    total_scheduled += phase_duration  # Update scheduled time
        else:  # Single pass mode
            total_phase_duration = base_total_phase_duration  # Total duration
            random.shuffle(phases_list)  # Shuffle phases
            logging.info(  # Log info
                f"Running {len(phases_list)} phases once in random order "
                f"(total duration ~{total_phase_duration}s)"
            )

            for idx, phase in enumerate(phases_list, start=1):  # For each phase with index
                phase_pattern = (phase.get("pattern", pattern) or pattern).lower()  # Get pattern
                phase_duration = int(phase.get("duration", duration))  # Get duration
                phase_rps = int(phase.get("rps", rps))  # Get RPS
                phase_burst = int(phase.get("burst", burst))  # Get burst
                phase_idle = int(phase.get("idle", idle))  # Get idle
                name = phase.get("name", f"phase-{idx}")  # Get name
                logging.info(  # Log phase info
                    f"Phase {idx}/{len(phases_list)}: {name}, "
                    f"pattern={phase_pattern}, duration={phase_duration}s, "
                    f"rps={phase_rps}, burst={phase_burst}, idle={phase_idle}"
                )
                schedule_pattern(  # Schedule phase
                    pattern=phase_pattern,  # Pattern
                    duration=phase_duration,  # Duration
                    rps=phase_rps,  # RPS
                    burst=phase_burst,  # Burst
                    idle=phase_idle,  # Idle
                    task_queue=task_queue,  # Queue
                )
    else:  # Single pattern mode
        schedule_pattern(  # Schedule single pattern
            pattern=pattern,  # Pattern
            duration=duration,  # Duration
            rps=rps,  # RPS
            burst=burst,  # Burst
            idle=idle,  # Idle
            task_queue=task_queue,  # Queue
        )

    # Signal workers to stop after scheduling completes
    for _ in range(concurrency):  # For each worker
        task_queue.put(None)  # Send poison pill

    # Wait for all tasks to complete
    task_queue.join()  # Block until all done

    # Wait for worker threads to finish
    for t in threads:  # For each thread
        t.join()  # Wait for completion

    # Stop power sampler
    if power_stop_event is not None and power_thread is not None:  # Power sampler running
        power_stop_event.set()  # Signal stop
        power_thread.join()  # Wait for thread

    # Compute statistics from collected results
    if not results:  # No results
        logging.warning("No successful requests recorded (results list is empty)")  # Log warning
        return  # Exit function

    latencies = [r.latency_ms for r in results if r.status_code == 200]  # Successful latencies
    errors = [r for r in results if r.status_code != 200]  # Failed requests
    total = len(results)  # Total requests
    window = max(r.ts_end for r in results) - min(r.ts_start for r in results)  # Time span
    throughput = total / window if window > 0 else float("nan")  # Requests per second

    # Compute latency percentiles
    if latencies:  # Has successful requests
        latencies_sorted = sorted(latencies)  # Sort latencies
        avg_latency = statistics.mean(latencies_sorted)  # Mean
        p50 = statistics.median(latencies_sorted)  # Median
        n = len(latencies_sorted)  # Count
        p95 = latencies_sorted[int(0.95 * (n - 1))]  # 95th percentile
        p99 = latencies_sorted[int(0.99 * (n - 1))]  # 99th percentile
    else:  # No successful requests
        avg_latency = p50 = p95 = p99 = float("nan")  # All NaN

    error_rate = len(errors) * 100.0 / total  # Error percentage

    logging.info(  # Log summary
        "Completed load test: "
        f"requests={total}, throughput={throughput:.2f} req/s, "
        f"avg={avg_latency:.2f}ms, p50={p50:.2f}ms, p95={p95:.2f}ms, "
        f"p99={p99:.2f}ms, error%={error_rate:.2f}"
    )

    # Write per-request CSV
    if csv_path:  # Output enabled
        logging.info(f"Writing per-request metrics to {csv_path}")  # Log info
        csv_dir = os.path.dirname(csv_path)  # Get directory
        if csv_dir:  # Not empty
            os.makedirs(csv_dir, exist_ok=True)  # Create directory
        try:  # Try to write
            with open(csv_path, "w", newline="") as f:  # Open file
                writer = csv.writer(f)  # Create writer
                writer.writerow(  # Write header
                    [
                        "ts_start",  # Start timestamp
                        "ts_end",  # End timestamp
                        "latency_ms",  # Latency
                        "status_code",  # HTTP status
                        "error",  # Error message
                    ]
                )
                for r in results:  # For each result
                    writer.writerow(  # Write row
                        [
                            f"{r.ts_start:.6f}",  # 6 decimals
                            f"{r.ts_end:.6f}",  # 6 decimals
                            f"{r.latency_ms:.3f}",  # 3 decimals
                            r.status_code,  # Status code
                            r.error or "",  # Error or empty
                        ]
                    )
        except Exception as e:  # Write failed
            logging.warning(f"Failed to write per-request CSV {csv_path}: {e}")  # Log warning
            return  # Exit function

        # Aggregate into windows
        meta: Dict[str, Any] = {  # Metadata dict
            "model_name": model_name,  # Model name
            "mode": mode,  # Mode
            "pattern": pattern,  # Pattern
            "concurrency": concurrency,  # Concurrency
            "window_s": window_s,  # Window size
        }
        if random_seed is not None:  # Has seed
            meta["random_seed"] = random_seed  # Add to metadata
        if phases_total_seconds is not None:  # Has budget
            meta["phases_total_seconds"] = phases_total_seconds  # Add to metadata
        if phases_used_names:  # Has phases
            try:  # Try JSON encoding
                meta["phases_used"] = json.dumps(phases_used_names)  # JSON string
            except Exception:  # Failed
                meta["phases_used"] = ",".join(phases_used_names)  # Comma-separated

        windows_csv_path = aggregate_window_metrics(  # Aggregate windows
            csv_path, window_s=window_s, meta=meta  # Arguments
        )

        # Attach energy to windows if available
        effective_energy_csv = energy_csv_path or power_csv_path  # Use either source

        if windows_csv_path and effective_energy_csv:  # Both available
            attach_energy_to_windows(  # Attach energy
                windows_csv_path=windows_csv_path,  # Windows CSV
                energy_csv_path=effective_energy_csv,  # Energy CSV
                idle_power_threshold_w=idle_power_threshold_w,  # Threshold
                idle_calibration_seconds=idle_calibration_seconds,  # Calibration
            )


def main() -> None:  # Main entry point
    """Parse command-line arguments and launch load test."""
    parser = argparse.ArgumentParser(description="TorchServe load generator")  # Create parser
    parser.add_argument(  # URL argument
        "--url",
        default=None,
        help="Base URL of TorchServe (default: http://$HOSTNAME:8080)",
    )
    parser.add_argument(  # Mode argument
        "--mode",
        choices=["image", "text"],
        default="image",
        help="Service type to test: 'image' or 'text'",
    )
    parser.add_argument(  # Model name argument
        "--model-name",
        default="resnet-18",
        help="TorchServe model name",
    )
    parser.add_argument(  # Duration argument
        "--duration",
        type=int,
        default=300,
        help="Test duration in seconds if not using --phases-json",
    )
    parser.add_argument(  # Burst argument
        "--burst",
        type=int,
        default=30,
        help="Mean burst duration (s) for 'burst' pattern",
    )
    parser.add_argument(  # Idle argument
        "--idle",
        type=int,
        default=15,
        help="Mean idle duration (s) for 'burst' pattern",
    )
    parser.add_argument(  # RPS argument
        "--rps",
        type=int,
        default=10,
        help="Target requests per second (pattern-dependent)",
    )
    parser.add_argument(  # Concurrency argument
        "--concurrency",
        type=int,
        default=8,
        help="Number of worker threads",
    )
    parser.add_argument(  # Pattern argument
        "--pattern",
        choices=["burst", "steady", "poisson"],
        default="burst",
        help="Traffic pattern: burst, steady, or poisson",
    )
    parser.add_argument(  # Warmup argument
        "--warmup-requests",
        type=int,
        default=50,
        help="Warmup requests to send before measurement starts",
    )
    parser.add_argument(  # CSV argument
        "--csv",
        default=None,
        help="Path to write per-request CSV metrics (optional)",
    )
    parser.add_argument(  # Window size argument
        "--window-s",
        type=float,
        default=1.0,
        help="Window size in seconds for aggregated metrics (default: 1.0)",
    )
    parser.add_argument(  # Token argument
        "--token",
        default=None,
        help="Auth token for TorchServe (optional)",
    )
    parser.add_argument(  # Phases JSON argument
        "--phases-json",
        default=None,
        help="Path to a JSON file describing multiple load phases",
    )
    parser.add_argument(  # Phase name argument
        "--phase-name",
        default=None,
        help=(
            "Optional: when used with --phases-json, select a single phase "
            "by its 'name'. If not set, all phases are used."
        ),
    )
    parser.add_argument(  # Phase duration scale argument
        "--phase-duration-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to all phase durations when using "
            "--phases-json (e.g., 0.5 halves all durations)."
        ),
    )
    parser.add_argument(  # Energy CSV argument
        "--energy-csv",
        default=None,
        help=(
            "Optional path to an energy CSV. Supports per-sample "
            "columns 'timestamp'/'energy_j'."
        ),
    )
    parser.add_argument(  # Power CSV argument
        "--power-csv",
        default=None,
        help=(
            "Optional path to a power sampling CSV collected via NVML "
            "(timestamp, power_w, energy_j). When provided, a background "
            "sampler will run during the load test."
        ),
    )
    parser.add_argument(  # Power sample period argument
        "--power-sample-period",
        type=float,
        default=0.1,
        help="Sampling period in seconds for --power-csv (default: 0.1)",
    )
    parser.add_argument(  # Power device index argument
        "--power-device-index",
        type=int,
        default=0,
        help="NVML GPU device index to sample with --power-csv (default: 0)",
    )
    parser.add_argument(  # Phases total seconds argument
        "--phases-total-seconds",
        type=int,
        default=None,
        help=(
            "When used with --phases-json, sets an approximate total number "
            "of seconds of scheduled phase time. Phases will be shuffled and "
            "repeated until this budget is exhausted."
        ),
    )
    parser.add_argument(  # Idle power threshold argument
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
    parser.add_argument(  # Windows CSV argument
        "--windows-csv",
        default=None,
        help=(
            "Path to an existing window-level metrics CSV to post-process "
            "with energy data (used with --attach-energy-only)."
        ),
    )
    parser.add_argument(  # Attach energy only argument
        "--attach-energy-only",
        action="store_true",
        help=(
            "Skip load generation and only attach energy data to an existing "
            "windows CSV. Requires --windows-csv and --energy-csv."
        ),
    )
    parser.add_argument(  # Random seed argument
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible load patterns.",
    )
    parser.add_argument(  # Idle calibration argument
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

    args = parser.parse_args()  # Parse arguments

    # Set global random seeds for reproducibility
    if args.random_seed is not None:  # Seed provided
        random.seed(args.random_seed)  # Set Python random seed
        try:  # Try numpy
            import numpy as _np  # Import numpy
            _np.random.seed(args.random_seed)  # Set numpy seed
        except Exception:  # Failed
            pass  # Ignore
        try:  # Try PyTorch
            import torch as _torch  # type: ignore[import]  # Import torch
            _torch.manual_seed(args.random_seed)  # Set CPU seed
            if _torch.cuda.is_available():  # Has CUDA
                _torch.cuda.manual_seed_all(args.random_seed)  # Set GPU seeds
        except Exception:  # Failed
            pass  # Ignore

    # Handle attach-energy-only mode (post-processing only)
    if args.attach_energy_only:  # Attach-only mode
        if not args.windows_csv:  # Missing required arg
            raise SystemExit("--attach-energy-only requires --windows-csv")  # Error
        if not args.energy_csv:  # Missing required arg
            raise SystemExit("--attach-energy-only requires --energy-csv")  # Error

        attach_energy_to_windows(  # Attach energy
            windows_csv_path=args.windows_csv,  # Windows CSV
            energy_csv_path=args.energy_csv,  # Energy CSV
            idle_power_threshold_w=args.idle_power_threshold_w,  # Threshold
            idle_calibration_seconds=args.idle_calibration_seconds,  # Calibration
        )
        return  # Exit after post-processing

    # Determine TorchServe URL
    url = args.url or f"http://{os.environ.get('HOSTNAME', '127.0.0.1')}:8080"  # Default URL

    # Set up auth headers if token provided
    headers: Dict[str, str] = {}  # Initialize headers
    if args.token:  # Token provided
        headers["Authorization"] = f"Bearer {args.token}"  # Add auth header

    # Load phases if provided
    phases = None  # Initialize
    if args.phases_json:  # Phases JSON provided
        phases = _load_phases_json(args.phases_json)  # Load phases

        # Filter to single phase by name if requested
        if args.phase_name:  # Phase name filter
            filtered = [p for p in phases if p.get("name") == args.phase_name]  # Filter
            if not filtered:  # Not found
                raise SystemExit(  # Error
                    f"No phase named '{args.phase_name}' found in {args.phases_json}"
                )
            phases = filtered  # Use filtered

        # Apply global duration scale to all phases
        if args.phase_duration_scale != 1.0:  # Scaling enabled
            factor = float(args.phase_duration_scale)  # Scale factor
            for p in phases:  # For each phase
                if "duration" in p:  # Has duration
                    p["duration"] = max(1, int(round(p["duration"] * factor)))  # Scale it
            logging.info(  # Log info
                "Scaled phase durations by factor %.3f based on --phase-duration-scale",
                factor,  # Factor value
            )

    # Launch the load test
    run_load(  # Run load test
        url=url,  # TorchServe URL
        mode=args.mode,  # Mode
        model_name=args.model_name,  # Model name
        duration=args.duration,  # Duration
        burst=args.burst,  # Burst
        idle=args.idle,  # Idle
        rps=args.rps,  # RPS
        headers=headers,  # Headers
        concurrency=args.concurrency,  # Concurrency
        pattern=args.pattern,  # type: ignore[arg-type]  # Pattern
        warmup_requests=args.warmup_requests,  # Warmup
        csv_path=args.csv,  # CSV path
        window_s=args.window_s,  # Window size
        phases=phases,  # Phases
        energy_csv_path=args.energy_csv,  # Energy CSV
        power_csv_path=args.power_csv,  # Power CSV
        power_sample_period_s=args.power_sample_period,  # Sample period
        power_device_index=args.power_device_index,  # Device index
        phases_total_seconds=args.phases_total_seconds,  # Time budget
        idle_power_threshold_w=args.idle_power_threshold_w,  # Threshold
        idle_calibration_seconds=args.idle_calibration_seconds,  # Calibration
        random_seed=args.random_seed,  # Seed
    )


if __name__ == "__main__":  # Script executed directly
    main()  # Run main function