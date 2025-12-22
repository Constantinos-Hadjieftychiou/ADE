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
    """
    Aggregate per-request metrics into fixed windows.

    window_s may be < 1.0 for sub-second windows; the resulting CSV
    will have:
      - window_index
      - window_start_ts
      - window_start_dt
      - requests_started
      - avg_latency_ms
      - p50_latency_ms
      - error_rate
      - rps (requests_started / window_s)
      - is_idle
      - idle_label
      - label_idle_gt (alias of idle_label; 1 if no requests in the window)
      - window_s (constant column)
    """
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

    if window_s <= 0:
        logging.warning(
            f"Invalid window_s={window_s}; using default 1.0s for aggregation"
        )
        window_s = 1.0

    t_min = df["ts_start"].min()
    t_max = df["ts_start"].max()
    if t_min is None or (isinstance(t_min, float) and (t_min != t_min)):
        logging.warning("ts_start has no valid values; skipping window aggregation")
        return None

    df["window_index"] = (
        ((df["ts_start"] - t_min) / window_s).astype("float64").floordiv(1).astype(int)
    )
    grouped = df.groupby("window_index")

    window_stats = grouped.agg(
        requests_started=("ts_start", "count"),
        avg_latency_ms=("latency_ms", "mean"),
        p50_latency_ms=("latency_ms", "median"),
        error_rate=("status_code", lambda s: (s != 200).mean() * 100.0),
    )

    max_index = int(((t_max - t_min) / window_s) // 1)
    full_index = pd.Index(range(0, max_index + 1), name="window_index")
    window_stats = window_stats.reindex(full_index)

    window_stats["requests_started"] = window_stats["requests_started"].fillna(0).astype(
        int
    )
    window_stats["error_rate"] = window_stats["error_rate"].fillna(0.0)

    window_stats = window_stats.reset_index()

    window_stats["window_start_ts"] = (
        t_min + window_stats["window_index"] * float(window_s)
    )

    try:
        window_stats["window_start_dt"] = pd.to_datetime(
            window_stats["window_start_ts"], unit="s"
        )
    except Exception:
        pass

    # Ground-truth idle from traffic: no requests in window
    window_stats["is_idle"] = window_stats["requests_started"] == 0
    window_stats["idle_label"] = window_stats["is_idle"].astype(int)
    # Explicit alias for modelling: label_idle_gt
    window_stats["label_idle_gt"] = window_stats["idle_label"]

    # Effective requests-per-second in each window
    window_stats["window_s"] = float(window_s)
    window_stats["rps"] = (
        window_stats["requests_started"] / window_stats["window_s"].replace(0.0, 1.0)
    )

    # Make the output file name stable for both integer and fractional window_s
    suffix = f"{window_s:.3f}".rstrip("0").rstrip(".").replace(".", "p")
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
    Background loop that samples GPU power via NVML and writes:
      timestamp (s since epoch),
      power_w,
      energy_j (approx power * sample_period_s)
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

            elapsed = time.time() - t0
            sleep_for = sample_period_s - elapsed
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
    """
    Helper to start the power sampler thread if csv_path is provided.
    Returns (thread, stop_event) or (None, None) if disabled/failed.
    """
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
    Attach energy information to window-level metrics.

    Supports two kinds of energy CSVs:

    1) Per-sample energy CSV (preferred, e.g. NVML sampler):
       - Columns: `energy_ts_col` (default: 'timestamp'),
                  `energy_col`   (default: 'energy_j')
       - `energy_j` is the Joules for that sample interval.
       We compute window_index for each energy sample based on
       the windows' time origin and window_s, aggregate by
       window_index, and join.

       If `idle_power_threshold_w` is not provided (>0) and
       `idle_calibration_seconds` > 0, we will automatically
       estimate an idle power threshold from the first
       `idle_calibration_seconds` seconds of power samples
       (using a high percentile, currently p99).

    2) Zeus summary CSV from measure_with_zeus.py:
       - Columns: 'wall_time_s', 'zeus_total_energy_j'
       - Single-row summary for the whole run
       In this case we compute a *constant* per-window energy based on
       average power = total_energy / wall_time and the window size.

    Energy-based idleness is always defined in terms of average power
    in watts per window, not raw joules.
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

    if "window_start_ts" not in windows.columns or "window_index" not in windows.columns:
        logging.warning(
            f"{windows_csv_path} missing window_start_ts/window_index; "
            "skipping energy/window join"
        )
        return

    # Ensure we always have a ground truth idle label column
    if "label_idle_gt" not in windows.columns and "idle_label" in windows.columns:
        windows["label_idle_gt"] = windows["idle_label"]

    # Determine the window size used during aggregation (defaults to 1.0)
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

    # ------------------------------------------------------------------
    # Optional: auto-calibrate idle power threshold from power samples.
    # We do this *before* we aggregate energy so it works even if some
    # idle time occurs before the first request window.
    # ------------------------------------------------------------------
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

        if power_col is None:
            logging.info(
                "Energy CSV has no explicit power column; "
                "skipping idle power auto-calibration."
            )
        else:
            try:
                ts_series = pd.to_numeric(
                    energy.get(energy_ts_col, energy.index),
                    errors="coerce",
                )
                t0_energy = ts_series.min()
                if not (t0_energy == t0_energy):
                    logging.warning(
                        "Energy timestamps are all NaN; cannot auto-calibrate idle power"
                    )
                else:
                    cutoff = float(t0_energy) + float(idle_calibration_seconds)
                    idle_mask = ts_series <= cutoff
                    idle_power = pd.to_numeric(
                        energy.loc[idle_mask, power_col],
                        errors="coerce",
                    ).dropna()
                    if idle_power.empty:
                        logging.warning(
                            "No power samples found in first %.1fs of energy CSV; "
                            "skipping idle power auto-calibration",
                            idle_calibration_seconds,
                        )
                    else:
                        mean_idle = float(idle_power.mean())
                        std_idle = float(idle_power.std(ddof=0))
                        p95_idle = float(idle_power.quantile(0.95))
                        p99_idle = float(idle_power.quantile(0.99))
                        auto_idle_thr_w = p99_idle
                        logging.info(
                            "Auto-calibrated idle power threshold from first "
                            "%.1fs of power samples: mean=%.2fW, std=%.2fW, "
                            "p95=%.2fW, p99=%.2fW -> threshold=%.2fW",
                            idle_calibration_seconds,
                            mean_idle,
                            std_idle,
                            p95_idle,
                            p99_idle,
                            auto_idle_thr_w,
                        )
            except Exception as e:
                logging.warning("Failed to auto-calibrate idle power threshold: %s", e)

    # Helper for applying energy-based idle label in a consistent way
    def _apply_energy_idle_label(df: "pd.DataFrame", idle_thr_w: Optional[float]) -> None:
        if idle_thr_w is None:
            logging.info(
                "No idle power threshold available; skipping energy_idle_label."
            )
            return
        if "avg_power_w" not in df.columns:
            logging.info(
                "avg_power_w not present in merged DataFrame; "
                "cannot compute energy_idle_label."
            )
            return
        try:
            mask_valid = df["avg_power_w"].notna()
            df["energy_idle_label"] = np.where(
                mask_valid,
                (df["avg_power_w"] <= float(idle_thr_w)).astype(int),
                np.nan,
            )
            n_idle = int((df["energy_idle_label"] == 1).sum())
            n_total = int(mask_valid.sum())
            logging.info(
                "Energy-idle threshold %.2fW: %d/%d windows flagged idle",
                float(idle_thr_w),
                n_idle,
                n_total,
            )
            df["idle_power_threshold_w"] = float(idle_thr_w)
        except Exception as e:
            logging.warning("Failed to compute energy_idle_label: %s", e)

    # ------------------------------------------------------------------
    # CASE 1: Per-sample energy CSV with timestamp + energy_j
    # ------------------------------------------------------------------
    if energy_ts_col in energy.columns and energy_col in energy.columns:
        logging.info(
            f"Energy CSV {energy_csv_path} has per-sample columns "
            f"({energy_ts_col}, {energy_col}); using window-index-based join."
        )

        # Filter to valid timestamps
        energy = energy[energy[energy_ts_col].notna()].copy()

        # Map each energy sample to the same window_index convention
        energy["window_index"] = (
            (
                energy[energy_ts_col].astype("float64")  # type: ignore[index]
                - float(t0)
            )
            / float(window_s)
        ).floordiv(1).astype(int)

        energy_agg = (
            energy.groupby("window_index", as_index=False)[energy_col]
            .sum()
            .rename(columns={energy_col: "energy_j_per_window"})
        )

        merged = windows.merge(
            energy_agg,
            on="window_index",
            how="left",
            validate="1:1",
        )

        # Average power = energy / window_s
        if "energy_j_per_window" in merged.columns:
            try:
                merged["avg_power_w"] = (
                    merged["energy_j_per_window"].astype("float64")
                    / merged["window_s"].astype("float64").replace(0.0, np.nan)
                )
            except Exception as e:
                logging.warning("Failed to compute avg_power_w: %s", e)

        merged["energy_per_request"] = (
            merged["energy_j_per_window"]
            / merged["requests_started"].clip(lower=1)
        )
        merged.loc[merged["requests_started"] == 0, "energy_per_request"] = np.nan

        # Some helpful correlations / stats
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

        if (
            "idle_label" in merged.columns
            and "energy_j_per_window" in merged.columns
        ):
            try:
                idle_energy = merged.loc[
                    merged["idle_label"] == 1, "energy_j_per_window"
                ].dropna()
                busy_energy = merged.loc[
                    merged["idle_label"] == 0, "energy_j_per_window"
                ].dropna()
                if not idle_energy.empty and not busy_energy.empty:
                    logging.info(
                        "Idle vs busy energy_j_per_window: "
                        f"idle_mean={idle_energy.mean():.2f}J, "
                        f"busy_mean={busy_energy.mean():.2f}J"
                    )
            except Exception as e:
                logging.warning(f"Failed to compute idle/busy energy stats: {e}")

        # Decide which idle power threshold to use (manual beats auto)
        effective_idle_thr_w: Optional[float]
        if idle_power_threshold_w is not None and idle_power_threshold_w > 0.0:
            effective_idle_thr_w = float(idle_power_threshold_w)
            logging.info(
                "Using user-provided idle_power_threshold_w=%.2fW",
                effective_idle_thr_w,
            )
        else:
            effective_idle_thr_w = auto_idle_thr_w

        _apply_energy_idle_label(merged, effective_idle_thr_w)

        try:
            merged.to_csv(windows_csv_path, index=False)
            logging.info(
                f"Attached energy features to window metrics in {windows_csv_path}"
            )
        except Exception as e:
            logging.warning(
                f"Failed to write energy-augmented CSV {windows_csv_path}: {e}"
            )

        return  # Done with case 1

    # ------------------------------------------------------------------
    # CASE 2: Zeus summary CSV (wall_time_s, zeus_total_energy_j)
    # ------------------------------------------------------------------
    if {"wall_time_s", "zeus_total_energy_j"}.issubset(energy.columns):
        logging.info(
            f"Energy CSV {energy_csv_path} looks like a Zeus summary "
            "(wall_time_s, zeus_total_energy_j); deriving per-window energy "
            "from average power."
        )

        summary = energy.iloc[0]
        try:
            wall_time_s = float(summary["wall_time_s"])
            total_energy_j = float(summary["zeus_total_energy_j"])
        except Exception as e:
            logging.warning(
                f"Failed to parse wall_time_s/zeus_total_energy_j from "
                f"{energy_csv_path}: {e}"
            )
            return

        if wall_time_s <= 0 or total_energy_j < 0:
            logging.warning(
                f"Non-positive wall_time_s ({wall_time_s}) or "
                f"zeus_total_energy_j ({total_energy_j}); "
                "skipping energy/window join"
            )
            return

        avg_power_w = total_energy_j / wall_time_s
        window_duration_s = float(window_s)
        energy_per_window = avg_power_w * window_duration_s

        merged = windows.copy()
        merged["energy_j_per_window"] = energy_per_window
        merged["avg_power_w"] = avg_power_w

        merged["energy_per_request"] = (
            merged["energy_j_per_window"]
            / merged["requests_started"].clip(lower=1)
        )
        merged.loc[merged["requests_started"] == 0, "energy_per_request"] = np.nan

        # Decide threshold (manual beats auto)
        effective_idle_thr_w: Optional[float]
        if idle_power_threshold_w is not None and idle_power_threshold_w > 0.0:
            effective_idle_thr_w = float(idle_power_threshold_w)
            logging.info(
                "Using user-provided idle_power_threshold_w=%.2fW (summary-based)",
                effective_idle_thr_w,
            )
        else:
            effective_idle_thr_w = auto_idle_thr_w

        _apply_energy_idle_label(merged, effective_idle_thr_w)

        try:
            merged.to_csv(windows_csv_path, index=False)
            logging.info(
                f"Attached summary-based energy features to window metrics in "
                f"{windows_csv_path}"
            )
        except Exception as e:
            logging.warning(
                f"Failed to write energy-augmented CSV (summary) "
                f"{windows_csv_path}: {e}"
            )

        return  # Done with case 2

    # ------------------------------------------------------------------
    # CASE 3: Unknown CSV format
    # ------------------------------------------------------------------
    logging.warning(
        f"{energy_csv_path} missing required columns for energy join. "
        f"Tried per-sample ({energy_ts_col}, {energy_col}) and "
        f"summary ('wall_time_s', 'zeus_total_energy_j'). "
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
) -> None:
    samples = load_samples(mode)

    # Optional GPU power sampling (high-frequency, NVML-based)
    power_thread: Optional[threading.Thread] = None
    power_stop_event: Optional[threading.Event] = None
    if power_csv_path:
        power_thread, power_stop_event = start_power_sampler(
            csv_path=power_csv_path,
            sample_period_s=power_sample_period_s,
            device_index=power_device_index,
        )

    # Optional pre-load idle calibration period (no requests, only power sampling).
    # This lets us capture a clean GPU idle baseline at the start of the run.
    if idle_calibration_seconds and idle_calibration_seconds > 0.0:
        logging.info(
            "Idle calibration: sleeping for %.1fs with no requests to measure "
            "GPU baseline power",
            idle_calibration_seconds,
        )
        time.sleep(idle_calibration_seconds)

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

    # Stop power sampler once all load has finished
    if power_stop_event is not None and power_thread is not None:
        power_stop_event.set()
        power_thread.join()

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

        windows_csv_path = aggregate_window_metrics(csv_path, window_s=window_s)

        if windows_csv_path and energy_csv_path:
            attach_energy_to_windows(
                windows_csv_path=windows_csv_path,
                energy_csv_path=energy_csv_path,
                idle_power_threshold_w=idle_power_threshold_w,
                idle_calibration_seconds=idle_calibration_seconds,
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
        "--energy-csv",
        default=None,
        help=(
            "Optional path to an energy CSV. Supports either per-sample "
            "columns 'timestamp'/'energy_j' or a Zeus summary CSV with "
            "'wall_time_s'/'zeus_total_energy_j'."
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
            idle_calibration_seconds=args.idle_calibration_seconds,
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
        window_s=args.window_s,
        phases=phases,
        energy_csv_path=args.energy_csv,
        power_csv_path=args.power_csv,
        power_sample_period_s=args.power_sample_period,
        power_device_index=args.power_device_index,
        phases_total_seconds=args.phases_total_seconds,
        idle_power_threshold_w=args.idle_power_threshold_w,
        idle_calibration_seconds=args.idle_calibration_seconds,
    )


if __name__ == "__main__":
    main()