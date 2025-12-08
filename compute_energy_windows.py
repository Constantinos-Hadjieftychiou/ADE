#!/usr/bin/env python3
"""
compute_energy_windows.py

Attach per-window GPU *dynamic* energy (in joules) and average power (in watts)
to a window-level metrics CSV, using NVML power samples from another CSV.

- Dynamic power is computed as: max(0, power_w - idle_power_threshold_w)
  so samples at/below the threshold count as 0 W dynamic.

Inputs:
  --windows-csv : window-level metrics from client.py
                  must have columns: window_start_ts, window_s
  --power-csv   : NVML samples CSV
                  must have timestamp + power columns (see below)
  --power-sample-period : sampling period in seconds (e.g. 0.1)
  --idle-power-threshold-w : idle baseline in watts (optional)
  --out-csv     : output CSV (optional, default: <windows>_energy.csv)
"""

import argparse
import os

import pandas as pd


def detect_power_columns(df: pd.DataFrame):
    """
    Try to find reasonable defaults for timestamp and power columns.
    Adjust here if your actual column names differ.
    """
    # Timestamp column
    for cand in ["timestamp", "sample_ts", "time_s", "ts"]:
        if cand in df.columns:
            ts_col = cand
            break
    else:
        raise KeyError(
            "Could not find a timestamp column in power CSV. "
            f"Available columns: {list(df.columns)}. "
            "Expected one of: timestamp, sample_ts, time_s, ts"
        )

    # Power column
    for cand in ["power_w", "gpu_power_w", "power", "power_draw_w"]:
        if cand in df.columns:
            pw_col = cand
            break
    else:
        raise KeyError(
            "Could not find a power column in power CSV. "
            f"Available columns: {list(df.columns)}. "
            "Expected one of: power_w, gpu_power_w, power, power_draw_w"
        )

    return ts_col, pw_col


def compute_energy_per_window(
    windows_df: pd.DataFrame,
    power_df: pd.DataFrame,
    ts_col: str,
    pw_col: str,
    sample_period: float,
):
    """
    Compute energy [J] and avg power [W] per window using continuous power samples.

    Power sample i (P_i) is assumed constant on [t_i, t_i + sample_period).
    For each window [ws, we), we integrate overlaps with these intervals.
    """
    # Ensure sorted by time
    power_df = power_df[[ts_col, pw_col]].dropna()
    power_df = power_df.sort_values(ts_col).reset_index(drop=True)

    samples = power_df.to_numpy()  # columns: [timestamp, power_w]
    n_samples = samples.shape[0]

    if n_samples == 0:
        raise ValueError("No power samples found in power CSV.")

    if "window_start_ts" not in windows_df.columns:
        raise KeyError("windows CSV must contain 'window_start_ts' column.")
    if "window_s" not in windows_df.columns:
        raise KeyError("windows CSV must contain 'window_s' column.")

    energy_j = []
    avg_power_w = []

    j = 0  # sliding pointer into samples
    dt = float(sample_period)

    for _, row in windows_df.iterrows():
        ws = float(row["window_start_ts"])
        we = ws + float(row["window_s"])

        # Skip all samples whose *entire* interval ends before ws
        while j < n_samples and samples[j, 0] + dt <= ws:
            j += 1

        k = j
        E = 0.0

        # Accumulate contributions from overlapping sample intervals
        while k < n_samples and samples[k, 0] < we:
            s_start = float(samples[k, 0])
            s_end = s_start + dt

            overlap_start = max(ws, s_start)
            overlap_end = min(we, s_end)
            overlap = overlap_end - overlap_start

            if overlap > 0:
                P = float(samples[k, 1])  # W (already dynamic)
                E += P * overlap         # J = W * s

            k += 1

        energy_j.append(E)
        wdur = float(row["window_s"])
        avg_power_w.append(E / wdur if wdur > 0 else 0.0)

    windows_df["energy_j"] = energy_j
    windows_df["avg_power_w"] = avg_power_w

    return windows_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-csv", required=True,
                        help="Window-level metrics CSV from client.py")
    parser.add_argument("--power-csv", required=True,
                        help="NVML power samples CSV")
    parser.add_argument("--power-sample-period", type=float, required=True,
                        help="NVML sampling period in seconds (e.g. 0.1)")
    parser.add_argument("--idle-power-threshold-w", type=float, default=0.0,
                        help="Idle baseline in watts; dynamic power = max(0, P - threshold)")
    parser.add_argument("--out-csv", default=None,
                        help="Output CSV (default: <windows>_energy.csv)")
    args = parser.parse_args()

    windows_df = pd.read_csv(args.windows_csv)
    power_df = pd.read_csv(args.power_csv)

    ts_col, pw_col = detect_power_columns(power_df)

    # Convert to *dynamic* power if idle threshold is provided
    if args.idle_power_threshold_w > 0.0:
        thr = float(args.idle_power_threshold_w)
        power_df[pw_col] = (power_df[pw_col] - thr).clip(lower=0.0)

    out_df = compute_energy_per_window(
        windows_df,
        power_df,
        ts_col=ts_col,
        pw_col=pw_col,
        sample_period=args.power_sample_period,
    )

    if args.out_csv is None:
        base, ext = os.path.splitext(args.windows_csv)
        out_csv = f"{base}_energy{ext}"
    else:
        out_csv = args.out_csv

    out_df.to_csv(out_csv, index=False)

    total_energy_j = float(out_df["energy_j"].sum())
    total_time_s = float(out_df["window_s"].sum())
    avg_power = total_energy_j / total_time_s if total_time_s > 0 else 0.0

    print(f"[Energy] Wrote {out_csv}")
    print(f"[Energy] Total dynamic energy: {total_energy_j:.2f} J "
          f"over {total_time_s:.2f} s => avg dynamic power {avg_power:.2f} W")


if __name__ == "__main__":
    main()
