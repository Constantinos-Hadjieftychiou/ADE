#!/usr/bin/env python3
"""
Merge all per-run window CSVs for a TorchServe model experiment and run sanity checks.

Usage (typically from sbatch):
  python analyze_and_merge_runs.py --runs-root /path/to/RUN_DIR --model-name resnet-18
"""

import argparse
import glob
import os
from typing import List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge and analyze window-level metrics from multiple runs."
    )
    p.add_argument(
        "--runs-root",
        required=True,
        help="Top-level RUN_DIR containing run_1, run_2, ... subdirectories.",
    )
    p.add_argument(
        "--model-name",
        required=True,
        help="Model name (used only for naming the merged CSV).",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help=(
            "Path for merged CSV. "
            "Default: <runs-root>/merged_windows_<model-name>.csv"
        ),
    )
    return p.parse_args()


def find_window_csvs(runs_root: str, model_name: str) -> List[str]:
    """
    Find per-run window CSVs.
    Pattern: run_*/requests_<MODEL_NAME>_windows_*s.csv
    """
    pattern = os.path.join(
        runs_root, "run_*", f"requests_{model_name}_windows_*s.csv"
    )
    return sorted(glob.glob(pattern))


def summarize_class_balance(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found; skipping class balance.")
        return
    print(f"\n=== CLASS BALANCE for {col} ===")
    counts = df[col].value_counts(dropna=False)
    total = len(df)
    for val, cnt in counts.items():
        frac = cnt / total * 100.0
        print(f"  {col}={val}: {cnt} ({frac:.2f}%)")


def check_label_consistency(df: pd.DataFrame) -> None:
    print("\n=== LABEL CONSISTENCY CHECKS (label_idle_gt vs requests) ===")
    needed = {"requests_started", "requests_finished", "label_idle_gt"}
    if not needed.issubset(df.columns):
        print(f"[WARN] Missing one of {needed}.")
        return

    no_traffic = (df["requests_started"] == 0) & (df["requests_finished"] == 0)
    some_traffic = (df["requests_started"] > 0) | (df["requests_finished"] > 0)

    wrong_idle = df[no_traffic & (df["label_idle_gt"] != 1)]
    wrong_busy = df[some_traffic & (df["label_idle_gt"] != 0)]

    print(f"Total windows: {len(df)}")
    print(f"  No-traffic windows:   {int(no_traffic.sum())}")
    print(f"  Traffic windows:      {int(some_traffic.sum())}")
    print(f"  Inconsistent idle windows:  {len(wrong_idle)}")
    print(f"  Inconsistent busy windows:  {len(wrong_busy)}")

    if len(wrong_idle) > 0:
        print("\nExample inconsistent idle windows (first 5):")
        print(
            wrong_idle[
                ["window_index", "requests_started", "requests_finished", "label_idle_gt"]
            ].head()
        )
    if len(wrong_busy) > 0:
        print("\nExample inconsistent busy windows (first 5):")
        print(
            wrong_busy[
                ["window_index", "requests_started", "requests_finished", "label_idle_gt"]
            ].head()
        )


def maybe_plot_histograms(
    df: pd.DataFrame,
    runs_root: str,
    value_cols: Optional[list] = None,
    label_col: str = "energy_idle_label",
) -> None:
    """
    If matplotlib is available, save simple histograms of selected value columns
    split by label_col.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[INFO] matplotlib not available; skipping plots.")
        return

    if value_cols is None:
        value_cols = ["avg_power_w", "energy_j_per_window"]

    os.makedirs(runs_root, exist_ok=True)

    for vcol in value_cols:
        if vcol not in df.columns:
            continue
        if label_col not in df.columns:
            continue

        sub = df[[vcol, label_col]].dropna()
        if sub.empty:
            continue

        plt.figure(figsize=(8, 5))
        for label_val in sorted(sub[label_col].dropna().unique()):
            mask = sub[label_col] == label_val
            vals = sub.loc[mask, vcol].values
            if len(vals) == 0:
                continue
            plt.hist(vals, bins=50, alpha=0.5, label=f"{label_col}={label_val}")
        plt.xlabel(vcol)
        plt.ylabel("Count")
        plt.title(f"{vcol} by {label_col}")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(runs_root, f"{vcol}_by_{label_col}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"[OK] Saved histogram: {out_path}")


def main() -> None:
    args = parse_args()
    runs_root = os.path.abspath(args.runs_root)
    model_name = args.model_name

    if args.output_csv is None:
        sanitized_model = model_name.replace("/", "_")
        output_csv = os.path.join(
            runs_root, f"merged_windows_{sanitized_model}.csv"
        )
    else:
        output_csv = os.path.abspath(args.output_csv)

    print(f"Runs root: {runs_root}")
    print(f"Model name: {model_name}")
    print(f"Merged output CSV: {output_csv}")

    csvs = find_window_csvs(runs_root, model_name)
    if not csvs:
        print(f"[ERROR] No window CSVs found under {runs_root} for model {model_name}")
        return

    print("\nFound window CSVs:")
    for p in csvs:
        print(f"  - {p}")

    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
            df["source_run_dir"] = os.path.dirname(p)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")

    if not frames:
        print("[ERROR] No CSVs could be read successfully.")
        return

    merged = pd.concat(frames, ignore_index=True)
    print(f"\nMerged rows: {len(merged)}")

    # Basic info
    print("\n=== BASIC COLUMNS ===")
    print(", ".join(merged.columns))

    # Class balances
    summarize_class_balance(merged, "label_idle_gt")
    summarize_class_balance(merged, "energy_idle_label")

    # Consistency checks
    check_label_consistency(merged)

    # Simple stats on energy and power if available
    if "energy_j_per_window" in merged.columns:
        print("\n=== ENERGY_j_PER_WINDOW SUMMARY ===")
        print(merged["energy_j_per_window"].describe())

    if "avg_power_w" in merged.columns:
        print("\n=== AVG_POWER_W SUMMARY ===")
        print(merged["avg_power_w"].describe())

    # Some correlations that are directly interesting for the thesis
    for cols in [
        ("requests_started", "energy_j_per_window"),
        ("requests_started", "avg_power_w"),
    ]:
        c1, c2 = cols
        if c1 in merged.columns and c2 in merged.columns:
            sub = merged[[c1, c2]].dropna()
            if not sub.empty:
                corr = sub.corr().iloc[0, 1]
                print(f"\nCorr({c1}, {c2}) = {corr:.4f}")

    # Optional histograms of power/energy by energy_idle_label
    maybe_plot_histograms(merged, runs_root)

    # Save merged CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    print(f"\n[OK] Wrote merged CSV to: {output_csv}")


if __name__ == "__main__":
    main()