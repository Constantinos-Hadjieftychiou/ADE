#!/usr/bin/env python3
"""
Offline analysis of window-level GPU energy measurements.

This script:
1. Estimates baseline GPU power
2. Defines energy-idle windows
3. Compares request-idle vs energy-idle
4. Generates plots using NumPy + Matplotlib
5. Computes:
   - Power vs RPS
   - Energy per Request vs RPS
   - Idle misclassification rate vs RPS
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epsilon-w", type=float, default=5.0)
    parser.add_argument("--out-prefix", default="idle_analysis")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # ------------------------------------------------------------------
    # 1️⃣ BASELINE POWER ESTIMATION
    # ------------------------------------------------------------------
    idle_df = df[df["completed_requests"] == 0]
    baseline_power = np.median(idle_df["avg_power_w"])

    print(f"Baseline GPU power: {baseline_power:.2f} W")

    threshold = baseline_power + args.epsilon_w
    df["energy_idle_window"] = df["avg_power_w"] <= threshold

    # ------------------------------------------------------------------
    # 2️⃣ IDLE LABEL COMPARISON (WINDOW LEVEL)
    # ------------------------------------------------------------------
    both = np.sum(df["request_idle_window"] & df["energy_idle_window"])
    req_only = np.sum(df["request_idle_window"] & ~df["energy_idle_window"])
    energy_only = np.sum(~df["request_idle_window"] & df["energy_idle_window"])
    neither = np.sum(~df["request_idle_window"] & ~df["energy_idle_window"])

    print("\nIdle window comparison:")
    print(f"Request-idle & Energy-idle : {both}")
    print(f"Request-idle only          : {req_only}")
    print(f"Energy-idle only           : {energy_only}")
    print(f"Neither idle               : {neither}")

    # ------------------------------------------------------------------
    # 3️⃣ ORIGINAL PLOTS (kept unchanged)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(df["avg_power_w"], bins=50)
    plt.axvline(baseline_power, linestyle="--", label="Baseline")
    plt.axvline(threshold, linestyle=":", label="Energy-idle threshold")
    plt.xlabel("Average power per window (W)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_power_hist.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.scatter(df.index, df["avg_power_w"], s=8,
                c=df["request_idle_window"])
    plt.axhline(baseline_power, linestyle="--")
    plt.axhline(threshold, linestyle=":")
    plt.xlabel("Window index")
    plt.ylabel("Average power (W)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_timeline.png")
    plt.close()

    # ------------------------------------------------------------------
    # 4️⃣ PHASE-LEVEL AGGREGATION (THIS IS THE IMPORTANT PART)
    # ------------------------------------------------------------------
    grouped = df.groupby("phase")

    rps_values = []
    mean_power = []
    energy_per_request = []
    misclassification_rate = []

    for phase, g in grouped:
        rps = g["rps"].iloc[0]

        total_energy = g["energy_j"].sum()
        total_requests = g["completed_requests"].sum()

        avg_power = g["avg_power_w"].mean()

        # misclassification = disagreement between labels
        disagreement = np.sum(g["request_idle_window"] != g["energy_idle_window"])
        mis_rate = disagreement / len(g)

        if total_requests > 0:
            e_per_req = total_energy / total_requests
        else:
            e_per_req = np.nan

        rps_values.append(rps)
        mean_power.append(avg_power)
        energy_per_request.append(e_per_req)
        misclassification_rate.append(mis_rate)

    # convert to numpy arrays for sorting
    rps_values = np.array(rps_values)
    mean_power = np.array(mean_power)
    energy_per_request = np.array(energy_per_request)
    misclassification_rate = np.array(misclassification_rate)

    order = np.argsort(rps_values)

    rps_values = rps_values[order]
    mean_power = mean_power[order]
    energy_per_request = energy_per_request[order]
    misclassification_rate = misclassification_rate[order]

    # ------------------------------------------------------------------
    # 5️⃣ POWER vs RPS
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(rps_values, mean_power, marker="o")
    plt.xlabel("Request rate (RPS)")
    plt.ylabel("Average GPU Power (W)")
    plt.title("Power vs RPS")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_power_vs_rps.png")
    plt.close()

    # ------------------------------------------------------------------
    # 6️⃣ ENERGY PER REQUEST vs RPS  (MOST IMPORTANT METRIC)
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(rps_values, energy_per_request, marker="o")
    plt.xlabel("Request rate (RPS)")
    plt.ylabel("Energy per Request (J)")
    plt.title("Energy Efficiency vs Load")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_energy_per_request_vs_rps.png")
    plt.close()

    # ------------------------------------------------------------------
    # 7️⃣ MISCLASSIFICATION RATE vs RPS
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(rps_values, misclassification_rate, marker="o")
    plt.xlabel("Request rate (RPS)")
    plt.ylabel("Idle Label Disagreement Rate")
    plt.title("Idle Detection Error vs Load")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_misclassification_vs_rps.png")
    plt.close()

    print("\nSaved derived analysis plots:")
    print(" - power_vs_rps.png")
    print(" - energy_per_request_vs_rps.png")
    print(" - misclassification_vs_rps.png")


if __name__ == "__main__":
    main()
