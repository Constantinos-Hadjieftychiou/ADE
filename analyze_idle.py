#!/usr/bin/env python3

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

    if len(df) == 0:
        print("CSV is empty")
        return

    # ---------------------------
    # BASELINE POWER
    # ---------------------------

    idle_df = df[df["completed_requests"] == 0]

    baseline_power = np.median(idle_df["avg_power_w"])

    print(f"Baseline GPU power: {baseline_power:.2f} W")

    threshold = baseline_power + args.epsilon_w

    df["energy_idle_window"] = df["avg_power_w"] <= threshold

    # ---------------------------
    # TRAFFIC METRICS
    # ---------------------------

    df["traffic_idle_window"] = df["requests_sent"] == 0

    df["queue_delta"] = df["requests_sent"] - df["completed_requests"]

    df["estimated_backlog"] = df["queue_delta"].cumsum()

    # ---------------------------
    # IDLE COMPARISON
    # ---------------------------

    both = np.sum(df["request_idle_window"] & df["energy_idle_window"])
    req_only = np.sum(df["request_idle_window"] & ~df["energy_idle_window"])
    energy_only = np.sum(~df["request_idle_window"] & df["energy_idle_window"])
    neither = np.sum(~df["request_idle_window"] & ~df["energy_idle_window"])

    print("\nIdle window comparison:")
    print(f"Request-idle & Energy-idle : {both}")
    print(f"Request-idle only          : {req_only}")
    print(f"Energy-idle only           : {energy_only}")
    print(f"Neither idle               : {neither}")

    # ---------------------------
    # POWER HISTOGRAM
    # ---------------------------

    plt.figure(figsize=(8,5))
    plt.hist(df["avg_power_w"], bins=50)
    plt.axvline(baseline_power, linestyle="--")
    plt.axvline(threshold, linestyle=":")
    plt.xlabel("Average power (W)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_power_hist.png")
    plt.close()

    # ---------------------------
    # BACKLOG
    # ---------------------------

    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["estimated_backlog"])
    plt.xlabel("Window")
    plt.ylabel("Estimated backlog")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_backlog.png")
    plt.close()

    # ---------------------------
    # PHASE AGGREGATION
    # ---------------------------

    grouped = df.groupby("phase")

    rps_values=[]
    mean_power=[]
    energy_per_request=[]
    misclassification=[]
    backlog=[]

    for phase,g in grouped:

        rps=g["rps"].iloc[0]

        total_energy=g["energy_j"].sum()
        total_completed=g["completed_requests"].sum()

        avg_power=g["avg_power_w"].mean()

        disagreement=np.sum(g["request_idle_window"] != g["energy_idle_window"])

        mis_rate=disagreement/len(g)

        if total_completed>0:
            e_per_req=total_energy/total_completed
        else:
            e_per_req=np.nan

        rps_values.append(rps)
        mean_power.append(avg_power)
        energy_per_request.append(e_per_req)
        misclassification.append(mis_rate)
        backlog.append(g["estimated_backlog"].mean())

    rps_values=np.array(rps_values)
    mean_power=np.array(mean_power)
    energy_per_request=np.array(energy_per_request)
    misclassification=np.array(misclassification)
    backlog=np.array(backlog)

    order=np.argsort(rps_values)

    rps_values=rps_values[order]
    mean_power=mean_power[order]
    energy_per_request=energy_per_request[order]
    misclassification=misclassification[order]
    backlog=backlog[order]

    # ---------------------------
    # PLOTS
    # ---------------------------

    plt.figure(figsize=(6,4))
    plt.plot(rps_values, mean_power, marker="o")
    plt.xlabel("RPS")
    plt.ylabel("GPU Power (W)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_power_vs_rps.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(rps_values, energy_per_request, marker="o")
    plt.xlabel("RPS")
    plt.ylabel("Energy per request (J)")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_energy_per_request_vs_rps.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(rps_values, misclassification, marker="o")
    plt.xlabel("RPS")
    plt.ylabel("Idle misclassification")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_misclassification_vs_rps.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(rps_values, backlog, marker="o")
    plt.xlabel("RPS")
    plt.ylabel("Backlog")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_backlog_vs_rps.png")
    plt.close()

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()