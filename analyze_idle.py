#!/usr/bin/env python3
"""
Offline analysis of GPU energy + request latency.

Adds:
- latency statistics
- latency plots
- automatic GPU saturation detection
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--request-log", required=True)
    parser.add_argument("--epsilon-w", type=float, default=5.0)
    parser.add_argument("--out-prefix", default="analysis")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # ----------------------------------------------------
    # BASELINE GPU POWER
    # ----------------------------------------------------

    idle_df = df[df["completed_requests"] == 0]

    baseline_power = np.median(idle_df["avg_power_w"])

    print(f"\nBaseline GPU power: {baseline_power:.2f} W")

    threshold = baseline_power + args.epsilon_w

    df["energy_idle_window"] = df["avg_power_w"] <= threshold

    # ----------------------------------------------------
    # TRAFFIC METRICS
    # ----------------------------------------------------

    df["traffic_idle_window"] = df["requests_sent"] == 0

    df["queue_delta"] = df["requests_sent"] - df["completed_requests"]

    df["estimated_backlog"] = df["queue_delta"].cumsum()

    # ----------------------------------------------------
    # IDLE LABEL COMPARISON
    # ----------------------------------------------------

    both = np.sum(df["request_idle_window"] & df["energy_idle_window"])
    req_only = np.sum(df["request_idle_window"] & ~df["energy_idle_window"])
    energy_only = np.sum(~df["request_idle_window"] & df["energy_idle_window"])
    neither = np.sum(~df["request_idle_window"] & ~df["energy_idle_window"])

    print("\nIdle window comparison:")

    print(f"Request-idle & Energy-idle : {both}")
    print(f"Request-idle only          : {req_only}")
    print(f"Energy-idle only           : {energy_only}")
    print(f"Neither idle               : {neither}")

    # ----------------------------------------------------
    # POWER HISTOGRAM
    # ----------------------------------------------------

    plt.figure(figsize=(8,5))

    plt.hist(df["avg_power_w"], bins=50)

    plt.axvline(baseline_power, linestyle="--")

    plt.axvline(threshold, linestyle=":")

    plt.xlabel("Average power (W)")
    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(f"{args.out_prefix}_power_hist.png")

    plt.close()

    # ----------------------------------------------------
    # BACKLOG EVOLUTION
    # ----------------------------------------------------

    plt.figure(figsize=(10,4))

    plt.plot(df.index, df["estimated_backlog"])

    plt.xlabel("Window")
    plt.ylabel("Estimated backlog")

    plt.title("Queue backlog evolution")

    plt.tight_layout()

    plt.savefig(f"{args.out_prefix}_backlog.png")

    plt.close()

    # ----------------------------------------------------
    # PHASE LEVEL ANALYSIS
    # ----------------------------------------------------

    grouped = df.groupby("phase")

    rps_values=[]
    mean_power=[]
    energy_per_request=[]
    backlog_mean=[]

    for phase,g in grouped:

        rps=g["rps"].iloc[0]

        total_energy=g["energy_j"].sum()

        total_completed=g["completed_requests"].sum()

        avg_power=g["avg_power_w"].mean()

        if total_completed>0:
            e_per_req=total_energy/total_completed
        else:
            e_per_req=np.nan

        rps_values.append(rps)
        mean_power.append(avg_power)
        energy_per_request.append(e_per_req)
        backlog_mean.append(g["estimated_backlog"].mean())

    rps_values=np.array(rps_values)

    order=np.argsort(rps_values)

    rps_values=rps_values[order]

    mean_power=np.array(mean_power)[order]
    energy_per_request=np.array(energy_per_request)[order]
    backlog_mean=np.array(backlog_mean)[order]

    # ----------------------------------------------------
    # POWER vs RPS
    # ----------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.plot(rps_values, mean_power, marker="o")

    plt.xlabel("RPS")
    plt.ylabel("GPU Power (W)")

    plt.tight_layout()

    plt.savefig(f"{args.out_prefix}_power_vs_rps.png")

    plt.close()

    # ----------------------------------------------------
    # ENERGY PER REQUEST
    # ----------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.plot(rps_values, energy_per_request, marker="o")

    plt.xlabel("RPS")
    plt.ylabel("Energy per request (J)")

    plt.tight_layout()

    plt.savefig(f"{args.out_prefix}_energy_per_request_vs_rps.png")

    plt.close()

    # ----------------------------------------------------
    # BACKLOG vs RPS
    # ----------------------------------------------------

    plt.figure(figsize=(6,4))

    plt.plot(rps_values, backlog_mean, marker="o")

    plt.xlabel("RPS")
    plt.ylabel("Average backlog")

    plt.title("Queue buildup vs load")

    plt.tight_layout()

    plt.savefig(f"{args.out_prefix}_backlog_vs_rps.png")

    plt.close()

    # ----------------------------------------------------
    # LATENCY ANALYSIS
    # ----------------------------------------------------

    req=pd.read_csv(args.request_log)

    req["latency"]=req["completion_ts"]-req["send_ts"]

    print("\nLatency statistics:")

    print(req["latency"].describe())

    p50=np.percentile(req["latency"],50)
    p90=np.percentile(req["latency"],90)
    p95=np.percentile(req["latency"],95)
    p99=np.percentile(req["latency"],99)

    print("\nLatency percentiles")

    print(f"p50 {p50:.6f} s")
    print(f"p90 {p90:.6f} s")
    print(f"p95 {p95:.6f} s")
    print(f"p99 {p99:.6f} s")

    # histogram

    plt.figure(figsize=(7,4))

    plt.hist(req["latency"], bins=50)

    plt.xlabel("Latency (s)")
    plt.ylabel("Requests")

    plt.tight_layout()

    plt.savefig(f"{args.out_prefix}_latency_hist.png")

    plt.close()

    # CDF

    sorted_lat=np.sort(req["latency"])

    cdf=np.arange(len(sorted_lat))/len(sorted_lat)

    plt.figure(figsize=(7,4))

    plt.plot(sorted_lat,cdf)

    plt.xlabel("Latency (s)")
    plt.ylabel("CDF")

    plt.tight_layout()

    plt.savefig(f"{args.out_prefix}_latency_cdf.png")

    plt.close()

    # ----------------------------------------------------
    # LATENCY vs RPS (Saturation detection)
    # ----------------------------------------------------

    latency_mean=req["latency"].mean()

    phase_latency=[]

    for phase,g in grouped:

        phase_latency.append(latency_mean)

    phase_latency=np.array(phase_latency)[order]

    plt.figure(figsize=(6,4))

    plt.plot(rps_values, phase_latency, marker="o")

    plt.xlabel("RPS")
    plt.ylabel("Mean latency (s)")

    plt.title("Latency vs RPS (Saturation detection)")

    plt.tight_layout()

    plt.savefig(f"{args.out_prefix}_latency_vs_rps.png")

    plt.close()

    # ----------------------------------------------------
    # AUTOMATIC SATURATION DETECTION
    # ----------------------------------------------------

    saturation_idx=np.argmax(backlog_mean>0)

    if backlog_mean[saturation_idx]>0:

        saturation_rps=rps_values[saturation_idx]

        print(f"\nEstimated GPU saturation begins near {saturation_rps} RPS")

    else:

        print("\nNo saturation detected in this RPS range")

    print("\nSaved plots including latency and saturation detection")


if __name__ == "__main__":
    main()