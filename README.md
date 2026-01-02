# ADE — TorchServe Load + NVML Power/Energy Dataset Pipeline (SLURM)

## Executive Summary

This project implements an **automated experimental pipeline** for measuring GPU power consumption and energy efficiency of deep learning inference servers. The pipeline generates controlled, reproducible workloads against TorchServe (a PyTorch model serving framework), samples GPU power via NVIDIA Management Library (NVML), and produces window-level datasets suitable for training machine learning models that predict idle/busy periods or estimate power consumption from request patterns.

**Key Deliverables:**
- Per-request latency/error metrics (CSV)
- Per-window aggregated metrics (CSV)
- GPU power samples with energy calculations (CSV)
- Merged multi-run datasets for ML training
- Reproducible, configurable workload definitions

---

## Project Architecture & Workflow

### 1. Overview: End-to-End Pipeline

---

## Key Features

### Traffic Patterns
- **`steady`**: Fixed requests-per-second (deterministic)
- **`poisson`**: Random arrival times with exponential inter-arrivals (realistic)
- **`burst`**: Alternating active/idle periods (bursty workloads)

### Multi-Phase Workloads
Define complex workload scenarios in `phases.json`:
- Each phase specifies pattern, duration, RPS, burst/idle timings
- Phases execute in randomized order (reproducible with `--random-seed`)
- Optional global duration scaling (e.g., 0.05× for quick tests)
- Per-run independent seeds for repeatable randomization

### NVML Power Sampling
- Background thread samples GPU power every ~100ms via NVIDIA Management Library
- Produces CSV with: `timestamp`, `power_w`, `energy_j`
- Optional idle calibration: measure baseline power before traffic
- Auto-calibrated idle threshold from early power samples

### Window-Level Aggregation
Aggregate per-request metrics into fixed time windows (e.g., 0.5s):
- Request counts (started/finished)
- Latency percentiles (avg, p50, p95, p99)
- Error rates
- Idle/busy classification
- Optional energy attachment (energy_j, avg_power_w, energy_idle_label)

### Multi-Run Orchestration
- `job_resnet_model.sbatch`: Run ALL phases with optional repeats
- Per-run subdirectories (`run_1/`, `run_2/`, etc.)
- Automatic merging and analysis of all runs
- Class balance and