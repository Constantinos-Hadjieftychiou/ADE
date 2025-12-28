# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository purpose (high level)
This repo contains scripts to:
- Package a few torchvision ImageNet classifiers for TorchServe (`build_resnet_mar.py`).
- Run a configurable load generator against TorchServe (`client.py`) and emit latency/throughput metrics.
- Measure/attribute GPU energy during those runs using either:
  - Zeus (`measure_with_zeus.py`, `measure_idle_baseline.py`) or
  - NVML power sampling + post-processing (`client.py` power sampler + window/energy join, and `compute_energy_windows.py`).

There is no dedicated library/package layout; most logic lives in the top-level Python scripts.

## Common commands
### Build TorchServe model archives (.mar)
Creates/updates `index_to_name.json`, TorchScript `.pt` files, and `.mar` files under `model_store/`.

```bash
python build_resnet_mar.py
```

### Start/stop TorchServe
TorchServe settings (ports, model_store, metrics) are in `torchserve.properties`.

Example: start TorchServe with ResNet-18:

```bash
torchserve --start --ncs \
  --disable-token-auth \
  --model-store ./model_store \
  --models resnet-18=resnet-18.mar \
  --ts-config ./torchserve.properties
```

Health checks / management endpoints used by the scripts and sbatch jobs:

```bash
curl -s http://127.0.0.1:8080/ping
curl -s http://127.0.0.1:8081/models
curl -s http://127.0.0.1:8082/metrics
```

Stop TorchServe:

```bash
torchserve --stop
```

### Run a load test (local/manual)
`client.py` generates load against TorchServe’s inference API (`/predictions/{model}`) and can emit:
- Per-request metrics CSV (`--csv`)
- Aggregated per-window metrics CSV (`--window-s`)

Example steady load (single phase, no phases.json):

```bash
python client.py \
  --url http://127.0.0.1:8080 \
  --mode image \
  --model-name resnet-18 \
  --pattern steady \
  --duration 60 \
  --rps 5 \
  --concurrency 16 \
  --warmup-requests 50 \
  --csv runs/local_requests.csv \
  --window-s 1.0
```

### Run multi-phase load (from `phases.json`)
Run all phases once (in randomized order):

```bash
python client.py \
  --url http://127.0.0.1:8080 \
  --mode image \
  --model-name resnet-18 \
  --phases-json phases.json \
  --csv runs/requests.csv \
  --window-s 0.5
```

Run a single named phase (useful for “run one test” style workflows):

```bash
python client.py \
  --url http://127.0.0.1:8080 \
  --mode image \
  --model-name resnet-18 \
  --phases-json phases.json \
  --phase-name high-traffic-steady-25rps \
  --csv runs/requests_single_phase.csv \
  --window-s 0.5
```

### Measure energy with Zeus around an arbitrary command
`measure_with_zeus.py` wraps a command (everything after `--`) and records a single-row summary.

```bash
python measure_with_zeus.py \
  --window-name resnet18_phases \
  --log-csv runs/zeus_windows.csv \
  --json-out runs/zeus_summary.json \
  -- \
  python client.py --url http://127.0.0.1:8080 --mode image --model-name resnet-18 --phases-json phases.json
```

Idle baseline measurement (no traffic; intended to run while TorchServe is up):

```bash
python measure_idle_baseline.py \
  --window-name idle_baseline \
  --duration 600 \
  --log-csv runs/idle_baseline.csv \
  --json-out runs/idle_baseline.json
```

### Collect NVML power samples during a load test
`client.py` can launch a background NVML sampler when `--power-csv` is provided.

```bash
python client.py \
  --url http://127.0.0.1:8080 \
  --mode image \
  --model-name resnet-18 \
  --phases-json phases.json \
  --csv runs/requests.csv \
  --window-s 0.5 \
  --power-csv runs/power_samples.csv \
  --power-sample-period 0.1 \
  --idle-calibration-seconds 60
```

Then attach energy/power features to the window-level CSV (post-processing-only path built into `client.py`):

```bash
python client.py \
  --attach-energy-only \
  --windows-csv runs/requests_windows_0p5s.csv \
  --energy-csv runs/power_samples.csv \
  --idle-calibration-seconds 60
```

### Alternative post-processing: compute dynamic energy per window
`compute_energy_windows.py` integrates a power trace into per-window energy and supports subtracting an idle baseline via `--idle-power-threshold-w`.

```bash
python compute_energy_windows.py \
  --windows-csv runs/requests_windows_0p5s.csv \
  --power-csv runs/power_samples.csv \
  --power-sample-period 0.1 \
  --idle-power-threshold-w 0.0
```

### HPC/Slurm workflows (authoritative examples)
The following sbatch scripts encode the most complete “end-to-end” workflows (env setup → build MARs → start TorchServe → run load → collect metrics/energy → post-process → stop TorchServe):
- `job_resnet18_phases.sbatch`
- `job-single-phase.sbatch`
- `idle_baseline.sbatch`
- `job_mobilenet_v2.sbatch`
- `job_resnet50_image_burst.sbatch`

If you need exact dependency lists / install steps, these scripts are the source of truth (there is no `requirements.txt`/`pyproject.toml` in this repo).

## Architecture map (how the scripts fit together)
### Model packaging
- `build_resnet_mar.py`
  - Downloads ImageNet class labels into `index_to_name.json` (if missing).
  - Builds TorchScript artifacts (`*_scripted.pt`) for resnet18/resnet50/mobilenet_v2 (if missing).
  - Runs `torch-model-archiver` to create `model_store/{model}.mar`.

### Serving configuration
- `torchserve.properties`
  - Ports:
    - inference: 8080
    - management: 8081
    - metrics: 8082
  - Sets `model_store=./model_store`.

### Load generation + metrics
- `client.py`
  - Sends inference requests to TorchServe:
    - image mode: JPEG-encoded CIFAR-10 samples
    - text mode: sentences (cached in `data/uci_sentiment_sentences.txt` when downloaded)
  - Traffic scheduling:
    - `steady` (fixed inter-arrival)
    - `poisson` (exponential inter-arrival)
    - `burst` (on/off bursts)
  - Outputs:
    - Per-request CSV (`--csv`) with `ts_start`, `ts_end`, `latency_ms`, `status_code`, `error`.
    - Windowed CSV derived from that per-request file (filename suffix depends on `--window-s`).
  - Energy integration paths:
    - NVML sampler (`--power-csv`, `--power-sample-period`) writes `timestamp,power_w,energy_j` samples.
    - `attach_energy_to_windows()` can merge either:
      - NVML per-sample energy CSV (window-index sum), or
      - Zeus “summary CSV” (single-row: `wall_time_s, zeus_total_energy_j`) by distributing average power.

### Energy measurement wrappers
- `measure_with_zeus.py`
  - Wraps an arbitrary command and logs one summary row per run/window.
- `measure_idle_baseline.py`
  - Sleeps (no workload) for a duration and records Zeus energy/time; intended to estimate idle baseline.

## Repository “gotchas” for automation
- No test runner / lint config is currently present in-repo; the scripts are executed directly.
- Many output artifacts are intentionally ignored by git (`model_store/`, `runs/`, `logs/`, `*.mar`). See `.gitignore`.
- `client.py` optionally depends on `pandas`/`numpy` for window aggregation and on `pynvml` for power sampling; when unavailable, those features are skipped/disabled at runtime (the scripts log warnings).
