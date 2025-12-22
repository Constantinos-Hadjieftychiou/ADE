WARP.md

This file provides guidance to WARP (warp.dev) and other tools when working with code in this repository.

It is written to reflect the research / thesis context of this project, not just the raw scripts. The repo implements a full experimental pipeline for:

•  Serving deep learning models with TorchServe.
•  Driving realistic multi‑phase and bursty workloads that include both busy and fully idle periods.
•  Measuring and attributing GPU energy at fine time granularity.
•  Labeling windows as idle vs. busy and computing features.
•  Merging multiple experimental runs into a single dataset that can be used to train machine‑learning models to predict idleness in microservice workloads.

There is no traditional Python package structure; most logic lives in top‑level scripts and Slurm job files.


1. Repository purpose (high level)


At a high level, this repo exists to support the thesis:

“Leveraging Machine Learning to Predict Idleness in Microservice Workloads”

The code here lets you:

1. Package CNN models (ResNet‑18, ResNet‑50, MobileNet‑V2) into TorchServe .mar archives.
2. Start a TorchServe server with consistent ports and model store.
3. Generate configurable workloads (steady, Poisson, bursty, multi‑phase traces with long idle periods).
4. Measure GPU power/energy using:
◦  Zeus wrappers around arbitrary commands.
◦  NVML power sampling (via pynvml) integrated with the load generator.
5. Aggregate per‑request metrics into windowed metrics.
6. Attach per‑window energy and power information.
7. Merge multiple runs into a single CSV suitable for ML training and analysis.

This project is focused on data generation, measurement, and labeling. The actual ML model training can be done in notebooks or a separate repo using the merged CSV outputs.


2. Key scripts and what they do


Top‑level Python scripts (main entry points):

•  build_resnet_mar.py  
  Build TorchServe .mar artifacts for:
•  resnet-18
•  resnet-50
•  mobilenet-v2  
  It:
•  Ensures index_to_name.json (ImageNet index → label) exists.
•  Produces TorchScript files (*_scripted.pt).
•  Uses torch-model-archiver to create .mar files under model_store/.
•  client.py  
  Core load generator and metrics collector. It:
•  Sends requests to TorchServe’s /predictions/{model} endpoint.
•  Supports:
◦  --mode image (CIFAR‑10 images as JPEG).
◦  --mode text (UCI sentiment sentences as JSON).
•  Traffic patterns:
◦  steady, poisson, burst.
•  Accepts either:
◦  Single pattern via CLI (--pattern, --duration, --rps), or
◦  Multi‑phase traffic via --phases-json phases.json.
•  Outputs:
◦  Per‑request CSV (--csv).
◦  Window‑level CSV (--window-s).
•  Optional NVML integration:
◦  --power-csv, --power-sample-period, --idle-calibration-seconds to record GPU power/energy.
•  compute_energy_windows.py  
  Post‑processing script to integrate NVML power samples into per‑window energy. It:
•  Reads a windows CSV (requires window_start_ts, window_s).
•  Reads a power samples CSV (timestamp + power).
•  Computes:
◦  Energy per window (energy_j).
◦  Average power per window (avg_power_w).
•  Can subtract an idle power baseline using --idle-power-threshold-w.
•  measure_with_zeus.py  
  Wrapper around an arbitrary command (e.g. client.py) that:
•  Uses Zeus (if available) to measure:
◦  Wall time
◦  Zeus time
◦  Total energy (joules) for a named window.
•  Produces:
◦  Optional JSON summary.
◦  Optional CSV log with one row per run.
•  measure_idle_baseline.py  
  Specialized Zeus script to measure idle baseline:
•  Runs a Zeus window for --duration seconds while TorchServe is up but no traffic is sent.
•  Writes:
◦  JSON summary.
◦  CSV row including avg_power_w.
•  Used as reference to distinguish idle vs. dynamic energy.
•  analyze_and_merge_runs.py  
  Analysis and merge script for window‑level CSVs across multiple runs. It:
•  Scans a root directory (e.g. runs/) for:
◦  run_*/requests_<MODEL_NAME>_windows_*s.csv
•  Reads each file, adds source_run_dir, and concatenates them.
•  Emits:
◦  Class balance for idle labels (e.g. label_idle_gt, energy_idle_label).
◦  Consistency checks between traffic‑based and energy‑based idle labels.
◦  Correlations between request volume and energy/power.
◦  Optional histograms (if matplotlib is available).
•  Writes a merged CSV, typically merged_windows_<model>.csv.
•  submit_experiments.py  
  Helper script to submit experiment batches via Slurm:
•  Reads experiments_resnet.json (list of experiment configs).
•  For each experiment, builds an sbatch --export=... environment and calls:
◦  sbatch ... job_resnet_model.sbatch
•  Encodes experiment parameters:
◦  MODEL_NAME, REPEATS, CONCURRENCY, WINDOW_S, PATTERN, IDLE_CALIBRATION_SECONDS, PHASE_DURATION_SCALE, etc.

Slurm job scripts:

•  job_resnet_model.sbatch  
  Main end‑to‑end experiment pipeline:
•  Loads modules (Python, CUDA, Java).
•  Creates/activates .venv_torchserve.
•  Installs:
◦  torch, torchvision
◦  torchserve, torch-model-archiver
◦  pillow, requests, transformers
◦  pandas, numpy, nvidia-ml-py3
◦  captum, zeus
•  Ensures .mar files exist via build_resnet_mar.py.
•  Defines:
◦  RUN_DIR under runs/<MODEL_NAME>_run_<timestamp>.
◦  Per‑repeat run_X subdirectories with config JSON.
•  Starts TorchServe with the chosen model (MODEL_NAME) using torchserve.properties.
•  For each repeat:
◦  Uses measure_with_zeus.py to wrap client.py.
◦  Uses phases from phases.json (or model‑specific phases if configured).
◦  Writes per‑request CSV, window CSV, and (optionally) power sample CSV.
◦  Optionally attaches energy features to windows.
•  Collects metrics from TorchServe (Prometheus‑style) and copies logs.
•  Stops TorchServe and finalizes outputs.
•  idle_baseline.sbatch  
  Utility job for idle baseline energy:
•  Sets up environment and virtualenv.
•  Installs required packages.
•  Builds .mar files if needed.
•  Starts TorchServe with ResNet‑18.
•  Calls measure_idle_baseline.py with --duration 600 (default) and logs CSV/JSON in:
◦  runs/resnet18_idle_baseline_<timestamp>/.
•  Stops TorchServe and tars the run directory.


3. Important configuration files


•  experiments_resnet.json  
  JSON array of experiment configs, each specifying:

•  model_name: "resnet-18", "resnet-50", "mobilenet-v2", etc.
•  repeats: number of repeats per configuration.
•  concurrency: number of concurrent client workers.
•  window_s: window size in seconds for windowed metrics.
•  pattern: "steady" or "burst" (used by job_resnet_model.sbatch together with phases).
•  idle_calibration_seconds: duration of initial idle calibration.
•  phase_duration_scale: global scaling factor for phase durations.
•  phases.json  
  Primary definition of multi‑phase traffic traces for experiments. Each phase object has:

•  name: e.g. "cold-start-steady-1rps", "mid-peak-poisson", "burst-high-30rps", "long-idle".
•  pattern: "steady", "poisson", "burst".
•  duration: seconds for that phase.
•  rps: mean RPS during active periods.
•  Optional (for "burst"):
◦  burst: mean burst duration.
◦  idle: mean idle duration between bursts.
•  phases_short.json  
  Minimal phase sequence for quick tests (e.g., a single short steady phase).

•  torchserve.properties  
  Configuration for TorchServe:

•  Inference port (typically 8080).
•  Management port (typically 8081).
•  Metrics port (typically 8082).
•  model_store=./model_store.
•  index_to_name.json  
  ImageNet index‑to‑label mapping used by ResNet/MobileNet models.


4. Common commands


4.1 Build TorchServe model archives (.mar)

Creates/updates:

•  index_to_name.json
•  TorchScript .pt files for:
◦  ResNet‑18
◦  ResNet‑50
◦  MobileNet‑V2
•  .mar files under model_store/.

Command:

python build_resnet_mar.py



4.2 Start / stop TorchServe (manual, local)

Start TorchServe with ResNet‑18:

torchserve --start --ncs \
  --disable-token-auth \
  --model-store ./model_store \
  --models resnet-18=resnet-18.mar \
  --ts-config ./torchserve.properties

Health checks:

curl -s http://127.0.0.1:8080/ping
curl -s http://127.0.0.1:8081/models
curl -s http://127.0.0.1:8082/metrics

Stop:

torchserve --stop



4.3 Run a simple local load test (no Slurm)

Example steady load (single pattern, no phases):

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

This produces:

•  runs/local_requests.csv (per‑request).
•  runs/local_requests_windows_1s.csv (windowed metrics).



4.4 Run multi‑phase load using phases.json

Run the full phase sequence once:

python client.py \
  --url http://127.0.0.1:8080 \
  --mode image \
  --model-name resnet-18 \
  --phases-json phases.json \
  --csv runs/requests.csv \
  --window-s 0.5

Run a single named phase from phases.json:

python client.py \
  --url http://127.0.0.1:8080 \
  --mode image \
  --model-name resnet-18 \
  --phases-json phases.json \
  --phase-name high-traffic-steady-25rps \
  --csv runs/requests_single_phase.csv \
  --window-s 0.5



4.5 Measure idle baseline energy with Zeus

Manual:

python measure_idle_baseline.py \
  --window-name resnet18_idle_baseline \
  --duration 600 \
  --log-csv runs/idle_baseline_resnet18.csv \
  --json-out runs/idle_baseline_resnet18.json

Slurm:

sbatch idle_baseline.sbatch

The Slurm job handles starting/stopping TorchServe and saving outputs to a timestamped run directory.



4.6 Run experiments via Slurm manifest

Submit all experiments from experiments_resnet.json:

python submit_experiments.py \
  --manifest experiments_resnet.json \
  --sbatch-script job_resnet_model.sbatch

This will:

•  Print each experiment’s environment and sbatch command.
•  Call sbatch with appropriate --export vars for job_resnet_model.sbatch.



4.7 Collect NVML power samples and attach energy

Integrated power sampling (during load test):

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

Then attach energy to window CSV (post‑processing‑only path built into client.py):

python client.py \
  --attach-energy-only \
  --windows-csv runs/requests_windows_0p5s.csv \
  --energy-csv runs/power_samples.csv \
  --idle-calibration-seconds 60

Alternatively, use compute_energy_windows.py for more flexible analysis:

python compute_energy_windows.py \
  --windows-csv runs/requests_windows_0p5s.csv \
  --power-csv runs/power_samples.csv \
  --power-sample-period 0.1 \
  --idle-power-threshold-w 0.0



4.8 Merge runs into a single ML‑ready CSV

After experiments finish and window CSVs exist under runs/:

python analyze_and_merge_runs.py \
  --runs-root runs \
  --model-name resnet-18 \
  --output-csv runs/merged_windows_resnet18.csv

This:

•  Finds all run_*/requests_resnet-18_windows_*s.csv.
•  Merges them.
•  Prints class balance and consistency checks for idle labels.
•  Optionally generates plots (if matplotlib available).
•  Writes runs/merged_windows_resnet18.csv.


5. Architecture map (how things fit together)


Model packaging:

•  build_resnet_mar.py
◦  Creates TorchScript .pt files for ResNet‑18/50 and MobileNet‑V2.
◦  Creates .mar archives in model_store/.
◦  Depends on torch, torchvision, and torch-model-archiver.

Serving:

•  torchserve.properties
◦  Sets ports and model store path.
•  TorchServe (invoked manually or via job scripts):
◦  Runs models as a microservice.
◦  Exposes:
▪  /predictions/{model} for inference.
▪  Management and metrics endpoints.

Load generation + metrics:

•  client.py
◦  Sends requests in either image or text mode.
◦  Uses phases from phases.json to simulate realistic daily traffic profiles.
◦  Emits per‑request and per‑window metrics.
◦  Optionally starts an NVML power sampler thread.

Energy measurement:

•  Zeus‑based:
◦  measure_with_zeus.py wraps an arbitrary command and logs total energy.
◦  measure_idle_baseline.py measures idle energy.
•  NVML‑based:
◦  client.py uses pynvml to sample GPU power.
◦  compute_energy_windows.py integrates power traces into window energy.

Aggregation and labeling:

•  analyze_and_merge_runs.py
◦  Combines window CSVs from many runs.
◦  Computes idle labels (traffic‑based and energy‑based).
◦  Performs sanity checks and correlation analysis.

Automation:

•  experiments_resnet.json + submit_experiments.py
◦  Define experiment grids and submit them via sbatch.
•  job_resnet_model.sbatch and idle_baseline.sbatch
◦  Encode the authoritative, reproducible workflows for experiments and idle baselines.


6. Gotchas and automation notes


•  There is no unified tests/CI:
◦  No dedicated pytest or lint configuration; scripts are executed directly.
•  Dependencies:
◦  Heavy dependencies (PyTorch, TorchServe, Zeus, NVML bindings, etc.) are installed at runtime in the Slurm jobs.
◦  Some features are optional:
▪  If pandas/numpy are missing, client.py may skip window aggregation.
▪  If pynvml is missing, power sampling is disabled.
•  Generated directories:
◦  model_store/, runs/, logs/ may be large and are often ignored by git.
◦  Reproducing past runs usually requires re‑running experiments, not just pulling artifacts.
•  Network requirements:
◦  client.py in text mode downloads the UCI sentiment dataset on first use (stored in data/).
◦  Ensure network access or pre‑populate data/uci_sentiment_sentences.txt.


7. How to use this repo as an AI agent (Warp)


For automated tasks (e.g., Warp’s agent):

•  Reading context:
◦  Prefer to inspect:
▪  experiments_resnet.json for experiment definitions.
▪  phases.json for traffic patterns.
▪  runs/ and logs/ for completed experiments.
•  Modifying / extending experiments:
◦  Add new entries to experiments_resnet.json if you want new combinations of:
▪  Model, concurrency, window size, pattern, phase scaling, idle calibration.
◦  Add or edit entries in phases.json to introduce new traffic phases or patterns.
•  Running pipelines:
◦  For HPC: use submit_experiments.py with job_resnet_model.sbatch.
◦  For local tests: run build_resnet_mar.py, start TorchServe manually, then run client.py and post‑processing scripts.
•  Data for ML:
◦  Use analyze_and_merge_runs.py to generate a single CSV per model.
◦  This merged CSV is the primary input to any downstream ML training code not present in this repo.

This WARP.md should be treated as the authoritative description of how the scripts, jobs, and configs work together to support your thesis experiments.