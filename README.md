Leveraging Machine Learning to Predict Idleness in Microservice Workloads

This repository contains the experimental harness for your thesis on:

> Leveraging machine learning to predict idleness in GPU‑accelerated microservice workloads.

The code here builds and serves image classification models with TorchServe, drives realistic multi‑phase workloads (including bursts and idle gaps), measures GPU power/energy, and post‑processes results into ML‑ready datasets for idleness prediction.

The implementation is designed primarily for HPC/Slurm environments but can be run locally for smaller tests.


1. High‑level Thesis Idea


Modern microservices backed by GPUs are often under‑utilized, with long stretches of low or zero load. These idle periods waste energy. Your thesis focuses on:

•  Simulating realistic workload patterns (steady, Poisson, bursty, long idle).
•  Measuring how these patterns translate into GPU power and energy usage.
•  Aggregating per‑request and windowed metrics (RPS, latency, status codes) with energy.
•  Generating labels for idle vs. busy windows using both traffic and energy views.
•  Producing datasets that can be used to train ML models that predict when the service is idle.

This repository handles the entire pipeline: from building models, to running Slurm jobs, to creating merged CSVs for ML.


2. Repository Layout


Key files and directories:

Top‑level scripts:
•  build_resnet_mar.py  
  Build TorchServe .mar artifacts (ResNet‑18, ResNet‑50, MobileNet‑V2).

•  client.py  
  Load generator and metrics collector (per‑request + windowed; supports NVML power sampling).

•  compute_energy_windows.py  
  Integrate power samples into per‑window energy and attach to window CSV.

•  measure_with_zeus.py  
  Wrap an arbitrary command with Zeus energy measurement.

•  measure_idle_baseline.py  
  Measure idle baseline energy (TorchServe up, no traffic) with Zeus.

•  analyze_and_merge_runs.py  
  Merge many run‑level window CSVs into a single ML‑ready dataset; check label consistency and basic stats.

•  submit_experiments.py  
  Read experiments_resnet.json and submit a grid of experiments via sbatch using job_resnet_model.sbatch.


Slurm job scripts:
•  job_resnet_model.sbatch  
  Main end‑to‑end experiment job:
•  Set up environment, install dependencies.
•  Build .mar files.
•  Start TorchServe, run multiple repeats of the workload (different patterns).
•  Optionally sample power.
•  Attach energy and collect metrics.
•  idle_baseline.sbatch  
  Spin up TorchServe, run no traffic, and measure idle baseline energy with Zeus.


Configuration / definitions:
•  experiments_resnet.json  
  Manifest of experiments (model_name, concurrency, window_s, pattern, repeats, idle calibration, phase scaling) for submit_experiments.py.

•  phases.json, phases_short.json  
  Definitions of multi‑phase traffic traces used by client.py (e.g., cold start, peak, bursts, long idle).

•  torchserve.properties  
  TorchServe configuration (ports, model store path, etc.).

•  index_to_name.json  
  ImageNet class index → label mapping used by the ResNet/MobileNet models.


Data and outputs:
•  data/  
  Contains cached datasets (e.g., uci_sentiment_sentences.txt for text workloads).

•  model_store/  
  Stores built .mar model artifacts for TorchServe.

•  runs/  
  Run outputs:
•  Per‑request CSVs
•  Window‑level CSVs
•  Power samples
•  Zeus logs
•  Config files per run
•  logs/  
  TorchServe logs (copied by the Slurm scripts).


Vendored TorchServe source:
•  serve/  
  Copy of the upstream TorchServe project (server implementation, docs, examples). The core thesis logic lives in the top‑level Python scripts above; serve/ is mostly used as a dependency and reference.


3. Workload Phases and Patterns


3.1 Phase definitions (phases.json and phases_short.json)

Your workloads are described by phase sequences. Each phase is a JSON object with at least:

•  name: Human‑readable label.
•  pattern: "steady", "poisson", or "burst".
•  duration: Length of the phase (seconds).
•  rps: Mean requests per second during active periods.
•  For pattern = "burst":
◦  burst: Mean burst duration (seconds).
◦  idle: Mean idle duration between bursts.

Example (from phases.json):

[
  {
    "name": "cold-start-steady-1rps",
    "pattern": "steady",
    "duration": 120,
    "rps": 1
  },
  {
    "name": "low-traffic-steady-2rps",
    "pattern": "steady",
    "duration": 300,
    "rps": 2
  },
  {
    "name": "low-traffic-poisson-5rps",
    "pattern": "poisson",
    "duration": 300,
    "rps": 5
  },
  {
    "name": "mid-traffic-steady-10rps",
    "pattern": "steady",
    "duration": 300,
    "rps": 10
  },
  {
    "name": "mid-peak-poisson",
    "pattern": "poisson",
    "duration": 300,
    "rps": 20
  },
  {
    "name": "burst-medium-15rps",
    "pattern": "burst",
    "duration": 240,
    "rps": 15,
    "burst": 30,
    "idle": 15
  },
  {
    "name": "burst-high-30rps",
    "pattern": "burst",
    "duration": 240,
    "rps": 30,
    "burst": 40,
    "idle": 20
  },
  {
    "name": "high-traffic-steady-25rps",
    "pattern": "steady",
    "duration": 300,
    "rps": 25
  },
  {
    "name": "evening-cooldown-steady-3rps",
    "pattern": "steady",
    "duration": 300,
    "rps": 3
  },
  {
    "name": "long-idle",
    "pattern": "steady",
    "duration": 600,
    "rps": 0
  }
]

This yields a day‑in‑the‑life trace with:

•  Warm‑up (low RPS),
•  Variable traffic (Poisson),
•  High bursts with gaps,
•  A long period of true idle (rps = 0).

For quick tests there is a shorter trace defined in phases_short.json:

[
  {
    "name": "test-steady",
    "pattern": "steady",
    "duration": 30,
    "rps": 2
  }
]


3.2 Experiment manifest (experiments_resnet.json)

experiments_resnet.json encodes the matrix of experiments to run:

[
  {
    "model_name": "resnet-18",
    "repeats": 3,
    "concurrency": 16,
    "window_s": 0.5,
    "pattern": "steady",
    "idle_calibration_seconds": 90,
    "phase_duration_scale": 1.0
  },
  {
    "model_name": "resnet-18",
    "repeats": 3,
    "concurrency": 16,
    "window_s": 0.5,
    "pattern": "burst",
    "idle_calibration_seconds": 90,
    "phase_duration_scale": 1.0
  },
  {
    "model_name": "resnet-50",
    "repeats": 3,
    "concurrency": 16,
    "window_s": 0.5,
    "pattern": "steady",
    "idle_calibration_seconds": 90,
    "phase_duration_scale": 1.0
  },
  {
    "model_name": "mobilenet-v2",
    "repeats": 3,
    "concurrency": 16,
    "window_s": 0.25,
    "pattern": "steady",
    "idle_calibration_seconds": 90,
    "phase_duration_scale": 0.5
  }
]

Each object becomes a group of Slurm jobs via submit_experiments.py and job_resnet_model.sbatch.

Across all experiments, you systematically vary:

•  Model architecture (ResNet‑18, ResNet‑50, MobileNet‑V2),
•  Concurrency (number of simultaneous clients),
•  Traffic pattern (steady vs burst),
•  Window size (temporal granularity of idle detection),
•  Phase duration scaling (for shorter/longer traces).


4. Models and Serving


4.1 Building TorchServe model archives

build_resnet_mar.py:

•  Downloads ImageNet labels into index_to_name.json (if missing).
•  Builds TorchScript models:
◦  resnet18_scripted.pt
◦  resnet50_scripted.pt
◦  mobilenet_v2_scripted.pt
•  Uses torch-model-archiver to create:
◦  model_store/resnet-18.mar
◦  model_store/resnet-50.mar
◦  model_store/mobilenet-v2.mar

Run:

python build_resnet_mar.py

This must be done before starting TorchServe in either Slurm jobs or local tests.


4.2 TorchServe configuration

torchserve.properties controls:

•  Inference port (default used by your scripts): 8080
•  Management port: 8081
•  Metrics port: 8082
•  model_store=./model_store

Manual start example (ResNet‑18):

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


5. Workload Generation (client.py)


client.py is the core load generator and metrics collector.

5.1 Modes and data

--mode image:
•  Loads CIFAR‑10 samples via torchvision.datasets.CIFAR10.
•  Converts them to JPEG bytes for POSTs to /predictions/{model}.

--mode text:
•  Downloads the UCI Sentiment Labeled Sentences dataset:
◦  URL: https://archive.ics.uci.edu/static/public/331/sentiment%2Blabelled%2Bsentences.zip
◦  Cache: data/uci_sentiment_sentences.txt
•  Sends JSON payloads {"text": "<sentence>"} with Content-Type: application/json.


5.2 Traffic patterns

The script supports three traffic patterns:

•  steady:
◦  Fixed inter‑arrival time; deterministic RPS.
•  poisson:
◦  Exponential inter‑arrival times; average RPS with jitter.
•  burst:
◦  Alternating burst and idle periods (governed by burst and idle parameters).

You can set either a single pattern (no phases) or drive a sequence of phases from phases.json.


5.3 Metrics: per‑request and per‑window

client.py always logs per‑request metrics when --csv is given:

•  ts_start, ts_end
•  latency_ms
•  status_code
•  error (if any)

If --window-s is set, it also aggregates window‑level metrics:

•  requests_started, requests_finished
•  Latency and error‑rate statistics by window
•  Time stamps for window starts
•  Optionally idle labels based on traffic (e.g. 0 requests → idle window)

Output:

•  Per‑request CSV: runs/.../requests_*.csv
•  Window CSV: runs/.../requests_*windows<suffix>s.csv
  (where <suffix> encodes the window size, e.g. 1, 0p5, 0p25)


5.4 Power sampling (NVML)

If the environment has pynvml and NVIDIA GPUs, client.py can sample GPU power:

Add flags like:

--power-csv runs/power_samples.csv \
--power-sample-period 0.1 \
--idle-calibration-seconds 60

A background thread logs power data:

•  timestamp
•  power_w
•  energy_j

If NVML is unavailable, it logs a warning and skips sampling.

After the run, you can attach power/energy to windows either:

•  Directly via client.py in attach‑only mode, or
•  Using compute_energy_windows.py (see below).


6. Energy Measurement and Idle Baseline


6.1 Zeus wrapper: measure_with_zeus.py

Wrap an arbitrary command (typically client.py) and record total time and energy using Zeus.

Example:

python measure_with_zeus.py \
  --window-name resnet18_phases \
  --log-csv runs/zeus_windows.csv \
  --json-out runs/zeus_summary.json \
  -- \
  python client.py \
    --url http://127.0.0.1:8080 \
    --mode image \
    --model-name resnet-18 \
    --phases-json phases.json \
    --csv runs/requests.csv \
    --window-s 0.5

Outputs:

•  Console log with wall time, Zeus time, and total energy.
•  Optional JSON summary.
•  Optional CSV log with columns:
◦  window_name
◦  wall_time_s
◦  zeus_time_s
◦  zeus_total_energy_j

If Zeus is not installed/importable, it prints a notice and runs the command without energy measurement.


6.2 Idle baseline: measure_idle_baseline.py and idle_baseline.sbatch

measure_idle_baseline.py measures idle baseline while TorchServe is up but there is no traffic:

python measure_idle_baseline.py \
  --window-name idle_baseline \
  --duration 600 \
  --log-csv runs/idle_baseline.csv \
  --json-out runs/idle_baseline.json

It records:

•  window_name, duration_s, wall_time_s
•  zeus_time_s, zeus_total_energy_j
•  avg_power_w (average power over the idle interval)

On the cluster, you typically use:

idle_baseline.sbatch

which:
•  Sets up the environment and virtualenv.
•  Installs required packages (PyTorch, TorchServe, Zeus, etc.).
•  Ensures .mar files exist.
•  Starts TorchServe with ResNet‑18.
•  Waits briefly, then calls measure_idle_baseline.py.
•  Stops TorchServe and packs outputs into a tarball under runs/resnet18_idle_baseline_*.

This baseline is crucial for computing dynamic power/energy (subtracting idle).


7. Post‑processing: Window Energy (compute_energy_windows.py)


compute_energy_windows.py attaches dynamic energy per window to an existing window CSV, using power samples from NVML.

Inputs:

•  --windows-csv: path to windowed metrics CSV (must contain window_start_ts and window_s).
•  --power-csv: path to power samples CSV (must contain a timestamp column like timestamp / time_s / ts and a power column like power_w / gpu_power_w).
•  --power-sample-period: e.g. 0.1 seconds.
•  --idle-power-threshold-w: baseline threshold; power below this is treated as idle and subtracted.

Example:

python compute_energy_windows.py \
  --windows-csv runs/requests_windows_0p5s.csv \
  --power-csv runs/power_samples.csv \
  --power-sample-period 0.1 \
  --idle-power-threshold-w 0.0

Outputs:

•  A new CSV (by default <windows>_energy.csv) with:
◦  energy_j_per_window (or similar energy field)
◦  avg_power_w

It also reports:

•  Total dynamic energy over all windows.
•  Total window duration.
•  Average dynamic power.


8. Aggregating Runs for ML (analyze_and_merge_runs.py)


analyze_and_merge_runs.py is where your raw experiments become ML datasets.

It:

1. Searches under a runs_root directory for files matching:
◦  run_*/requests_<MODEL_NAME>windows*s.csv
2. Reads each window CSV and adds a source_run_dir column.
3. Concatenates all runs into a single DataFrame.
4. Computes:
◦  Basic column set.
◦  Class balance for:
▪  label_idle_gt (ground‑truth idle label from traffic).
▪  energy_idle_label (label from energy/power behavior).
◦  Consistency checks:
▪  Windows with 0 requests that are not labeled idle.
▪  Windows with traffic that are labeled idle.
◦  Power/energy stats and correlations where available:
▪  Corr(requests_started, energy_j_per_window)
▪  Corr(requests_started, avg_power_w)
5. Optionally generates histograms/PNGs if matplotlib is installed.
6. Writes a merged CSV:
◦  Default name: merged_windows_<model_name>.csv under the runs root (or path given by --output-csv).

This merged CSV is the primary input for your ML training code (which can live in a separate analysis repo/notebook).


9. Experiment Automation with Slurm


9.1 Submitting experiment grids (submit_experiments.py)

submit_experiments.py reads a manifest like experiments_resnet.json and submits one or more jobs per entry:

python submit_experiments.py \
  --manifest experiments_resnet.json \
  --sbatch-script job_resnet_model.sbatch

For each experiment object, it:

•  Builds an --export environment string mapping JSON keys to uppercase env vars.
•  Calls:

sbatch --export=ALL,MODEL_NAME=...,REPEATS=...,CONCURRENCY=...,WINDOW_S=...,PATTERN=...,IDLE_CALIBRATION_SECONDS=...,PHASE_DURATION_SCALE=... \
  job_resnet_model.sbatch

job_resnet_model.sbatch then reads these env vars to configure the run.


9.2 What job_resnet_model.sbatch does

At a high level, each job:

1. Sets up environment:
◦  Loads Python, CUDA, Java modules.
◦  Creates/activates a dedicated virtualenv (.venv_torchserve).
◦  Installs:
▪  torch, torchvision
▪  torchserve, torch-model-archiver
▪  pillow, requests, transformers
▪  pandas, numpy, nvidia-ml-py3
▪  captum, zeus
2. Builds .mar files:
◦  Calls build_resnet_mar.py (safe to rerun; skips if artifacts exist).
3. Configures run directories:
◦  RUN_DIR under runs/<MODEL_NAME>run<timestamp>.
◦  Per‑repeat run_X subdirectories with JSON config files.
4. Starts TorchServe with the chosen model (MODEL_NAME) and torchserve.properties.
5. Runs the workload for each repeat:
◦  Uses measure_with_zeus.py around client.py.
◦  Uses phases.json or model‑specific phases file if present.
◦  Logs:
▪  Per‑request CSV.
▪  Window CSV.
▪  Power samples CSV (if NVML enabled).
◦  Optionally calls client.py or compute_energy_windows.py to attach energy to windows.
6. Collects metrics:
◦  Fetches TorchServe metrics endpoint and writes Prometheus metrics to a file.
◦  Copies TorchServe logs into each run directory.
7. Stops TorchServe and finalizes outputs (tarballs, etc).

The result is a structured set of runs under runs/ that can be aggregated by analyze_and_merge_runs.py.


10. Typical End‑to‑End Workflow


For cluster/HPC:

1. Build models and (optionally) test locally:

   python build_resnet_mar.py

2. Run idle baseline (once per GPU/system config):

   sbatch idle_baseline.sbatch

3. Run experiment grid:

   python submit_experiments.py \
     --manifest experiments_resnet.json \
     --sbatch-script job_resnet_model.sbatch

4. After jobs finish, merge and analyze for a given model (example: ResNet‑18):

   python analyze_and_merge_runs.py \
     --runs-root runs \
     --model-name resnet-18 \
     --output-csv runs/merged_windows_resnet18.csv

5. Use runs/merged_windows_resnet18.csv as the input dataset for your ML experiments:
◦  Features: request/latency stats, power/energy features, phase metadata.
◦  Labels: traffic‑based idle labels, energy‑based labels.


For quick local testing (no Slurm):

1. Start TorchServe manually with a model.
2. Run:

   python client.py \
     --url http://127.0.0.1:8080 \
     --mode image \
     --model-name resnet-18 \
     --phases-json phases_short.json \
     --csv runs/local_requests.csv \
     --window-s 0.5

3. Optionally add --power-csv flags and compute_energy_windows.py for local energy testing.


11. How This Repository Connects to the Thesis


This code base provides all the data‑generation and measurement needed for your thesis:

Model‑side:
•  Standard CNNs (ResNet‑18/50, MobileNet‑V2) running on TorchServe, representing a GPU‑backed microservice.

Workload‑side:
•  Multi‑phase, realistic traces spanning cold starts, bursts, peak traffic, and long idle intervals.

Measurement‑side:
•  Detailed per‑request and per‑window performance metrics.
•  GPU power/energy from two independent paths (Zeus and NVML).

Labeling and dataset‑side:
•  Clear idle vs. busy labels from both request counts and energy observations.
•  Automatically merged, consistent CSVs ready for ML.

The actual ML models and training code (e.g., classifiers for predicting idle windows from observable metrics) can live in your thesis notebooks or another repository. This repository is the experimental backbone that makes those ML models possible and reproducible.