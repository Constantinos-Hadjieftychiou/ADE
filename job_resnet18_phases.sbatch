#!/bin/bash
# This script is submitted to SLURM with `sbatch job.sbatch`.
# It launches:
#   1. A TorchServe server hosting a ResNet-18 model on a GPU.
#   2. A long-running load test using client.py.
#   3. Zeus-based energy measurement in parallel with the load test.
#
# The goal is to:
#   - Generate realistic, multi-phase traffic against the model.
#   - Record per-request metrics (latency, errors) and per-second windows.
#   - Record per-second GPU energy usage via Zeus.
#   - Correlate energy with load to study idleness and energy efficiency.

#SBATCH --job-name=torchserve_resnet18       # Name of the job shown in SLURM queue.
#SBATCH --output=slurm-resnet18-%j.out      # Stdout log (includes %j = job ID).
#SBATCH --error=slurm-resnet18-%j.err       # Stderr log.
#SBATCH --gres=gpu:1                        # Request 1 GPU from the cluster.
#SBATCH --mem=32G                           # Request 32 GB of system RAM.
#SBATCH --time=02:00:00                     # Maximum wall-clock time: 2 hours.
#SBATCH --partition=GPU                     # Use the GPU partition (cluster-specific).

# `set -u`   : treat unset variables as an error.
# `set -o pipefail` : fail the script if any command in a pipeline fails.
set -u
set -o pipefail

echo "📦 Starting TorchServe ResNet-18 job on $HOSTNAME"
echo "SLURM_JOBID=${SLURM_JOB_ID:-unknown} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---------------------- Paths & caches (persistent) -------------------------
# SLURM_SUBMIT_DIR is the directory from which `sbatch` was called.
PROJECT_DIR="$SLURM_SUBMIT_DIR"

# We keep a persistent virtual environment inside the project directory.
ENV_DIR="$PROJECT_DIR/.venv_torchserve"

# Each run gets a separate directory with timestamp for logs and metrics.
RUN_DIR="$PROJECT_DIR/runs/resnet18_run_$(date +%Y%m%d_%H%M%S)"

# TorchServe model store directory: where the .mar model archives live.
MODEL_STORE="$PROJECT_DIR/model_store"

# Create directories if they do not already exist.
mkdir -p "$RUN_DIR" "$MODEL_STORE"

# Limit number of CPU threads used by libraries like OpenMP/MKL to avoid
# oversubscribing CPU cores on a multi-user cluster.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Cache directories for HuggingFace, transformers, and torch.
# This helps avoid re-downloading large model files for each job.
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.hf_cache"
export TORCH_HOME="$PROJECT_DIR/.torch_cache"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "RUN_DIR=$RUN_DIR"

# Clear PYTHONPATH to avoid interference from user/local Python modules.
unset PYTHONPATH || true

# ----------------------------- Modules --------------------------------------
# Reset loaded modules to a clean state.
module purge

# Load the site-wide module collection (path is cluster-specific).
module use /share/apps/eb/modules/all || true

# Load a specific Python module (3.11.5 here), or fall back to a generic Python.
module load Python/3.11.5-GCCcore-13.2.0 || module load Python || {
  echo "❌ Failed to load Python module"
  exit 1
}

# Load CUDA module for GPU support (if available on the cluster).
module load CUDA || true

# TorchServe requires Java (for the backend process).
module load Java || {
  echo "❌ Failed to load Java module (required for TorchServe)"
  exit 1
}

echo "🔧 GPU inventory:"
# Print GPU info for debugging (which GPU we got, memory, etc.).
nvidia-smi || true

# ------------------------ Persistent virtualenv ------------------------------
# We use a Python virtual environment inside the project directory.
# This allows us to install exact versions of libraries consistently.
if [[ ! -d "$ENV_DIR" ]]; then
  echo "🐍 Creating persistent venv at $ENV_DIR"
  python -m venv "$ENV_DIR"
fi
# Activate the virtual environment.
source "$ENV_DIR/bin/activate"

# Disable build isolation to speed up installs and avoid some build issues.
export PIP_NO_BUILD_ISOLATION=1

# Upgrade pip, setuptools, wheel to fairly recent versions.
pip install --quiet --upgrade pip setuptools wheel

# Install PyTorch and torchvision with pre-built CUDA 12.1 wheels from
# the official PyTorch index. `PIP_ONLY_BINARY=":all:"` ensures we only
# install binary wheels and do not try to build from source.
PIP_ONLY_BINARY=":all:" pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 --no-cache-dir || {
  echo "❌ Failed to install torch/torchvision"
  exit 1
}

# TorchServe + utilities + analysis dependencies:
# - torchserve, torch-model-archiver: serving framework and model archiver.
# - pillow: image handling (JPEG, PNG, etc.).
# - requests: HTTP client used by client.py.
# - transformers: used by some models or build scripts (if needed).
# - pandas, numpy: used for metrics aggregation and energy correlation.
pip install --quiet \
  torchserve torch-model-archiver pillow requests transformers pandas numpy

# captum: interpretability library (no dependencies here; we installed torch already).
pip install --quiet --no-deps captum

# zeus: GPU power/energy measurement framework used in measure_with_zeus.py.
pip install --quiet zeus

# ------------------ Build MARs (multi-model script) -------------------------
# This step ensures that the model archive files (.mar) exist in MODEL_STORE.
# build_resnet_mar.py is a custom script that:
#   - Exports a PyTorch model.
#   - Packages it into TorchServe's MAR format.
echo "📦 Ensuring *.mar exists (resnet-18, resnet-50, mobilenet-v2)…"
python "$PROJECT_DIR/build_resnet_mar.py" || {
  echo "❌ Failed to build .mar files"
  exit 1
}

# ------------------------------ Start TorchServe -----------------------------
cd "$PROJECT_DIR"

echo "🚀 Starting TorchServe (ResNet-18)…"
# Start TorchServe in the background:
#   --ncs              : no configuration snapshot.
#   --disable-token-auth: disable authentication for simplicity (cluster is trusted).
#   --model-store      : directory with .mar files.
#   --models           : which models to load on startup.
#   --ts-config        : TorchServe configuration file.
torchserve --start --ncs \
  --disable-token-auth \
  --model-store "$MODEL_STORE" \
  --models resnet-18=resnet-18.mar \
  --ts-config "$PROJECT_DIR/torchserve.properties"

# Give TorchServe some time to start and load the model.
sleep 20

echo "⏳ Checking TorchServe /ping…"
# /ping is a simple health endpoint. It may return 404 until the model is fully loaded.
curl -s "http://$HOSTNAME:8080/ping" || echo "ℹ️ /ping may 404 until models are loaded"

# ------------------------------ Run client -----------------------------------
echo "💥 Multi-phase image workload against resnet-18 (Zeus monitored)…"

# We wrap the client invocation inside measure_with_zeus.py.
# That script:
#   - Uses Zeus to sample GPU energy/power at a fixed interval.
#   - Runs the client.py script in a sub-process.
#   - Writes per-second energy data to zeus_windows_resnet18.csv.
#   - Writes a summary JSON with total energy, total time, etc.

# We set --phases-total-seconds to ~6600s (~1h50m), slightly less than the
# 2h SLURM time limit, leaving some buffer for:
#   - Environment setup.
#   - TorchServe startup and shutdown.
#   - Data archiving.
python "$PROJECT_DIR/measure_with_zeus.py" \
  --window-name "resnet18_phases" \
  --log-csv "$RUN_DIR/zeus_windows_resnet18.csv" \
  --json-out "$RUN_DIR/zeus_summary_resnet18.json" \
  -- \
  python "$PROJECT_DIR/client.py" \
    --url "http://$HOSTNAME:8080" \
    --mode image \
    --model-name resnet-18 \
    --pattern burst \
    --warmup-requests 100 \
    --concurrency 16 \
    --phases-json "$PROJECT_DIR/phases.json" \
    --phases-total-seconds 6600 \
    --csv "$RUN_DIR/requests_resnet18.csv" \
    --energy-csv "$RUN_DIR/zeus_windows_resnet18.csv" \
  > "$RUN_DIR/client_resnet18.log" 2>&1 || true
# Note:
#   - Stdout+stderr from client.py are redirected to client_resnet18.log.
#   - `|| true` ensures that even if client.py fails, the script continues
#     to clean up TorchServe etc. (we still want logs).

# ------------------------- TorchServe metrics snapshot -----------------------
echo "📊 Fetching TorchServe metrics…"
# TorchServe exposes Prometheus metrics on port 8082.
curl -s "http://$HOSTNAME:8082/metrics" > "$RUN_DIR/torchserve_metrics_resnet18.prom" \
  || echo "⚠️ Failed to fetch TorchServe /metrics"

# ------------------------------- Teardown ------------------------------------
echo "🧹 Stopping TorchServe…"
# Stop the TorchServe process. If this fails (e.g. already stopped), ignore error.
torchserve --stop || true
sleep 5

# Copy TorchServe logs (if any) into the run directory for debugging.
if [[ -d "$PROJECT_DIR/logs" ]]; then
  cp -r "$PROJECT_DIR/logs" "$RUN_DIR/torchserve_logs"
fi

# Archive all run outputs and logs into a single tar.gz file for easier download.
tar -czf "$RUN_DIR/logs_resnet18.tar.gz" -C "$RUN_DIR" . 2>/dev/null || true

echo "✅ ResNet-18 job complete. Logs archived to $RUN_DIR/logs_resnet18.tar.gz"
