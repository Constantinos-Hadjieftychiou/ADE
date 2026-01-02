#!/bin/bash
# SLURM job script to run TorchServe + load test for a SINGLE phase
# Usage: sbatch job_single_phase.sbatch
# Or with overrides: sbatch --export=ALL,MODEL_NAME=resnet-50,PHASE_NAME=burst-10rps job_single_phase.sbatch

#SBATCH --job-name=torchserve_one_phase
# Set the SLURM job name (appears in squeue output)
#SBATCH --output=slurm-%x-%j.out
# Redirect stdout to slurm-<jobname>-<jobid>.out (e.g., slurm-torchserve_one_phase-12345.out)
#SBATCH --error=slurm-%x-%j.err
# Redirect stderr to slurm-<jobname>-<jobid>.err
#SBATCH --gres=gpu:1
# Request 1 GPU resource from SLURM
#SBATCH --mem=32G
# Request 32GB of CPU RAM
#SBATCH --time=00:50:00
# Set job time limit to 50 minutes (kills job if exceeded)
#SBATCH --partition=GPU
# Submit to the GPU partition (must exist on your cluster)

set -euo pipefail
# Strict bash: exit on error (-e), fail on undefined vars (-u), pipe fails propagate (-o pipefail)

# -----------------------
# Configuration variables (all can be overridden via --export in sbatch)
# -----------------------
MODEL_NAME="${MODEL_NAME:-resnet-18}"
# Model to load (default: resnet-18); override: --export=ALL,MODEL_NAME=resnet-50
PHASE_NAME="${PHASE_NAME:-cold-start-steady-1rps}"
# Name of the phase to run from phases.json (default: cold-start-steady-1rps)
PHASE_DURATION_SCALE="${PHASE_DURATION_SCALE:-0.05}"
# Scale factor for phase durations (0.05 = 5% of original; useful for quick tests)
IDLE_CALIBRATION_SECONDS="${IDLE_CALIBRATION_SECONDS:-0}"
# Seconds to idle at start to measure baseline GPU power (0 = skip)
WARMUP_REQUESTS="${WARMUP_REQUESTS:-10}"
# Number of requests to send before measurement (not recorded in results)
CONCURRENCY="${CONCURRENCY:-2}"
# Number of worker threads that send requests in parallel
WINDOW_S="${WINDOW_S:-0.5}"
# Aggregate per-request metrics into 0.5 second windows
POWER_SAMPLE_PERIOD="${POWER_SAMPLE_PERIOD:-0.1}"
# Sample GPU power every 0.1 seconds via NVML
SEED="${SEED:-42}"
# Random seed for reproducibility (controls phase order, sampling, etc.)
PHASES_JSON="${PHASES_JSON:-$SLURM_SUBMIT_DIR/phases.json}"
# Path to JSON file defining phases (default: phases.json in submit dir)

PROJECT_DIR="$SLURM_SUBMIT_DIR"
# Project root = directory where sbatch was run
ENV_DIR="$PROJECT_DIR/.venv_torchserve"
# Python venv directory for persistent installations
RUN_DIR="$PROJECT_DIR/runs/${MODEL_NAME}_${PHASE_NAME}_$(date +%Y%m%d_%H%M%S)"
# Output directory for this run (includes timestamp to avoid overwrites)
MODEL_STORE="$PROJECT_DIR/model_store"
# Directory where .mar model archives are stored
mkdir -p "$RUN_DIR" "$MODEL_STORE"
# Create output directories if they don't exist

# Export environment variables to control OpenMP, MKL, and Python behavior
export OMP_NUM_THREADS=1
# Single-threaded OpenMP (avoid contention with TorchServe worker pool)
export MKL_NUM_THREADS=1
# Single-threaded Intel MKL (same reason)
export HF_HOME="$PROJECT_DIR/.hf_cache"
# Cache Hugging Face models locally in project dir
export TRANSFORMERS_CACHE="$PROJECT_DIR/.hf_cache"
# Cache transformers library locally
export TORCH_HOME="$PROJECT_DIR/.torch_cache"
# Cache PyTorch models locally
export PYTHONUNBUFFERED=1
# Flush Python stdout/stderr immediately (no buffering; good for log files)

# Define a logging function (prepend timestamp to all log messages)
log() { echo "[$(date -Iseconds)] $*"; }

# Define cleanup trap: runs on EXIT, SIGTERM, SIGINT (normal/error/interrupt exit)
cleanup() {
  set +e
  # Disable strict error handling for cleanup (don't fail if commands error)
  log "Cleanup: stopping watchdog/torchserve"
  # Kill watchdog background process if it was started
  [[ -n "${WATCHDOG_PID:-}" ]] && kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
  # Stop TorchServe gracefully
  torchserve --stop >/dev/null 2>&1 || true
  # Wait for TorchServe to shut down
  sleep 2
  # Copy TorchServe logs to the run directory
  if [[ -d "$PROJECT_DIR/logs" ]]; then
    cp -r "$PROJECT_DIR/logs" "$RUN_DIR/torchserve_logs" 2>/dev/null || true
  fi
  log "RUN_DIR=$RUN_DIR"
}
trap cleanup EXIT SIGTERM SIGINT
# Ensure cleanup runs even if script fails or is interrupted

log "Starting TorchServe job on ${HOSTNAME}"
# Print start message with hostname
log "CONFIG: MODEL=$MODEL_NAME PHASE=$PHASE_NAME SCALE=$PHASE_DURATION_SCALE CONCURRENCY=$CONCURRENCY"
# Log configuration for reproducibility

# Unset PYTHONPATH to avoid interference from system Python installations
unset PYTHONPATH || true

# Load cluster modules (environment setup)
module purge
# Clear any previously loaded modules
module use /share/apps/eb/modules/all || true
# Add custom module path (cluster-specific; adjust if needed)
module load Python/3.11.5-GCCcore-13.2.0 || module load Python || { log "Failed to load Python"; exit 1; }
# Load Python 3.11 or fallback to any Python; fail if neither available
module load CUDA || true
# Load CUDA (optional; GPU may work without it if drivers already present)
module load Java || { log "Failed to load Java (required for TorchServe)"; exit 1; }
# Load Java (mandatory for TorchServe); exit if not available

log "GPU inventory:"
nvidia-smi || true
# Print GPU info (useful for debugging OOM/GPU issues); don't fail if unavailable

# Create or reuse persistent Python venv
if [[ ! -d "$ENV_DIR" ]]; then
  log "Creating venv at $ENV_DIR"
  # Create venv only once (reuse on subsequent runs)
  python -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
# Activate venv (all pip installs go here)

pip install --quiet --upgrade pip setuptools wheel
# Upgrade pip/setuptools/wheel (required for many packages)

# Install Python dependencies (only first time)
if [[ ! -f "$ENV_DIR/.deps_installed" ]]; then
  log "Installing dependencies..."
  # Install PyTorch with CUDA 12.1 wheels (specific version for reproducibility)
  PIP_ONLY_BINARY=":all:" pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 --no-cache-dir || { log "Failed to install torch"; exit 1; }
  # Install TorchServe and other dependencies
  pip install --quiet torchserve torch-model-archiver pillow requests pandas numpy nvidia-ml-py3
  # Mark dependencies as installed (skip on re-runs)
  touch "$ENV_DIR/.deps_installed"
else
  log "Dependencies already installed; skipping pip installs."
fi

log "Building .mar archives"
# Build model archives (TorchServe requires .mar format)
python "$PROJECT_DIR/build_resnet_mar.py" || { log "Failed to build .mar files"; exit 1; }

cd "$PROJECT_DIR"
# Change to project directory (client.py and torchserve.properties are here)

log "Pre-start cleanup: stopping existing TorchServe"
# Kill any leftover TorchServe process from previous run
torchserve --stop 2>/dev/null || true
sleep 2
# Wait for graceful shutdown

# Forcefully kill if ports are still in use
if ss -ltnp 2>/dev/null | grep -qE ':8080|:8081|:8082'; then
  log "Ports in use; killing processes"
  # TorchServe uses 8080 (inference), 8081 (management), 8082 (metrics)
  pkill -f torchserve 2>/dev/null || true
  pkill -f 'java.*ModelServer' 2>/dev/null || true
  sleep 3
fi

log "Starting TorchServe"
torchserve --start --ncs \
  # --ncs: No console logging (TorchServe logs go to ./logs/model_server.log)
  --disable-token-auth \
  # Disable authentication (safe for single-user research; don't use in production)
  --model-store "$MODEL_STORE" \
  # Directory containing .mar model archives
  --models "${MODEL_NAME}=${MODEL_NAME}.mar" \
  # Load model immediately on startup (name=archive.mar)
  --ts-config "$PROJECT_DIR/torchserve.properties"
  # Configuration file (ports, worker count, etc.)

sleep 8
# Wait for TorchServe to fully start

log "Waiting for ${MODEL_NAME} to become READY..."
TS_READY_START=$(date +%s)
# Record start time of readiness check
while true; do
  # Check two conditions: (1) /ping responds, (2) model status is READY
  if curl -fsS http://127.0.0.1:8080/ping >/dev/null 2>&1 && \
     curl -sS "http://127.0.0.1:8081/models/${MODEL_NAME}" | grep -q '"status"[[:space:]]*:[[:space:]]*"READY"'; then
    # Both checks passed; model is ready
    break
  fi
  
  NOW=$(date +%s)
  # Check if we've exceeded timeout
  if (( NOW - TS_READY_START > 120 )); then
    log "ERROR: ${MODEL_NAME} not READY after 120s"
    # Diagnostic output for debugging
    curl -sS "http://127.0.0.1:8081/models/${MODEL_NAME}" || true
    tail -n 50 "$PROJECT_DIR/logs/model_server.log" 2>/dev/null || true
    exit 1
  fi
  sleep 2
  # Wait before retrying
done
log "${MODEL_NAME} is READY."

# Create subdirectory for this run's output
SUBRUN_DIR="$RUN_DIR/run_1"
mkdir -p "$SUBRUN_DIR"

# Define output file paths
POWER_CSV="$SUBRUN_DIR/power_samples_${MODEL_NAME}.csv"
# Where NVML power samples will be written (timestamp, power_w, energy_j)
CLIENT_LOG="$SUBRUN_DIR/client_${MODEL_NAME}.log"
# Where client.py stdout/stderr will be redirected
REQUESTS_CSV="$SUBRUN_DIR/requests_${MODEL_NAME}.csv"
# Where per-request metrics will be written (ts_start, ts_end, latency, status, error)

log "===== RUNNING PHASE: $PHASE_NAME ====="
log "Logs: $CLIENT_LOG"

# Run the load generator (client.py)
python "$PROJECT_DIR/client.py" \
  --url "http://127.0.0.1:8080" \
  # Target TorchServe inference endpoint
  --mode image \
  # Use image mode (loads CIFAR-10)
  --model-name "${MODEL_NAME}" \
  # Which model to test
  --phases-json "$PHASES_JSON" \
  # JSON file defining phases (phases.json)
  --phase-name "$PHASE_NAME" \
  # Run only this one phase (e.g., "cold-start-steady-1rps")
  --phase-duration-scale "$PHASE_DURATION_SCALE" \
  # Scale duration (0.05 = 5% of original; for quick tests)
  --warmup-requests "$WARMUP_REQUESTS" \
  # Send this many requests before measurement starts (not recorded)
  --concurrency "$CONCURRENCY" \
  # Number of parallel worker threads
  --csv "$REQUESTS_CSV" \
  # Output CSV for per-request metrics
  --window-s "$WINDOW_S" \
  # Aggregate into 0.5s windows
  --power-csv "$POWER_CSV" \
  # Output CSV for NVML power samples
  --power-sample-period "$POWER_SAMPLE_PERIOD" \
  # Sample power every 0.1 seconds
  --idle-calibration-seconds "$IDLE_CALIBRATION_SECONDS" \
  # Idle time at start to measure baseline power
  --random-seed "$SEED" \
  # Random seed for reproducibility
  > "$CLIENT_LOG" 2>&1
  # Redirect stdout and stderr to log file

# Check if client succeeded
CLIENT_EXIT=$?
log "Client exited with code: $CLIENT_EXIT"

# Verify that output CSV was created
if [[ ! -f "$REQUESTS_CSV" ]]; then
  log "ERROR: Requests CSV not created!"
  # Print last 200 lines of log for debugging
  log "Last 200 lines of client.log:"
  tail -n 200 "$CLIENT_LOG" || true
  exit 1
fi

# Count how many requests were recorded
REQUEST_COUNT=$(tail -n +2 "$REQUESTS_CSV" 2>/dev/null | wc -l || echo 0)
# Skip header row (+2), count remaining lines
log "Requests recorded: $REQUEST_COUNT"

log "Job complete."
log "  Run directory: $RUN_DIR"
log "  Requests CSV:  $REQUESTS_CSV"
log "  Power CSV:     $POWER_CSV"