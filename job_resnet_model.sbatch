#!/bin/bash
# SLURM job script to run TorchServe + load test for ALL phases (with repeats)
# Usage: sbatch job_resnet_model.sbatch
# Or: sbatch --export=ALL,REPEATS=3,CONCURRENCY=16 job_resnet_model.sbatch

#SBATCH --job-name=torchserve_model
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# 4 hour time limit (longer than single-phase since we run all phases + repeats)
#SBATCH --partition=GPU

set -euo pipefail

# -----------------------
# Configuration variables
# -----------------------
MODEL_NAME="${MODEL_NAME:-resnet-18}"
# Model name (default: resnet-18)
REPEATS="${REPEATS:-1}"
# Number of independent runs to perform (each run cycles through all phases)
IDLE_CALIBRATION_SECONDS="${IDLE_CALIBRATION_SECONDS:-90}"
# Seconds to idle at start to measure baseline power (longer than single-phase for better calibration)
WINDOW_S="${WINDOW_S:-0.5}"
# Aggregate metrics into 0.5s windows
POWER_SAMPLE_PERIOD="${POWER_SAMPLE_PERIOD:-0.1}"
# Sample power every 0.1 seconds
CONCURRENCY="${CONCURRENCY:-16}"
# Default to 16 workers (higher than single-phase script)
WARMUP_REQUESTS="${WARMUP_REQUESTS:-100}"
# Send 100 warmup requests (more than single-phase for better model priming)
SEED="${SEED:-42}"
# Random seed
PHASE_DURATION_SCALE="${PHASE_DURATION_SCALE:-1.0}"
# Scale all phase durations (1.0 = no scaling; 0.5 = half duration)

# Auto-detect phases JSON (model-specific or generic)
DEFAULT_PHASES_JSON="$SLURM_SUBMIT_DIR/phases_${MODEL_NAME}.json"
# Look for model-specific phases file first (e.g., phases_resnet-18.json)
if [[ -f "$DEFAULT_PHASES_JSON" ]]; then
  PHASES_JSON_DEFAULT="$DEFAULT_PHASES_JSON"
else
  PHASES_JSON_DEFAULT="$SLURM_SUBMIT_DIR/phases.json"
  # Fall back to generic phases.json
fi
PHASES_JSON="${PHASES_JSON:-$PHASES_JSON_DEFAULT}"
# Allow override via --export=ALL,PHASES_JSON=/path/to/custom/phases.json

PROJECT_DIR="$SLURM_SUBMIT_DIR"
ENV_DIR="$PROJECT_DIR/.venv_torchserve"
RUN_DIR="$PROJECT_DIR/runs/${MODEL_NAME}_run_$(date +%Y%m%d_%H%M%S)"
# Output directory (note: no phase name, since we run all phases)
MODEL_STORE="$PROJECT_DIR/model_store"
mkdir -p "$RUN_DIR" "$MODEL_STORE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.hf_cache"
export TORCH_HOME="$PROJECT_DIR/.torch_cache"
export PYTHONUNBUFFERED=1

log() { echo "[$(date -Iseconds)] $*"; }

cleanup() {
  set +e
  log "Cleanup: stopping TorchServe"
  torchserve --stop >/dev/null 2>&1 || true
  sleep 2
  # Copy TorchServe logs to output directory
  if [[ -d "$PROJECT_DIR/logs" ]]; then
    cp -r "$PROJECT_DIR/logs" "$RUN_DIR/torchserve_logs" 2>/dev/null || true
  fi
  log "RUN_DIR=$RUN_DIR"
}
trap cleanup EXIT SIGTERM SIGINT

log "Starting TorchServe job on ${HOSTNAME}"
log "CONFIG: MODEL=$MODEL_NAME REPEATS=$REPEATS CONCURRENCY=$CONCURRENCY PHASES=$PHASES_JSON"

unset PYTHONPATH || true

module purge
module use /share/apps/eb/modules/all || true
module load Python/3.11.5-GCCcore-13.2.0 || module load Python || { log "Failed to load Python"; exit 1; }
module load CUDA || true
module load Java || { log "Failed to load Java (required for TorchServe)"; exit 1; }

log "GPU inventory:"
nvidia-smi || true

if [[ ! -d "$ENV_DIR" ]]; then
  log "Creating venv at $ENV_DIR"
  python -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

pip install --quiet --upgrade pip setuptools wheel

if [[ ! -f "$ENV_DIR/.deps_installed" ]]; then
  log "Installing dependencies..."
  PIP_ONLY_BINARY=":all:" pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 --no-cache-dir || { log "Failed to install torch"; exit 1; }
  pip install --quiet torchserve torch-model-archiver pillow requests pandas numpy nvidia-ml-py3
  touch "$ENV_DIR/.deps_installed"
else
  log "Dependencies already installed; skipping pip installs."
fi

log "Building .mar archives"
python "$PROJECT_DIR/build_resnet_mar.py" || { log "Failed to build .mar files"; exit 1; }

cd "$PROJECT_DIR"

log "Pre-start cleanup: stopping existing TorchServe"
torchserve --stop 2>/dev/null || true
sleep 2

if ss -ltnp 2>/dev/null | grep -qE ':8080|:8081|:8082'; then
  log "Ports in use; killing processes"
  pkill -f torchserve 2>/dev/null || true
  pkill -f 'java.*ModelServer' 2>/dev/null || true
  sleep 3
fi

log "Starting TorchServe"
torchserve --start --ncs \
  --disable-token-auth \
  --model-store "$MODEL_STORE" \
  --models "${MODEL_NAME}=${MODEL_NAME}.mar" \
  --ts-config "$PROJECT_DIR/torchserve.properties"

sleep 8

log "Waiting for ${MODEL_NAME} to become READY..."
TS_READY_START=$(date +%s)
while true; do
  if curl -fsS http://127.0.0.1:8080/ping >/dev/null 2>&1 && \
     curl -sS "http://127.0.0.1:8081/models/${MODEL_NAME}" | grep -q '"status"[[:space:]]*:[[:space:]]*"READY"'; then
    break
  fi
  
  NOW=$(date +%s)
  if (( NOW - TS_READY_START > 120 )); then
    log "ERROR: ${MODEL_NAME} not READY after 120s"
    curl -sS "http://127.0.0.1:8081/models/${MODEL_NAME}" || true
    tail -n 50 "$PROJECT_DIR/logs/model_server.log" 2>/dev/null || true
    exit 1
  fi
  sleep 2
done
log "${MODEL_NAME} is READY."

# Main loop: run the load test REPEATS times
RUN_IDX=1
# Loop counter for multiple independent runs
while [[ "$RUN_IDX" -le "$REPEATS" ]]; do
  # Create subdirectory for this specific run (run_1, run_2, etc.)
  SUBRUN_DIR="$RUN_DIR/run_${RUN_IDX}"
  mkdir -p "$SUBRUN_DIR"

  # Generate unique seed for each repeat (deterministic: SEED + index)
  SEED_RUN=$((SEED + RUN_IDX - 1))
  # Ensures reproducible but different randomization across repeats
  
  # Define output paths for this run
  POWER_CSV="$SUBRUN_DIR/power_samples_${MODEL_NAME}.csv"
  CLIENT_LOG="$SUBRUN_DIR/client_${MODEL_NAME}.log"
  REQUESTS_CSV="$SUBRUN_DIR/requests_${MODEL_NAME}.csv"

  log "===== RUN $RUN_IDX / $REPEATS ====="
  log "Logs: $CLIENT_LOG"

  # Execute client.py for this run (note: NO --phase-name, so runs all phases)
  python "$PROJECT_DIR/client.py" \
    --url "http://127.0.0.1:8080" \
    --mode image \
    --model-name "${MODEL_NAME}" \
    --phases-json "$PHASES_JSON" \
    # All phases from JSON (not filtered by name)
    --phase-duration-scale "$PHASE_DURATION_SCALE" \
    --warmup-requests "$WARMUP_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --csv "$REQUESTS_CSV" \
    --window-s "$WINDOW_S" \
    --power-csv "$POWER_CSV" \
    --power-sample-period "$POWER_SAMPLE_PERIOD" \
    --idle-calibration-seconds "$IDLE_CALIBRATION_SECONDS" \
    --random-seed "$SEED_RUN" \
    # Unique seed per repeat (SEED + RUN_IDX - 1)
    > "$CLIENT_LOG" 2>&1 || true
    # || true: don't fail the script if client.py errors; we check below

  # Validate that output was created
  if [[ ! -f "$REQUESTS_CSV" ]]; then
    log "ERROR: Requests CSV not created in run $RUN_IDX"
    log "Last 100 lines of client.log:"
    tail -n 100 "$CLIENT_LOG" || true
  else
    # Count successful requests recorded
    REQUEST_COUNT=$(tail -n +2 "$REQUESTS_CSV" 2>/dev/null | wc -l || echo 0)
    log "Run $RUN_IDX completed: $REQUEST_COUNT requests recorded"
  fi

  # Increment run counter and loop if REPEATS > 1
  RUN_IDX=$((RUN_IDX + 1))
done

log "All runs completed. Running post-job analysis..."
# Merge all run_N/requests_*_windows_*.csv files into one combined dataset
python "$PROJECT_DIR/analyze_and_merge_runs.py" \
  --runs-root "$RUN_DIR" \
  # Top-level directory containing run_1, run_2, etc.
  --model-name "$MODEL_NAME" \
  # Used to find matching CSVs and name output
  > "$RUN_DIR/analysis_${MODEL_NAME}.txt" 2>&1 || log "Analysis script failed"

# Construct path to merged output CSV
SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
# Replace any slashes in model name with underscores (for safe filenames)
MERGED_CSV="$RUN_DIR/merged_windows_${SANITIZED_MODEL_NAME}.csv"

log "Job complete."
log "  Run directory: $RUN_DIR"
log "  Merged windows: $MERGED_CSV"
# This CSV contains all window-level metrics from all runs, stacked together
log "  Analysis log: $RUN_DIR/analysis_${MODEL_NAME}.txt"
# Contains statistics, class balance, correlations, etc.