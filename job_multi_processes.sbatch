#!/bin/bash
# ============================================================================
# SLURM job: TorchServe + Poisson load + windowed GPU energy measurement
#
# Architecture:
#   - TorchServe: inference service (CPU + GPU)
#   - workload_generator.py: CPU 0 (request arrivals + completions)
#   - energy_monitor.py:    CPU 1 (energy + NVML sampling)
#
# Measurement:
#   - Fixed-size energy windows (sub-100ms)
#   - Zeus for GPU energy
#   - NVML for binary GPU activity
#
# Scientifically valid for NVIDIA V100 GPUs
# ============================================================================

#SBATCH --job-name=torchserve_poisson
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=GPU

set -euo pipefail

# ----------------------------------------------------------------------------
# Experiment configuration (override via --export)
# ----------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-resnet-18}"
REPEATS="${REPEATS:-1}"

CONCURRENCY="${CONCURRENCY:-16}"
WINDOW_S="${WINDOW_S:-0.05}"          # 50 ms windows
UTIL_SAMPLE_MS="${UTIL_SAMPLE_MS:-2}" # 2 ms NVML sampling
SEED_BASE="${SEED_BASE:-42}"

PHASES_JSON="${PHASES_JSON:-$SLURM_SUBMIT_DIR/phases_poisson.json}"

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_DIR="$SLURM_SUBMIT_DIR"
ENV_DIR="$PROJECT_DIR/.venv_torchserve"
MODEL_STORE="$PROJECT_DIR/model_store"
RUN_DIR="$PROJECT_DIR/runs/${MODEL_NAME}_poisson_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RUN_DIR" "$MODEL_STORE"

# ----------------------------------------------------------------------------
# Environment hygiene (reproducibility)
# ----------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_HOME="$PROJECT_DIR/.torch_cache"
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.hf_cache"

log() { echo "[$(date -Iseconds)] $*"; }

cleanup() {
  set +e
  log "Stopping TorchServe"
  torchserve --stop >/dev/null 2>&1 || true
  sleep 2
  if [[ -d "$PROJECT_DIR/logs" ]]; then
    cp -r "$PROJECT_DIR/logs" "$RUN_DIR/torchserve_logs" 2>/dev/null || true
  fi
  log "Run directory: $RUN_DIR"
}
trap cleanup EXIT SIGINT SIGTERM

# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------
module purge
module use /share/apps/eb/modules/all || true
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1
module load Java

log "GPU info:"
nvidia-smi

# ----------------------------------------------------------------------------
# Python environment
# ----------------------------------------------------------------------------
if [[ ! -d "$ENV_DIR" ]]; then
  log "Creating virtualenv: $ENV_DIR"
  python -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

pip install --quiet --upgrade pip setuptools wheel

if [[ ! -f "$ENV_DIR/.deps_installed" ]]; then
  log "Installing Python dependencies"

  PIP_ONLY_BINARY=":all:" pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 --no-cache-dir

  pip install \
    torchserve torch-model-archiver \
    pillow requests pandas numpy \
    nvidia-ml-py3 zeus-ml-energy

  touch "$ENV_DIR/.deps_installed"
else
  log "Dependencies already installed"
fi

# ----------------------------------------------------------------------------
# Build model archive if needed
# ----------------------------------------------------------------------------
MAR_PATH="$MODEL_STORE/${MODEL_NAME}.mar"
if [[ ! -f "$MAR_PATH" ]]; then
  log "Building TorchServe MAR"
  python "$PROJECT_DIR/build_resnet_mar.py"
else
  log "Using existing MAR: $MAR_PATH"
fi

# ----------------------------------------------------------------------------
# Start TorchServe
# ----------------------------------------------------------------------------
cd "$PROJECT_DIR"

torchserve --stop >/dev/null 2>&1 || true
sleep 2

log "Starting TorchServe"
torchserve --start --ncs \
  --disable-token-auth \
  --model-store "$MODEL_STORE" \
  --models "${MODEL_NAME}=${MODEL_NAME}.mar" \
  --ts-config "$PROJECT_DIR/torchserve.properties"

# ----------------------------------------------------------------------------
# Wait for TorchServe readiness (HARD guarantee)
# ----------------------------------------------------------------------------
log "Waiting for TorchServe to become READY..."
READY_START=$(date +%s)
while true; do
  if curl -fsS http://127.0.0.1:8080/ping >/dev/null && \
     curl -sS http://127.0.0.1:8081/models/${MODEL_NAME} | grep -q '"READY"'; then
    break
  fi
  (( $(date +%s) - READY_START > 120 )) && { log "TorchServe not READY"; exit 1; }
  sleep 1
done
log "TorchServe is READY"

# ----------------------------------------------------------------------------
# Run experiments
# ----------------------------------------------------------------------------
for (( RUN=1; RUN<=REPEATS; RUN++ )); do
  SUBRUN_DIR="$RUN_DIR/run_${RUN}"
  mkdir -p "$SUBRUN_DIR"

  SEED=$((SEED_BASE + RUN - 1))

  COMPLETION_LOG="$SUBRUN_DIR/completions.log"
  WINDOWS_CSV="$SUBRUN_DIR/windows_${MODEL_NAME}.csv"

  log "===== RUN $RUN / $REPEATS (seed=$SEED) ====="
  log "WINDOW_S=$WINDOW_S | UTIL_SAMPLE_MS=$UTIL_SAMPLE_MS | CONCURRENCY=$CONCURRENCY"

  # --------------------------------------------------------------------------
  # Start workload generator (CPU 0)
  # --------------------------------------------------------------------------
  taskset -c 0 python "$PROJECT_DIR/workload_generator.py" \
    --url http://127.0.0.1:8080 \
    --model-name "$MODEL_NAME" \
    --phases-json "$PHASES_JSON" \
    --concurrency "$CONCURRENCY" \
    --log-file "$COMPLETION_LOG" &

  LOAD_PID=$!

  # --------------------------------------------------------------------------
  # Start energy monitor (CPU 1)
  # --------------------------------------------------------------------------
  taskset -c 1 python "$PROJECT_DIR/energy_monitor.py" \
    --phases-json "$PHASES_JSON" \
    --completion-log "$COMPLETION_LOG" \
    --window-s "$WINDOW_S" \
    --util-sample-ms "$UTIL_SAMPLE_MS" \
    --csv "$WINDOWS_CSV"

  # Ensure workload generator exits cleanly
  wait "$LOAD_PID"

  log "Run $RUN completed"
done

log "All runs finished"
log "Results in: $RUN_DIR"
