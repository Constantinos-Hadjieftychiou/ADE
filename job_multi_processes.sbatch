#!/bin/bash
# ============================================================================
# SLURM job: TorchServe + Poisson load + windowed GPU energy measurement
# Correct version with:
# - sent.log
# - completions.log
# - merged request_log.csv
# ============================================================================

#SBATCH --job-name=torchserve_poisson
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-resnet-18}"
REPEATS="${REPEATS:-1}"
CONCURRENCY="${CONCURRENCY:-16}"
WINDOW_S="${WINDOW_S:-0.05}"
UTIL_SAMPLE_MS="${UTIL_SAMPLE_MS:-2}"
ENERGY_IDLE_EPS_W="${ENERGY_IDLE_EPS_W:-5.0}"
PHASES_JSON="${PHASES_JSON:-$SLURM_SUBMIT_DIR/phases_poisson.json}"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
ENV_DIR="$PROJECT_DIR/.venv_torchserve"
MODEL_STORE="$PROJECT_DIR/model_store"
RUN_DIR="$PROJECT_DIR/runs/${MODEL_NAME}_poisson_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$RUN_DIR" "$MODEL_STORE"

# ---------------- CPU CONTROL ----------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export PYTHONUNBUFFERED=1

log() { echo "[$(date -Iseconds)] $*"; }

cleanup() {
  set +e
  log "Stopping TorchServe"
  torchserve --stop >/dev/null 2>&1 || true
  sleep 2
  log "Run directory: $RUN_DIR"
}
trap cleanup EXIT SIGINT SIGTERM

# ---------------- MODULES ----------------
module purge
module use /share/apps/eb/modules/all || true
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1
module load Java
module load matplotlib

log "GPU info:"
nvidia-smi

# ---------------- PYTHON ENV ----------------
if [[ ! -d "$ENV_DIR" ]]; then
  python -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"

pip install --quiet --upgrade pip setuptools wheel

if [[ ! -f "$ENV_DIR/.deps_installed" ]]; then

  PIP_ONLY_BINARY=":all:" pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 --no-cache-dir

  pip install \
    torchserve torch-model-archiver \
    pillow requests pandas numpy matplotlib \
    nvidia-ml-py3 zeus-ml-energy

  touch "$ENV_DIR/.deps_installed"
fi

# ---------------- MODEL ----------------
MAR_PATH="$MODEL_STORE/${MODEL_NAME}.mar"

if [[ ! -f "$MAR_PATH" ]]; then
  python "$PROJECT_DIR/build_resnet_mar.py"
fi

cd "$PROJECT_DIR"

torchserve --stop >/dev/null 2>&1 || true
sleep 2

# ---------------- START TORCHSERVE ----------------
log "Starting TorchServe pinned to CPUs 2-3"

srun --ntasks=1 \
     --cpus-per-task=2 \
     --exclusive \
     --cpu-bind=map_cpu:2,3 \
     torchserve --start --ncs \
        --disable-token-auth \
        --model-store "$MODEL_STORE" \
        --models "${MODEL_NAME}=${MODEL_NAME}.mar" \
        --ts-config "$PROJECT_DIR/torchserve.properties" &

sleep 8

# ---------------- WAIT FOR READY ----------------
log "Waiting for TorchServe READY..."

READY_START=$(date +%s)

while true; do
  if \
    curl -s http://127.0.0.1:8080/ping >/dev/null 2>&1 && \
    curl -s http://127.0.0.1:8081/models/${MODEL_NAME} 2>/dev/null | grep -q '"READY"'
  then
    break
  fi

  if (( $(date +%s) - READY_START > 180 )); then
    log "TorchServe failed to become READY"
    exit 1
  fi

  sleep 2
done

log "TorchServe READY"

# ---------------- WARMUP ----------------
log "Running warm-up inferences..."

for i in {1..40}; do
  curl -s -X POST http://127.0.0.1:8080/predictions/${MODEL_NAME} \
    -T "$PROJECT_DIR/sample.jpg" >/dev/null 2>&1 || true
done

sleep 5
log "Warm-up completed"

# ============================================================================
# EXPERIMENT
# ============================================================================

for (( RUN=1; RUN<=REPEATS; RUN++ )); do

  SUBRUN_DIR="$RUN_DIR/run_${RUN}"
  mkdir -p "$SUBRUN_DIR"

  SENT_LOG="$SUBRUN_DIR/sent.log"
  COMPLETION_LOG="$SUBRUN_DIR/completions.log"
  REQUEST_LOG="$SUBRUN_DIR/request_log.csv"

  WINDOWS_CSV="$SUBRUN_DIR/windows_${MODEL_NAME}.csv"
  ANALYSIS_PREFIX="$SUBRUN_DIR/analysis"

  : > "$SENT_LOG"
  : > "$COMPLETION_LOG"

  log "===== RUN $RUN ====="

  # ---------------- LOAD GENERATOR ----------------
  taskset -c 0 python "$PROJECT_DIR/workload_generator.py" \
      --url http://127.0.0.1:8080 \
      --model-name "$MODEL_NAME" \
      --phases-json "$PHASES_JSON" \
      --concurrency "$CONCURRENCY" \
      --completion-log "$COMPLETION_LOG" \
      --sent-log "$SENT_LOG" &

  LOAD_PID=$!

  # ---------------- ENERGY MONITOR ----------------
  taskset -c 1 python "$PROJECT_DIR/energy_monitor.py" \
      --phases-json "$PHASES_JSON" \
      --completion-log "$COMPLETION_LOG" \
      --sent-log "$SENT_LOG" \
      --window-s "$WINDOW_S" \
      --util-sample-ms "$UTIL_SAMPLE_MS" \
      --csv "$WINDOWS_CSV"

  wait "$LOAD_PID"

  # ---------------- MERGE REQUEST LOG ----------------
  python - <<EOF
import csv

sent=open("$SENT_LOG").read().strip().splitlines()
comp=open("$COMPLETION_LOG").read().strip().splitlines()

n=min(len(sent),len(comp))

with open("$REQUEST_LOG","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["request_id","send_ts","completion_ts"])

    for i in range(n):
        w.writerow([i,sent[i],comp[i]])
EOF

  # ---------------- ANALYSIS ----------------
  python "$PROJECT_DIR/analyze_idle.py" \
      --csv "$WINDOWS_CSV" \
      --epsilon-w "$ENERGY_IDLE_EPS_W" \
      --out-prefix "$ANALYSIS_PREFIX"

  log "Run $RUN completed"

done

log "All runs finished"
log "Results stored in: $RUN_DIR"