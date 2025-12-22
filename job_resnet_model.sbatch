#!/bin/bash

#SBATCH --job-name=torchserve_model
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=GPU

set -u
set -o pipefail

echo "🚀 Starting TorchServe job on $HOSTNAME"
echo "SLURM_JOBID=${SLURM_JOB_ID:-unknown} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
ENV_DIR="$PROJECT_DIR/.venv_torchserve"

# ------------------------------------------------------------------------------
# Configurable parameters (override via sbatch --export)
# ------------------------------------------------------------------------------

MODEL_NAME="${MODEL_NAME:-resnet-18}"   # e.g. resnet-18, resnet-50, mobilenet-v2
REPEATS="${REPEATS:-1}"

# Idle calibration: seconds of NVML sampling with NO requests (per run).
IDLE_CALIBRATION_SECONDS="${IDLE_CALIBRATION_SECONDS:-90}"

WINDOW_S="${WINDOW_S:-0.5}"
POWER_SAMPLE_PERIOD="${POWER_SAMPLE_PERIOD:-0.1}"
CONCURRENCY="${CONCURRENCY:-16}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-100}"
PATTERN="${PATTERN:-steady}"  # burst/steady/poisson
SEED="${SEED:-42}"

# Optional global scale factor for all phase durations in phases JSON.
PHASE_DURATION_SCALE="${PHASE_DURATION_SCALE:-1.0}"

DEFAULT_PHASES_JSON="$PROJECT_DIR/phases_${MODEL_NAME}.json"
if [[ -f "$DEFAULT_PHASES_JSON" ]]; then
  PHASES_JSON_DEFAULT="$DEFAULT_PHASES_JSON"
else
  PHASES_JSON_DEFAULT="$PROJECT_DIR/phases.json"
fi
PHASES_JSON="${PHASES_JSON:-$PHASES_JSON_DEFAULT}"

RUN_DIR="$PROJECT_DIR/runs/${MODEL_NAME}_run_$(date +%Y%m%d_%H%M%S)"
MODEL_STORE="$PROJECT_DIR/model_store"

mkdir -p "$RUN_DIR" "$MODEL_STORE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.hf_cache"
export TORCH_HOME="$PROJECT_DIR/.torch_cache"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "RUN_DIR=$RUN_DIR"
echo "MODEL_STORE=$MODEL_STORE"
echo "MODEL_NAME=$MODEL_NAME"
echo "REPEATS=$REPEATS"
echo "WINDOW_S=$WINDOW_S"
echo "POWER_SAMPLE_PERIOD=$POWER_SAMPLE_PERIOD"
echo "CONCURRENCY=$CONCURRENCY"
echo "WARMUP_REQUESTS=$WARMUP_REQUESTS"
echo "PATTERN=$PATTERN"
echo "PHASES_JSON=$PHASES_JSON"
echo "SEED(base)=$SEED"
echo "IDLE_CALIBRATION_SECONDS=$IDLE_CALIBRATION_SECONDS"
echo "PHASE_DURATION_SCALE=$PHASE_DURATION_SCALE"

unset PYTHONPATH || true

module purge
module use /share/apps/eb/modules/all || true

module load Python/3.11.5-GCCcore-13.2.0 || module load Python || {
  echo "Failed to load Python module"
  exit 1
}

module load CUDA || true

module load Java || {
  echo "Failed to load Java module (required for TorchServe)"
  exit 1
}

echo "GPU inventory:"
nvidia-smi || true

if [[ ! -d "$ENV_DIR" ]]; then
  echo "Creating persistent venv at $ENV_DIR"
  python -m venv "$ENV_DIR"
fi

# shellcheck source=/dev/null
source "$ENV_DIR/bin/activate"

export PIP_NO_BUILD_ISOLATION=1

pip install --quiet --upgrade pip setuptools wheel

PIP_ONLY_BINARY=":all:" pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 --no-cache-dir || {
  echo "Failed to install torch/torchvision"
  exit 1
}

pip install --quiet \
  torchserve torch-model-archiver pillow requests transformers pandas numpy nvidia-ml-py3
pip install --quiet --no-deps captum
pip install --quiet zeus

echo "Ensuring *.mar exists (resnet-18, resnet-50, mobilenet-v2)"
python "$PROJECT_DIR/build_resnet_mar.py" || {
  echo "Failed to build .mar files"
  exit 1
}

cd "$PROJECT_DIR"

# ------------------------------------------------------------------------------
# Top-level job config
# ------------------------------------------------------------------------------
cat > "$RUN_DIR/config_job_${MODEL_NAME}.json" <<EOF
{
  "job_name": "torchserve_${MODEL_NAME}",
  "slurm_job_id": "${SLURM_JOB_ID:-unknown}",
  "host": "${HOSTNAME}",
  "model_name": "${MODEL_NAME}",
  "window_s": ${WINDOW_S},
  "power_sample_period": ${POWER_SAMPLE_PERIOD},
  "idle_calibration_seconds": ${IDLE_CALIBRATION_SECONDS},
  "concurrency": ${CONCURRENCY},
  "warmup_requests": ${WARMUP_REQUESTS},
  "pattern": "${PATTERN}",
  "phases_json": "${PHASES_JSON}",
  "seed_base": ${SEED},
  "repeats": ${REPEATS},
  "phase_duration_scale": ${PHASE_DURATION_SCALE}
}
EOF

echo "Starting TorchServe (${MODEL_NAME})"
torchserve --start --ncs \
  --disable-token-auth \
  --model-store "$MODEL_STORE" \
  --models "${MODEL_NAME}=${MODEL_NAME}.mar" \
  --ts-config "$PROJECT_DIR/torchserve.properties"

sleep 20

echo "Checking TorchServe /ping"
curl -s "http://$HOSTNAME:8080/ping" || echo "/ping may 404 until models are loaded"

echo "Scaling ${MODEL_NAME} to 1 GPU worker via management API"
curl -s -X PUT "http://127.0.0.1:8081/models/${MODEL_NAME}" \
  -H "Content-Type: application/json" \
  -d '{"min_worker": 1, "max_worker": 1, "number_gpu": 1}' \
  || echo "Warning: failed to scale ${MODEL_NAME} to GPU"

curl -s "http://127.0.0.1:8081/models/${MODEL_NAME}" || true

# ------------------------------------------------------------------------------
# Main experiment loop
# ------------------------------------------------------------------------------
RUN_IDX=1
while [[ "$RUN_IDX" -le "$REPEATS" ]]; do
  SUBRUN_DIR="$RUN_DIR/run_${RUN_IDX}"
  mkdir -p "$SUBRUN_DIR"

  SEED_RUN=$((SEED + RUN_IDX - 1))
  POWER_CSV="$SUBRUN_DIR/power_samples_${MODEL_NAME}.csv"

  echo ""
  echo "==================== RUN $RUN_IDX / $REPEATS ===================="
  echo "SUBRUN_DIR=$SUBRUN_DIR"
  echo "SEED_RUN=$SEED_RUN"
  echo "POWER_CSV=$POWER_CSV"
  echo "================================================================="

  cat > "$SUBRUN_DIR/config_run_${MODEL_NAME}.json" <<EOF
{
  "job_name": "torchserve_${MODEL_NAME}",
  "slurm_job_id": "${SLURM_JOB_ID:-unknown}",
  "host": "${HOSTNAME}",
  "model_name": "${MODEL_NAME}",
  "window_s": ${WINDOW_S},
  "power_sample_period": ${POWER_SAMPLE_PERIOD},
  "idle_calibration_seconds": ${IDLE_CALIBRATION_SECONDS},
  "concurrency": ${CONCURRENCY},
  "warmup_requests": ${WARMUP_REQUESTS},
  "pattern": "${PATTERN}",
  "phases_json": "${PHASES_JSON}",
  "phase_duration_scale": ${PHASE_DURATION_SCALE},
  "seed_run": ${SEED_RUN},
  "run_index": ${RUN_IDX}
}
EOF

  echo "Running image workload against ${MODEL_NAME} across ALL phases (run $RUN_IDX)"

  python "$PROJECT_DIR/measure_with_zeus.py" \
    --window-name "${MODEL_NAME}_phases_run${RUN_IDX}" \
    --zeus-update-period 1.0 \
    --log-csv "$SUBRUN_DIR/zeus_windows_${MODEL_NAME}.csv" \
    --json-out "$SUBRUN_DIR/zeus_summary_${MODEL_NAME}.json" \
    -- \
    python "$PROJECT_DIR/client.py" \
      --url "http://$HOSTNAME:8080" \
      --mode image \
      --model-name "${MODEL_NAME}" \
      --pattern "$PATTERN" \
      --warmup-requests "$WARMUP_REQUESTS" \
      --concurrency "$CONCURRENCY" \
      --phases-json "$PHASES_JSON" \
      --phase-duration-scale "$PHASE_DURATION_SCALE" \
      --csv "$SUBRUN_DIR/requests_${MODEL_NAME}.csv" \
      --window-s "$WINDOW_S" \
      --power-csv "$POWER_CSV" \
      --power-sample-period "$POWER_SAMPLE_PERIOD" \
      --idle-calibration-seconds "$IDLE_CALIBRATION_SECONDS" \
      --random-seed "$SEED_RUN" \
    > "$SUBRUN_DIR/client_${MODEL_NAME}.log" 2>&1 || true

  WINDOW_SUFFIX=$(python - <<PY
from client import make_window_suffix
print(make_window_suffix(float("$WINDOW_S")))
PY
)

  WINDOWS_CSV="$SUBRUN_DIR/requests_${MODEL_NAME}_windows_${WINDOW_SUFFIX}s.csv"

  echo "Expecting windows CSV at: $WINDOWS_CSV"
  echo "Expecting NVML power samples at: $POWER_CSV"

  if [[ -f "$WINDOWS_CSV" && -f "$POWER_CSV" ]]; then
    echo "Post-processing: attaching NVML energy to windows CSV (run $RUN_IDX)"
    python "$PROJECT_DIR/client.py" \
      --attach-energy-only \
      --windows-csv "$WINDOWS_CSV" \
      --energy-csv "$POWER_CSV" \
      --idle-calibration-seconds "$IDLE_CALIBRATION_SECONDS" \
    > "$SUBRUN_DIR/attach_energy_${MODEL_NAME}.log" 2>&1 || true
  else
    echo "Skipping energy/window join for run $RUN_IDX: $WINDOWS_CSV or $POWER_CSV missing"
  fi

  RUN_IDX=$((RUN_IDX + 1))
done

echo "Fetching TorchServe metrics"
curl -s "http://$HOSTNAME:8082/metrics" > "$RUN_DIR/torchserve_metrics_${MODEL_NAME}.prom" \
  || echo "Failed to fetch TorchServe /metrics"

echo "Stopping TorchServe"
torchserve --stop || true
sleep 5

if [[ -d "$PROJECT_DIR/logs" ]]; then
  cp -r "$PROJECT_DIR/logs" "$RUN_DIR/torchserve_logs"
fi

tar -czf "$RUN_DIR/logs_${MODEL_NAME}.tar.gz" -C "$RUN_DIR" . 2>/dev/null || true

# ------------------------------------------------------------------------------
# Automatic merge + sanity-check analysis over all runs
# ------------------------------------------------------------------------------
echo "Running post-job analysis and merge over all runs in $RUN_DIR"

python "$PROJECT_DIR/analyze_and_merge_runs.py" \
  --runs-root "$RUN_DIR" \
  --model-name "$MODEL_NAME" \
  > "$RUN_DIR/analysis_${MODEL_NAME}.txt" 2>&1 || echo "Analysis script failed"

SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
MERGED_CSV="$RUN_DIR/merged_windows_${SANITIZED_MODEL_NAME}.csv"

# ------------------------------------------------------------------------------
# Append this job to a global experiment index CSV
# ------------------------------------------------------------------------------
INDEX_CSV="$PROJECT_DIR/experiment_index.csv"
if [[ ! -f "$INDEX_CSV" ]]; then
  echo "job_id,submitted_at,model_name,run_dir,repeats,concurrency,window_s,pattern,phases_json,seed_base,idle_calibration_seconds,phase_duration_scale" > "$INDEX_CSV"
fi
NOW_TS=$(date -Iseconds)
echo "${SLURM_JOB_ID:-unknown},${NOW_TS},${MODEL_NAME},${RUN_DIR},${REPEATS},${CONCURRENCY},${WINDOW_S},${PATTERN},${PHASES_JSON},${SEED},${IDLE_CALIBRATION_SECONDS},${PHASE_DURATION_SCALE}" >> "$INDEX_CSV"

echo "✅ TorchServe multi-phase job complete for ${MODEL_NAME}."
echo "  Top-level run dir: $RUN_DIR"
echo "  Merged windows CSV: $MERGED_CSV"
echo "  Analysis log: $RUN_DIR/analysis_${MODEL_NAME}.txt"
echo "  Logs archive: $RUN_DIR/logs_${MODEL_NAME}.tar.gz"
echo "  Experiment index updated at: $INDEX_CSV"