#!/bin/bash

#SBATCH --job-name=torchserve_resnet18_phase
#SBATCH --output=slurm-resnet18-%j.out
#SBATCH --error=slurm-resnet18-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=GPU

set -u
set -o pipefail

########################################
# Select a single phase by name
# (must match the "name" field in phases.json)
########################################
PHASE_NAME="high-traffic-steady-25rps"   # <-- change this

echo "Starting TorchServe ResNet-18 single-phase job on $HOSTNAME"
echo "SLURM_JOBID=${SLURM_JOB_ID:-unknown} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Using PHASE_NAME=${PHASE_NAME}"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
ENV_DIR="$PROJECT_DIR/.venv_torchserve"
RUN_DIR="$PROJECT_DIR/runs/resnet18_${PHASE_NAME}_$(date +%Y%m%d_%H%M%S)"
MODEL_STORE="$PROJECT_DIR/model_store"

# Idle baseline ~53.64 W → use a slightly higher threshold
IDLE_POWER_THRESHOLD_W=58.0

# <<< TUNE THIS: window size in seconds for aggregated metrics >>>
# Examples:
#   WINDOW_S=1.0  # 1s
#   WINDOW_S=0.5  # 500ms
#   WINDOW_S=0.25 # 250ms
WINDOW_S=0.5

# High-frequency power sampling (seconds) via NVML
POWER_SAMPLE_PERIOD=0.1

mkdir -p "$RUN_DIR" "$MODEL_STORE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.hf_cache"
export TORCH_HOME="$PROJECT_DIR/.torch_cache"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "RUN_DIR=$RUN_DIR"

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

echo "Starting TorchServe (ResNet-18)"
torchserve --start --ncs \
  --disable-token-auth \
  --model-store "$MODEL_STORE" \
  --models resnet-18=resnet-18.mar \
  --ts-config "$PROJECT_DIR/torchserve.properties"

sleep 20

echo "Checking TorchServe /ping"
curl -s "http://$HOSTNAME:8080/ping" || echo "/ping may 404 until models are loaded"

echo "Running image workload against resnet-18 (Zeus monitored), phase=${PHASE_NAME}"

POWER_CSV="$RUN_DIR/power_samples_resnet18.csv"

python "$PROJECT_DIR/measure_with_zeus.py" \
  --window-name "resnet18_${PHASE_NAME}" \
  --zeus-update-period 1.0 \
  --log-csv "$RUN_DIR/zeus_windows_resnet18.csv" \
  --json-out "$RUN_DIR/zeus_summary_resnet18.json" \
  -- \
  python "$PROJECT_DIR/client.py" \
    --url "http://$HOSTNAME:8080" \
    --mode image \
    --model-name resnet-18 \
    --pattern steady \
    --warmup-requests 100 \
    --concurrency 16 \
    --phases-json "$PROJECT_DIR/phases.json" \
    --phase-name "$PHASE_NAME" \
    --csv "$RUN_DIR/requests_resnet18.csv" \
    --window-s "$WINDOW_S" \
    --power-csv "$POWER_CSV" \
    --power-sample-period "$POWER_SAMPLE_PERIOD" \
    --random-seed 42 \
  > "$RUN_DIR/client_resnet18_${PHASE_NAME}.log" 2>&1 || true

# Figure out the window CSV name based on WINDOW_S (must match client.py logic):
WINDOW_SUFFIX=$(python - <<PY
import sys
ws = float(sys.argv[1])
suffix = f"{ws:.3f}".rstrip("0").rstrip(".").replace(".", "p")
print(suffix)
PY
"$WINDOW_S"
)

WINDOWS_CSV="$RUN_DIR/requests_resnet18_windows_${WINDOW_SUFFIX}s.csv"

echo "Expecting windows CSV at: $WINDOWS_CSV"

# Now attach NVML-based energy to the windows CSV in a separate pass.

if [[ -f "$WINDOWS_CSV" && -f "$POWER_CSV" ]]; then
  echo "Post-processing: attaching NVML energy to windows CSV for phase=${PHASE_NAME}"
  python "$PROJECT_DIR/client.py" \
    --attach-energy-only \
    --windows-csv "$WINDOWS_CSV" \
    --energy-csv "$POWER_CSV" \
    --idle-power-threshold-w "$IDLE_POWER_THRESHOLD_W" \
    > "$RUN_DIR/attach_energy_resnet18_${PHASE_NAME}.log" 2>&1 || true
else
  echo "Skipping energy/window join: $WINDOWS_CSV or $POWER_CSV missing"
fi

echo "Fetching TorchServe metrics"
curl -s "http://$HOSTNAME:8082/metrics" > "$RUN_DIR/torchserve_metrics_resnet18_${PHASE_NAME}.prom" \
  || echo "Failed to fetch TorchServe /metrics"

echo "Stopping TorchServe"
torchserve --stop || true
sleep 5

if [[ -d "$PROJECT_DIR/logs" ]]; then
  cp -r "$PROJECT_DIR/logs" "$RUN_DIR/torchserve_logs"
fi

tar -czf "$RUN_DIR/logs_resnet18_${PHASE_NAME}.tar.gz" -C "$RUN_DIR" . 2>/dev/null || true

echo "ResNet-18 single-phase job complete (phase=${PHASE_NAME}). Logs archived to $RUN_DIR/logs_resnet18_${PHASE_NAME}.tar.gz"
