#!/bin/bash
#SBATCH --job-name=torchserve_gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=GPU

set -u
set -o pipefail

echo "📦 Starting TorchServe job on $HOSTNAME"
echo "SLURM_JOBID=${SLURM_JOB_ID:-unknown} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---------------------- Paths & caches (persistent) -------------------------
PROJECT_DIR="$SLURM_SUBMIT_DIR"
ENV_DIR="$PROJECT_DIR/.venv_torchserve"
RUN_DIR="$PROJECT_DIR/runs/run_$(date +%Y%m%d_%H%M%S)"
MODEL_STORE="$PROJECT_DIR/model_store"
mkdir -p "$RUN_DIR" "$MODEL_STORE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME="$PROJECT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$PROJECT_DIR/.hf_cache"
export TORCH_HOME="$PROJECT_DIR/.torch_cache"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "RUN_DIR=$RUN_DIR"

# Make sure we don't inherit a weird PYTHONPATH from a login shell
unset PYTHONPATH || true

# ----------------------------- Modules --------------------------------------
module purge
module use /share/apps/eb/modules/all || true

# Use a clean Python module (this matched your old working setup)
module load Python/3.11.5-GCCcore-13.2.0 || module load Python || {
  echo "❌ Failed to load Python module"
  exit 1
}
module load CUDA || true
module load Java || true

echo "🔧 GPU inventory:"
nvidia-smi || true

# ------------------------ Persistent virtualenv ------------------------------
if [[ ! -d "$ENV_DIR" ]]; then
  echo "🐍 Creating persistent venv at $ENV_DIR"
  python -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

# Prefer binary wheels, avoid heavy source builds
export PIP_ONLY_BINARY=":all:"
export PIP_NO_BUILD_ISOLATION=1

# -------------------- Core Python deps --------------------------------------
pip install --quiet --upgrade pip setuptools wheel

# Install torch + torchvision with CUDA wheels (cu121)
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 --no-cache-dir || {
  echo "❌ Failed to install torch/torchvision"
  exit 1
}

# TorchServe + model archiver + utilities + Captum + Zeus
pip install --quiet \
  torchserve torch-model-archiver pillow requests transformers

# Captum is required by TorchServe's image_classifier handler
pip install --quiet --no-deps captum

# Zeus for GPU energy / power measurement
pip install --quiet zeus

# ------------------ Build ResNet-18 MAR (external script) -------------------
echo "📦 Ensuring resnet-18.mar exists…"
python "$PROJECT_DIR/build_resnet_mar.py" || {
  echo "❌ Failed to build resnet-18.mar"
  exit 1
}

# ------------------------------ Start TorchServe -----------------------------
cd "$PROJECT_DIR"

echo "🚀 Starting TorchServe…"
torchserve --start --ncs \
  --disable-token-auth \
  --model-store "$MODEL_STORE" \
  --models resnet-18=resnet-18.mar \
  --ts-config "$PROJECT_DIR/torchserve.properties"

# TorchServe logs to ./logs by default
sleep 20
echo "⏳ Checking TorchServe /ping…"
curl -s "http://$HOSTNAME:8080/ping" || echo "ℹ️ /ping may 404 until models are loaded"

# ------------------------------ Run client -----------------------------------
echo "💥 Image workload against resnet-18 (Zeus monitored)…"
python "$PROJECT_DIR/measure_with_zeus.py" \
  --window-name "resnet18_image_burst" \
  --log-csv "$RUN_DIR/zeus_windows.csv" \
  --json-out "$RUN_DIR/zeus_summary.json" \
  -- \
  python "$PROJECT_DIR/client.py" \
    --url "http://$HOSTNAME:8080" \
    --mode image \
    --model-name resnet-18 \
    --duration 300 \
    --burst 20 \
    --idle 10 \
    --rps 15 \
    --concurrency 16 \
    --pattern burst \
    --warmup-requests 100 \
    --csv "$RUN_DIR/requests.csv" \
  > "$RUN_DIR/client_image.log" 2>&1 || true

# ------------------------- TorchServe metrics snapshot -----------------------
echo "📊 Fetching TorchServe metrics…"
curl -s "http://$HOSTNAME:8082/metrics" > "$RUN_DIR/torchserve_metrics.prom" \
  || echo "⚠️ Failed to fetch TorchServe /metrics"

# ------------------------------- Teardown ------------------------------------
echo "🧹 Stopping TorchServe…"
torchserve --stop || true
sleep 5

# Copy TorchServe logs into this run dir
if [[ -d "$PROJECT_DIR/logs" ]]; then
  cp -r "$PROJECT_DIR/logs" "$RUN_DIR/torchserve_logs"
fi

tar -czf "$RUN_DIR/logs.tar.gz" -C "$RUN_DIR" . 2>/dev/null || true
echo "✅ Job complete. Logs archived to $RUN_DIR/logs.tar.gz"
