#!/bin/bash
#SBATCH --job-name=resnet18_idle_baseline
#SBATCH --output=slurm-idle-baseline-%j.out
#SBATCH --error=slurm-idle-baseline-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=GPU

set -u
set -o pipefail

echo "Starting ResNet-18 idle-baseline job on $HOSTNAME"
echo "SLURM_JOBID=${SLURM_JOB_ID:-unknown} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

PROJECT_DIR="$SLURM_SUBMIT_DIR"
ENV_DIR="$PROJECT_DIR/.venv_torchserve"
RUN_DIR="$PROJECT_DIR/runs/resnet18_idle_baseline_$(date +%Y%m%d_%H%M%S)"
MODEL_STORE="$PROJECT_DIR/model_store"

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
  echo "Creating virtualenv at $ENV_DIR"
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
  torchserve torch-model-archiver pillow requests transformers pandas numpy

pip install --quiet --no-deps captum
pip install --quiet zeus

echo "Ensuring .mar files exist (resnet-18, resnet-50, mobilenet-v2)"
python "$PROJECT_DIR/build_resnet_mar.py" || {
  echo "Failed to build .mar files"
  exit 1
}

cd "$PROJECT_DIR"

echo "Starting TorchServe (ResNet-18) for idle-baseline measurement"
torchserve --start --ncs \
  --disable-token-auth \
  --model-store "$MODEL_STORE" \
  --models resnet-18=resnet-18.mar \
  --ts-config "$PROJECT_DIR/torchserve.properties"

sleep 20

echo "Checking TorchServe /ping"
curl -s "http://$HOSTNAME:8080/ping" || echo "/ping may fail until models are fully loaded"

echo "Measuring idle baseline with Zeus (TorchServe up, no traffic)"
python "$PROJECT_DIR/measure_idle_baseline.py" \
  --window-name "resnet18_idle_baseline" \
  --duration 600 \
  --log-csv "$RUN_DIR/idle_baseline_resnet18.csv" \
  --json-out "$RUN_DIR/idle_baseline_resnet18.json"

echo "Stopping TorchServe"
torchserve --stop || true
sleep 5

if [[ -d "$PROJECT_DIR/logs" ]]; then
  cp -r "$PROJECT_DIR/logs" "$RUN_DIR/torchserve_logs"
fi

tar -czf "$RUN_DIR/idle_baseline_resnet18.tar.gz" -C "$RUN_DIR" . 2>/dev/null || true

echo "Idle-baseline job complete. Outputs in $RUN_DIR"
