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

# Prefer binary wheels (avoid heavy source builds)
export PIP_ONLY_BINARY=":all:"
export PIP_NO_BUILD_ISOLATION=1

# ----------------------------- Modules --------------------------------------
module purge
module use /share/apps/eb/modules/all || true
module load Python/3.11.5-GCCcore-13.2.0 || module load Python || true
module load CUDA || true
module load Java || true   # adjust module name if your cluster uses a different Java module

echo "🔧 GPU inventory:"
nvidia-smi || true

# ------------------------ Persistent virtualenv ------------------------------
if [[ ! -d "$ENV_DIR" ]]; then
  echo "🐍 Creating persistent venv at $ENV_DIR"
  python -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"

python -c 'import sys; print("Python:", sys.version)'
pip install --quiet --upgrade pip setuptools wheel || true

# -------------------- Core Python deps --------------------------------------
# torch + torchvision with CUDA (cu121 wheels)
python - <<'PY' || _miss=1
import importlib
importlib.import_module("torch")
importlib.import_module("torchvision")
PY
if [[ "${_miss:-0}" == "1" ]]; then
  echo "⬇️ Installing torch/torchvision (cu121 wheels)…"
  pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 --no-cache-dir || echo "⚠️ torch/vision install failed"
fi
unset _miss

# TorchServe + model archiver + utils + captum (required by image_classifier handler)
pip install --quiet torchserve torch-model-archiver transformers pillow requests captum || true

python - <<'PY'
import torch
print(f"Torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ------------------ Build ResNet-18 MAR if missing --------------------------
if [[ ! -f "$MODEL_STORE/resnet-18.mar" ]]; then
  echo "📦 Building resnet-18.mar for TorchServe…"
  cd "$PROJECT_DIR"

  # 1) Create TorchScript file
  python - <<'PY'
from torchvision import models
import torch

print("Loading ResNet-18 with ImageNet weights…")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()
scripted = torch.jit.script(model)
scripted.save("resnet18_scripted.pt")
print("Saved TorchScript model to resnet18_scripted.pt")
PY

  # 2) Build simple index_to_name.json from ImageNet classes (for nicer responses)
  python - <<'PY'
import json, urllib.request
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
print("Downloading ImageNet class names…")
txt = urllib.request.urlopen(url).read().decode("utf-8").strip().splitlines()
mapping = {str(i): name for i, name in enumerate(txt)}
with open("index_to_name.json", "w") as f:
    json.dump(mapping, f)
print("Wrote index_to_name.json")
PY

  # 3) Archive into .mar using built-in image_classifier handler
  torch-model-archiver \
    --model-name resnet-18 \
    --version 1.0 \
    --serialized-file resnet18_scripted.pt \
    --handler image_classifier \
    --extra-files index_to_name.json \
    --export-path "$MODEL_STORE" \
    --force

  echo "✅ Created $MODEL_STORE/resnet-18.mar"
fi

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
echo "💥 Image workload against resnet-18…"
python "$PROJECT_DIR/client.py" \
  --url "http://$HOSTNAME:8080" \
  --mode image \
  --model-name resnet-18 \
  --duration 300 \
  --burst 20 \
  --idle 10 \
  --rps 10 > "$RUN_DIR/client_image.log" 2>&1 || true

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
