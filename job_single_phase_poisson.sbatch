#!/bin/bash
# SLURM job script to run TorchServe + load test for a SINGLE Poisson phase
# 
# Usage:
#   sbatch job_single_phase_poisson.sbatch                          # Uses default phase
#   sbatch --export=ALL,PHASE_NAME=poisson-5min-10rps job_single_phase_poisson.sbatch
#   sbatch --export=ALL,PHASE_NAME=poisson-10min-20rps,MODEL_NAME=resnet-50 job_single_phase_poisson.sbatch
#
# Available phases in phases_poisson.json:
#   - poisson-2min-1rps    (120s, 1 req/s)
#   - poisson-2min-5rps    (120s, 5 req/s)
#   - poisson-2min-10rps   (120s, 10 req/s)
#   - poisson-2min-20rps   (120s, 20 req/s)
#   - poisson-5min-1rps    (300s, 1 req/s)
#   - poisson-5min-5rps    (300s, 5 req/s)
#   - poisson-5min-10rps   (300s, 10 req/s)
#   - poisson-5min-20rps   (300s, 20 req/s)
#   - poisson-10min-1rps   (600s, 1 req/s)
#   - poisson-10min-5rps   (600s, 5 req/s)
#   - poisson-10min-10rps  (600s, 10 req/s)
#   - poisson-10min-20rps  (600s, 20 req/s)
#   - poisson-5min-idle    (300s, 0 req/s)

#SBATCH --job-name=poisson_single
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=GPU

set -euo pipefail

# =============================================================================
# CONFIGURATION - Change PHASE_NAME to run a different phase
# =============================================================================
PHASE_NAME="${PHASE_NAME:-poisson-5min-10rps}"      # <-- CHANGE THIS for different phases
MODEL_NAME="${MODEL_NAME:-resnet-18}"                # Model to test
PHASES_JSON="${PHASES_JSON:-$SLURM_SUBMIT_DIR/phases_poisson.json}"  # Poisson phases file

# Other settings (usually don't need to change)
IDLE_CALIBRATION_SECONDS="${IDLE_CALIBRATION_SECONDS:-60}"
WINDOW_S="${WINDOW_S:-0.5}"
POWER_SAMPLE_PERIOD="${POWER_SAMPLE_PERIOD:-0.1}"
CONCURRENCY="${CONCURRENCY:-8}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-50}"
SEED="${SEED:-42}"

# =============================================================================
# Setup paths
# =============================================================================
PROJECT_DIR="$SLURM_SUBMIT_DIR"
ENV_DIR="$PROJECT_DIR/.venv_torchserve"
RUN_DIR="$PROJECT_DIR/runs/${MODEL_NAME}_${PHASE_NAME}_$(date +%Y%m%d_%H%M%S)"
MODEL_STORE="$PROJECT_DIR/model_store"
mkdir -p "$RUN_DIR" "$MODEL_STORE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME="$PROJECT_DIR
