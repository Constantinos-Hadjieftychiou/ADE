# ADE – Evaluating GPU Idleness in ML Inference Microservices

This repository contains the implementation and experimental framework developed for the thesis:

**"Evaluating Idleness of Machine Learning Microservices on GPUs"**

The project studies GPU idleness, queue dynamics, energy behavior, and workload effects in GPU-based inference systems using TorchServe.

---

# Project Overview

The framework deploys deep learning models using TorchServe and evaluates GPU behavior under controlled Poisson-distributed workloads.

The system collects:
- Request arrival/completion logs
- GPU utilization metrics
- GPU power and energy telemetry
- Queue backlog statistics
- Window-based aggregated datasets

The experiments are executed on GPU-enabled HPC infrastructure using SLURM.

---

# Repository Structure

```text
ADE/
│
├── configs/               # TorchServe and experiment configs
├── datasets/              # Generated datasets and logs
├── models/                # TorchScript / MAR models
├── monitoring/            # GPU telemetry collection
├── workload/              # Poisson workload generator
├── analysis/              # Dataset analysis scripts
├── scripts/               # Utility scripts
├── logs/                  # Experiment logs
├── results/               # Generated figures/results
├── run_experiments.sbatch # Main SLURM execution script
└── README.md
```

---

# Requirements

- Python 3.11+
- CUDA 12+
- NVIDIA GPU
- TorchServe
- PyTorch
- SLURM environment (for HPC execution)

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Constantinos-Hadjieftychiou/ADE.git
cd ADE
```

Create environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Experiments

The experiments are executed using the SLURM batch script.

Submit the job using:

```bash
sbatch run_experiments.sbatch
```

This script:
- Starts TorchServe
- Deploys the model
- Executes workload phases
- Collects GPU telemetry
- Generates datasets and logs

---

# Experimental Workflow

```text
Workload Generator
        ↓
TorchServe Inference Service
        ↓
GPU Execution
        ↓
Telemetry Collection
        ↓
Window-Based Dataset Construction
        ↓
Analysis & Visualization
```

---

# Thesis Objectives

The framework is designed to:
- Analyze GPU idleness
- Study queue backlog dynamics
- Measure GPU power behavior
- Evaluate energy efficiency
- Investigate hidden saturation effects

---

# Author

Constantinos Hadjieftychiou  
Department of Computer Science  
University of Cyprus

---

# License

This repository is provided for academic and research purposes.
