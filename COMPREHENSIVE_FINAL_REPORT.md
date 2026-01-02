# Progress Report (Semester 1) — Individual Thesis
**Student:** <Your Name> · **Supervisor:** <Supervisor Name> · **Program:** <Department> · **Date:** <DD Mon YYYY>

## 1) Topic and methodology (precise description)
**Topic.** This thesis leverages **machine learning** to **predict idleness in microservice workloads**. The goal is to build and validate an ML-ready pipeline that labels fixed time windows as **idle vs. busy**, using both **workload activity** (requests) and **system telemetry** (GPU power/energy), to support energy-aware decisions (e.g., scaling, power-state transitions).

**Methodology.**
- **Service under test:** A GPU-backed inference microservice deployed with TorchServe (`torchserve.properties`): `number_of_gpu=1`, `default_workers_per_model=1`, inference `:8080`, management `:8081`, metrics `:8082`, Prometheus enabled and system metrics disabled.
- **Workload generation:** `client.py` sends inference requests to `POST /predictions/<model>` under **steady**, **Poisson**, and **burst** patterns; supports multi-phase workloads via phases JSON and reproducible runs via fixed seeds.
- **Telemetry + feature extraction:** Per-request metrics (timestamps, latency, status) are logged and aggregated into fixed windows (e.g., 0.5–1.0s) computing counts, latency statistics, error rate, and RPS. In parallel, GPU power is sampled using NVML (`pynvml`) at ~0.1s intervals, producing `power_w` and derived `energy_j`. Energy is joined into windows to compute `energy_j_per_window`, `avg_power_w`, and `energy_per_request`.
- **Labels:** Primary supervised label is **ground-truth idleness** `label_idle_gt` (idle if window has zero requests). A secondary label `energy_idle_label` is produced using an energy-based threshold (manual or auto-calibrated from an idle baseline period).

## 2) Previous work on the topic (brief)
Prior research in energy-aware systems and ML serving uses request-rate heuristics and hardware telemetry (power/utilization) for idleness detection, autoscaling, and power management. ML-based predictors are promising but require well-aligned, validated datasets linking workload dynamics to system signals. This thesis focuses on producing such a dataset and evaluating predictive models.

## 3) Stage of work + results (end of Semester 1)
Pipeline implemented: TorchServe config + load generator + NVML sampling + window aggregation + energy join + multi-run merge and sanity checks (`analyze_and_merge_runs.py`). Initial experiments show strong data quality:

| Metric | Value | Assessment |
|---|---:|---|
| Windows merged | 6007 | ✓ sufficient for ML |
| Label consistency conflicts | 0 idle, 0 busy | ✓ perfect |
| Class balance (`label_idle_gt`) | 67.37% busy / 32.63% idle | ✓ reasonable |
| Class balance (`energy_idle_label`) | 64.04% idle / 35.96% busy | ✓ useful complementary view |
| Corr(req, energy) | 0.4520 | ✓ moderate signal |
| Corr(req, avg power) | 0.4520 | ✓ moderate signal |
| Avg power (mean ± std) | 57.74W ± 1.74W | ✓ stable |
| Auto idle threshold | 57.63W | ✓ sensible |
| Errors/timeouts | none observed | ✓ healthy |

## 4) Timetable for next semester (implementation plan)
**Weeks 1–2:** Finalize dataset schema and ML target definition; enforce leakage-safe splits (split by runs); add run summaries.  
**Weeks 3–5:** Train baseline models (logistic regression, tree-based); evaluate with F1/ROC-AUC; ablations (request-only vs power-only vs combined).  
**Weeks 6–8:** Feature engineering (rolling stats, lag features) and temporal modeling if needed; cross-pattern generalization tests.  
**Weeks 9–11:** Scale experiments across more models/workloads; validate robustness to baseline drift and sampling jitter (use real Δt for energy integration).  
**Weeks 12–14:** Write methodology/results; finalize figures/tables; document reproducibility and deliverables.