# VLUI Research Project

Locality-Constrained Spatial Indexing with Provable Cross-Region Interference (CRI) bounds.

This repository benchmarks spatial index structures under multiple workload families and measures cross-region interference using an empirical CRI framework plus live SPRT-based falsifiability tracking.

## 1. What This Project Does

The project compares six spatial index implementations on synthetic and real-world traces:

- R-tree (global baseline)
- LCHI (VLUI proposed index)
- Buffered R-tree
- Grid + Global B-tree
- LSM Spatial
- Learned Grid

For each index/workload run, it computes:

- Isolation score (lower is better): max off-diagonal |CRI|
- Mean cross-CRI (lower is better)
- Query latency statistics
- SPRT verdicts for zero-CRI claim support/rejection

## 2. Core Features

### 2.1 Index Implementations

Located in `src/indexes/`:

- `lchi_index.py`: partition-confined index with independent local structures per partition
- `rtree_index.py`: global R-tree baseline (single shared structure)
- `buffered_rtree.py`: global R-tree with write buffer flushes
- `grid_btree.py`: partitioned query routing with one global sorted backing structure
- `lsm_spatial.py`: LSM-style spatial index with memtable + leveled runs
- `learned_grid.py`: partitioned index with per-partition learned CDF predictor

### 2.2 Workload Generators

Located in `src/generators/`:

- `uniform.py`: uniform random control workload
- `zipf_spatial.py`: Zipf-skewed regional update pressure (primary CRI stressor)
- `moving_hotspot.py`: temporally non-stationary moving hotspot workload
- `adversarial.py`: burst-style adversarial workload to maximize global-index interference
- `real_dataset.py`: real dataset loaders for Porto Taxi and NYC Taxi style CSVs

### 2.3 CRI Measurement Framework

Located in `src/metrics/cri.py`:

- Windowed empirical CRI estimation (cross-effect of updates in region i on query latency in region j)
- Normalized and raw CRI matrix support
- Isolation score, mean cross-CRI, significance helpers
- Epsilon-isolation checks

### 2.4 SPRT Falsifiability Layer

Located in `src/analysis/sprt.py` and integrated in dashboard:

- Pairwise and index-level Wald SPRT decision tracking
- Verdict states: continue / accept_h0 / reject_h0
- Real-data warmup and recalibration support
- Two runtime modes in live dashboard:
  - `neutral` (default): pooled calibration across all indexes
  - `vlui_anchor`: legacy LCHI-anchored calibration mode

### 2.5 Live Web Dashboard

Backend and frontend:

- `scripts/web_dashboard.py`
- `scripts/templates/index.html`
- `scripts/templates/evaluator.html`

Capabilities:

- Live benchmark execution and polling metrics API
- Synthetic mode and real CSV upload mode
- Per-index timeseries and latest metrics cards
- CRI heatmaps and scoring table
- SPRT panel with verdicts and mean LLR history
- Evaluator view over scaling outputs (`/evaluator`)

## 3. Architecture Overview

### 3.1 Execution Flow

1. Workload generation (synthetic or real dataset trace)
2. Replay operations against each index
3. Record update/query events for CRI estimation
4. Compute CRI matrix + aggregate metrics periodically
5. Update SPRT state from CRI observations
6. Expose live snapshots via API for frontend polling

### 3.2 Main Modules

- `src/models/`: operations, points, regions, traces
- `src/generators/`: synthetic + real trace builders
- `src/indexes/`: index implementations under comparison
- `src/metrics/`: CRI computation and derived metrics
- `src/analysis/`: SPRT logic
- `src/benchmark/`: offline experiment runner, executor, visualization
- `scripts/`: runnable experiment scripts + dashboard server
- `tests/`: generator/integration/API validations

## 4. Repository Structure

```text
VLUI/
├─ configs/
│  └─ experiment_baseline.yaml
├─ data/
│  ├─ porto_taxi/
│  ├─ real/
│  ├─ synthetic/
│  └─ sample_porto.csv
├─ output/
├─ results/
├─ paper/
├─ scripts/
│  ├─ web_dashboard.py
│  ├─ start_web_dashboard.py
│  ├─ run_real_benchmark.py
│  ├─ run_nyc_benchmark.py
│  ├─ run_grid_sweep.py
│  ├─ run_window_sweep.py
│  ├─ run_all.py
│  ├─ collect_scaling_curve.py
│  ├─ compare_datasets.py
│  └─ templates/
├─ src/
│  ├─ analysis/
│  ├─ benchmark/
│  ├─ datasets/
│  ├─ generators/
│  ├─ indexes/
│  ├─ metrics/
│  └─ models/
├─ tests/
│  ├─ test_generators.py
│  └─ test_web_dashboard_api.py
├─ pyproject.toml
├─ requirements-web.txt
├─ README_WEB.md
└─ README.md
```

## 5. Environment & Dependencies

### 5.1 Python Version

- Python >= 3.10

### 5.2 Core Dependencies (from `pyproject.toml`)

- numpy
- scipy
- matplotlib
- pandas
- pyyaml
- tqdm
- rtree

### 5.3 Web Dashboard Dependencies (from `requirements-web.txt`)

- flask
- flask-cors

### 5.4 Dev/Test Dependencies

- pytest
- pytest-benchmark
- black
- mypy

## 6. Recreate Locally (From GitHub)

## 6.1 Clone

```bash
git clone https://github.com/redwolf261/VLUI.git
cd VLUI
```

## 6.2 Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 6.3 Install Dependencies

```bash
pip install -e .
pip install -r requirements-web.txt
pip install pytest pytest-benchmark black mypy
```

## 6.4 Run Tests

```bash
pytest tests -q
```

Current suite composition:

- `tests/test_generators.py`: 14 tests (workload behavior, CRI sanity, integration)
- `tests/test_web_dashboard_api.py`: 4 tests (API guardrails and validation)

Total: 18 tests.

## 6.5 Run Live Dashboard

### Option A (recommended)

```bash
python scripts/start_web_dashboard.py
```

### Option B

```bash
python scripts/web_dashboard.py
```

Open:

- `http://127.0.0.1:5000` (live dashboard)
- `http://127.0.0.1:5000/evaluator` (scaling evaluator frontend)

## 6.6 Run Main Offline Experiment Suite

```bash
python scripts/run_all.py
```

Quick smoke version:

```bash
python scripts/run_all.py --quick
```

With no figure generation:

```bash
python scripts/run_all.py --skip-plots
```

## 6.7 Real Dataset Benchmarks

### Porto Taxi

```bash
python scripts/run_real_benchmark.py --n-points 1000 --n-queries 200
```

### NYC Taxi

```bash
python scripts/run_nyc_benchmark.py --n-points 1000 --n-queries 200
```

### Grid Sweep

```bash
python scripts/run_grid_sweep.py --n-points 10000 --n-queries 2000
```

### Window Sweep (Lemma-3 consistency style validation)

```bash
python scripts/run_window_sweep.py --n-points 100000 --n-queries 20000
```

## 6.8 Scaling Curve Assembly + Evaluator Data

After you have multiple scale run summaries under `output/scale/<n>/summary.csv`:

```bash
python scripts/collect_scaling_curve.py
```

Then view evaluator at `/evaluator`.

## 7. Real Data Input Formats

Implemented in `src/generators/real_dataset.py`.

### 7.1 Porto Taxi CSV

- Uses `POLYLINE` JSON array field
- Extracts trip origin points for inserts

### 7.2 NYC Taxi CSV

- Auto-detects lon/lat columns (e.g., pickup_longitude/pickup_latitude)

### 7.3 Generic Lon/Lat CSV

- Auto-detection heuristics for common longitude/latitude headers

### 7.4 Sample Data Generator

```bash
python scripts/gen_sample_porto.py
```

Creates `data/sample_porto.csv` for quick local experiments.

## 8. Dashboard API Summary

Base server: `scripts/web_dashboard.py`

- `GET /api/metrics` : current live state snapshot
- `GET /api/dataset_info` : pending uploaded dataset metadata
- `POST /api/upload` : upload real dataset CSV
- `POST /api/upload/clear` : clear pending dataset
- `POST /api/start` : start benchmark (supports JSON `sprt_mode`)
- `POST /api/reset` : reset state (blocked while running)
- `GET /api/scaling_curve` : evaluator data from `output/scale/scaling_curve.csv`

## 9. Security/Operational Notes

Recent hardening in dashboard backend includes:

- Thread-safe state access with lock
- Upload size cap (8 MiB)
- Input range checks for `n_points` and `n_queries`
- Reset blocked during active run
- API CORS allowlist via `VLUI_CORS_ORIGINS`

For local defaults, allowed origins are localhost variants on port 5000.

## 10. Reproducibility Notes

- Most scripts use deterministic seeds (commonly 42 by default)
- Config-driven experiments are defined in `configs/experiment_baseline.yaml`
- Output artifacts are written as CSV/JSON/NumPy for downstream analysis
- Scaling/evaluator flow expects `output/scale/scaling_curve.csv`

## 11. Typical End-to-End Reproduction Checklist

1. Clone repo
2. Create/activate venv
3. Install dependencies
4. Run `pytest tests -q`
5. Run one benchmark script (e.g., `run_real_benchmark.py`)
6. Start dashboard and verify live metrics
7. (Optional) run scale jobs and build evaluator curve

## 12. Troubleshooting

- `ModuleNotFoundError`: ensure venv activated and dependencies installed
- Dashboard upload rejected: verify file size <= 8 MiB and valid CSV format
- `413` on upload: reduce dataset file or preprocess/downsample
- `409` on reset/start: benchmark currently running; wait for completion
- Missing evaluator data: run scaling jobs then `collect_scaling_curve.py`

## 13. Attribution

Project name: VLUI Research

Focus: Locality-constrained spatial indexing and CRI falsifiability under realistic and adversarial workloads.
