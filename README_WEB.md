# 🔬 CRI Spatial Index Benchmark Web Dashboard

A professional, real-time web-based visualization for spatial index benchmarking, similar to neural network training dashboards.

## Features

✨ **Real-time Metrics Display**
- Live isolation score evolution (R-tree vs LCHI)
- Cross-region interference (CRI) tracking
- Query latency distribution histogram
- Operation progress bar

📊 **Interactive Charts**
- Isolation Score Timeline (animated line chart)
- Cross-CRI Evolution (dual-axis comparison)
- Query Latency Distribution (histogram)
- Live metric updates every 500ms

📈 **Performance Comparison**
- Side-by-side metric comparison table
- Automatic improvement calculation
- Color-coded status badges
- Real-time delta computation

⚡ **Live Statistics**
- Mean query latency (ns)
- Operations per second
- Isolation score delta
- Progress percentage

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│         Web Browser (HTML5/CSS3/JavaScript)                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Chart.js (real-time line/histogram charts)          │  │
│  │  Progress bar + stat boxes + comparison table        │  │
│  │  Auto-updating via Fetch API (500ms polling)         │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬─────────────────────────────────────┘
                       │ HTTP/CORS
┌──────────────────────▼─────────────────────────────────────┐
│            Flask Backend (Python)                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  /api/metrics  → Read benchmark state (JSON)         │  │
│  │  /api/start    → Launch benchmark in bg thread       │  │
│  │  /api/reset    → Reset all metrics                   │  │
│  │                                                      │  │
│  │  run_benchmark() → BG thread execution:              │  │
│  │    - Generate Zipf-skewed workload                   │  │
│  │    - Replay ops on RTTRee + LCHI                     │  │
│  │    - Update global state every 40 ops                │  │
│  │    - Compute CRI via windowed regression             │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬─────────────────────────────────────┘
                       │ In-process
┌──────────────────────▼──────────────────────────────────────┐
│         VLUI Research Framework                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  - ZipfSpatialGenerator (500 ops)                    │   │
│  │  - NaiveRTreeIndex vs LCHIIndex                      │   │
│  │  - CRIMeasurer (windowed OLS + bootstrap)            │   │
│  │  - Real-time CRI computation                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors
```

### 2. Start the Server

**Option A: Using the startup script**
```bash
cd c:\Users\HP\Projects\VLUI\scripts
python start_web_dashboard.py
```

**Option B: Direct Python**
```bash
cd c:\Users\HP\Projects\VLUI\scripts
python web_dashboard.py
```

### 3. Open in Browser

```
http://localhost:5000
```

### 4. Run the Benchmark

1. Click **"▶ Start Benchmark"** button
2. Watch real-time metrics update:
   - Progress bar advances
   - Charts animate with new data
   - Statistics update every 500ms
3. Wait for completion (≈30-40 seconds for 500 ops)

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  🔬 CRI Spatial Index Benchmark Dashboard    [▶ Start] [↻ Reset] │
├─────────────────────────────────────────────────────────────────┤
│ Progress: [████████░░░░░░░░░░░░░░░░░░░░░░░░] 50.0%             │
│                                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │   250    │  │  2.455   │  │  1.704   │  │   9.6s   │        │
│  │ Ops Done │  │ R-tree I │  │ LCHI I   │  │  Elapsed │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ ┌──────────────────────────┐  ┌──────────────────────────┐     │
│ │ Isolation Score Timeline │  │  Cross-CRI Evolution     │     │
│ │  [Line chart animating]  │  │  [Line chart animating]  │     │
│ │                          │  │                          │     │
│ │  R-tree (red) ―――――      │  │  R-tree (red) ―――――      │     │
│ │  LCHI  (orange) ―――――    │  │  LCHI  (orange) ―――――    │     │
│ └──────────────────────────┘  └──────────────────────────┘     │
│                                                                │
├────────────────────────────────────────────────────────────────┤
│                 Performance Comparison                         │
│                                                                │
│ Metric              │ R-tree   │ LCHI     │ Improvement        │
│ ─────────────────────┼──────────┼──────────┼───────────────    │
│ Isolation (higher) │ 2.4552   │ 1.7045   │ -30.6% ⚠️           │
│ Mean CRI (lower)  │ 0.457898 │ 0.316245 │ -30.9% ✓             │
│ ─────────────────────┴──────────┴──────────┴────────────────   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌──────────────────────────┐  ┌──────────────────────────┐      │
│ │ Query Latency Distrib.   │  │ Live Metrics             │      │
│ │  [Histogram updating]    │  │  ┌──────────────────┐    │      │
│ │                          │  │  │ 38017 ns μ       │    │      │
│ │  ▁▂▃▆████▆▃▂▁          │  │  │ 52 ops/sec       │    │      │
│ │  └──────────────────┘    │  │  │ +0.0327 Δ iso    │    │      │
│ └──────────────────────────┘  │  └──────────────────┘    │      │
│                               └──────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

### `GET /api/metrics`
Returns current benchmark state (polled every 500ms):

```json
{
  "state": true,
  "progress": 50.0,
  "current_op": 250,
  "total_ops": 500,
  "elapsed_time": 9.6,
  "metrics": {
    "rtree": {
      "isolation": [0, 2.41, 2.42, ...],
      "mean_cri": [0, 0.45, 0.46, ...],
      "timestamps": [0, 1.1, 2.9, ...]
    },
    "lchi": {
      "isolation": [0, 4.84, 2.01, ...],
      "mean_cri": [0, 0.62, 0.35, ...],
      "timestamps": [0, 1.1, 2.9, ...]
    }
  },
  "current": {
    "rtree_isolation": 2.4552,
    "lchi_isolation": 1.7045,
    "rtree_cri": 0.4579,
    "lchi_cri": 0.3162
  },
  "query_latencies": [38000, 39500, ...]
}
```

### `POST /api/start`
Launch benchmark in background thread and begin updates.

**Response:**
```json
{"status": "started"}
```

### `POST /api/reset`
Reset benchmark state and clear metrics.

**Response:**
```json
{"status": "reset"}
```

## What You're Seeing

This dashboard visualizes **real-time spatial index CRI measurements**:

1. **Isolation Score (higher = better)**
   - Measures how isolated regions are from updates in other regions
   - R-tree: shared global tree → high CRI → low isolation
   - LCHI: partitioned trees → zero CRI → high isolation

2. **Cross-CRI (lower = better)**
   - Average interference across all region pairs
   - Captures severity of cross-region effects

3. **Query Latency**
   - Per-operation timing distribution
   - Shows impact of index structure on performance

## Comparison to Neural Network Dashboards

| Feature | NN Training | VLUI Dashboard |
|---------|-------------|----------------|
| **Metric 1** | Loss curve | Isolation score evolution |
| **Metric 2** | Accuracy | CRI (interference) |
| **Metric 3** | Learning rate | Query latency |
| **Update Rate** | Every epoch | Every 40 ops (~500ms) |
| **Real-time** | Yes (live training) | Yes (live benchmark) |
| **Comparison** | Train/Val curves | R-tree vs LCHI |

## File Structure

```
scripts/
├── web_dashboard.py           ← Backend (Flask server)
├── start_web_dashboard.py     ← Startup helper
├── templates/
│   └── index.html             ← Frontend (HTML5/CSS3/JS)
└── README_WEB.md              ← This file
```

## Troubleshooting

**"Address already in use"**
- Another process is using port 5000
- Kill it: `taskkill /IM python.exe` or change port in `web_dashboard.py`

**"No module named 'flask'"**
- Install: `pip install flask flask-cors`

**Charts not updating**
- Check browser console (F12) for JavaScript errors
- Ensure Flask server is running
- Check if `/api/metrics` endpoint returns data

**Benchmark hangs**
- Increase update interval in backend (`time.sleep(0.1)`)
- Check CPU usage (might be busy)

## Customization

Edit `web_dashboard.py`:

```python
# Change update interval (update_interval=40 → every 40 ops)
if (i + 1) % 40 == 0:
    ...

# Change number of ops
gen = ZipfSpatialGenerator(..., ..., ..., ..., seed=42)
trace = gen.generate()  # Change n_points, n_queries

# Change workload parameters
alpha=1.5  # Zipf exponent (higher = more skewed)
grid_dims=(2, 2)  # Number of regions (4 total)
```

## License

Part of VLUI research project (2026).
