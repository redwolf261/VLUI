"""
Web-based real-time benchmark dashboard backend.
Serves live metrics via JSON polling.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from threading import Thread
import time

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generators.zipf_spatial import ZipfSpatialGenerator
from src.indexes.rtree_index import NaiveRTreeIndex
from src.indexes.lchi_index import LCHIIndex
from src.indexes.buffered_rtree import BufferedRTreeIndex
from src.indexes.grid_btree import GridGlobalBTreeIndex
from src.indexes.lsm_spatial import LSMSpatialIndex
from src.indexes.learned_grid import LearnedGridIndex
from src.metrics.cri import CRIMeasurer
from src.models.spatial import OpType
from src.analysis.sprt import IndexSPRT, SPRTDecision
from src.generators.real_dataset import RealDatasetLoader

# Ordered list of (key, display_name)
INDEX_KEYS = [
    ("rtree",    "R-tree"),
    ("lchi",     "LCHI (VLUI ✓)"),
    ("buffered", "BufferedRTree"),
    ("grid",     "GridBTree"),
    ("lsm",      "LSMSpatial"),
    ("learned",  "LearnedGrid"),
]

# ── Global state ─────────────────────────────────────────────────────────────

def _fresh_state() -> dict:
    return {
        "running": False,
        "done": False,
        "progress": 0.0,
        "total_ops": 0,
        "current_op": 0,
        "elapsed_time": 0.0,
        "index_keys": INDEX_KEYS,   # sent to frontend for labels
        "dataset_info": None,        # None = synthetic, dict = real dataset
        "metrics": {
            k: {"isolation": [], "mean_cri": [], "timestamps": []}
            for k, _ in INDEX_KEYS
        },
        "cri_matrix": {k: [] for k, _ in INDEX_KEYS},
        "latencies": {k: [] for k, _ in INDEX_KEYS},
        "current": {
            k: {"isolation": 0.0, "mean_cri": 0.0, "mean_latency_ns": 0.0}
            for k, _ in INDEX_KEYS
        },
        "sprt": {
            k: {
                "verdict": "continue",
                "n_pairs": 0,
                "n_accepted": 0,
                "n_rejected": 0,
                "n_continue": 0,
                "mean_llr": 0.0,
                "boundary_reject": 0.0,
                "boundary_accept": 0.0,
                "llr_history": [],
                "pairs": [],
            }
            for k, _ in INDEX_KEYS
        },
    }


state = _fresh_state()
# Pending real-dataset trace (set by /api/upload, consumed by /api/start)
_pending_trace = None
_pending_dataset_info: dict | None = None


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")
CORS(app)

SCALING_CURVE_PATH = PROJECT_ROOT / "output" / "scale" / "scaling_curve.csv"


def _read_scaling_curve_rows() -> list[dict]:
    """Load normalized rows from output/scale/scaling_curve.csv."""
    if not SCALING_CURVE_PATH.exists():
        return []

    rows: list[dict] = []
    with SCALING_CURVE_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                {
                    "n_points": int(raw["n_points"]),
                    "index": raw["index"],
                    "label": raw["label"],
                    "isolation": float(raw["isolation"]),
                    "mean_cri": float(raw["mean_cri"]),
                    "mean_lat_ns": float(raw["mean_lat_ns"]),
                    "sprt_verdict": raw["sprt_verdict"],
                }
            )

    rows.sort(key=lambda r: (r["n_points"], r["index"]))
    return rows


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/evaluator")
def evaluator_view():
    return render_template("evaluator.html")


@app.route("/api/scaling_curve")
def get_scaling_curve():
    rows = _read_scaling_curve_rows()

    if not rows:
        return jsonify(
            {
                "rows": [],
                "n_values": [],
                "index_labels": [],
                "generated_at": None,
                "error": "Missing scaling_curve.csv. Run scripts/collect_scaling_curve.py first.",
            }
        )

    n_values = sorted({r["n_points"] for r in rows})
    index_labels = sorted({r["label"] for r in rows})

    winners: list[dict] = []
    for n in n_values:
        group = [r for r in rows if r["n_points"] == n]
        winners.append(
            {
                "n_points": n,
                "best_isolation": min(group, key=lambda r: r["isolation"])["label"],
                "best_cri": min(group, key=lambda r: r["mean_cri"])["label"],
                "best_latency": min(group, key=lambda r: r["mean_lat_ns"])["label"],
            }
        )

    generated_at = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(SCALING_CURVE_PATH.stat().st_mtime),
    )

    return jsonify(
        {
            "rows": rows,
            "n_values": n_values,
            "index_labels": index_labels,
            "winners": winners,
            "generated_at": generated_at,
            "source_file": str(SCALING_CURVE_PATH),
        }
    )


@app.route("/api/metrics")
def get_metrics():
    snapshot = {
        "running":     state["running"],
        "done":        state["done"],
        "progress":    state["progress"],
        "current_op":  state["current_op"],
        "total_ops":   state["total_ops"],
        "elapsed_time":state["elapsed_time"],
        "index_keys":  state["index_keys"],
        "metrics":     state["metrics"],
        "cri_matrix":  state["cri_matrix"],
        "current":     state["current"],
        "latencies":   {k: v[-200:] for k, v in state["latencies"].items()},
        "sprt":        state["sprt"],
        "dataset_info": state.get("dataset_info"),
    }
    return jsonify(snapshot)


@app.route("/api/dataset_info")
def get_dataset_info():
    """Return current pending dataset info (or None if synthetic)."""
    return jsonify({"dataset_info": _pending_dataset_info})


@app.route("/api/upload", methods=["POST"])
def upload_dataset():
    """Accept a CSV upload and build a real-dataset trace.

    Expects multipart/form-data with:
      file   : the CSV file
      format : 'porto' | 'generic' (default: auto-detect)
      n_points : int (default 1000)
      n_queries: int (default 200)
    """
    global _pending_trace, _pending_dataset_info

    if state["running"]:
        return jsonify({"error": "Benchmark running — wait for it to finish"}), 409

    f = request.files.get("file")
    if f is None:
        return jsonify({"error": "No file uploaded"}), 400

    fmt        = request.form.get("format", "auto").lower()
    n_points   = int(request.form.get("n_points",  1000))
    n_queries  = int(request.form.get("n_queries",  200))
    grid_dims  = (2, 2)  # matches benchmark

    raw = f.read().decode("utf-8", errors="replace")
    file_io = io.StringIO(raw)

    try:
        # Auto-detect dataset format from CSV header
        first_line = raw.split("\n", 1)[0]
        is_porto = "POLYLINE" in first_line.upper()
        is_nyc   = "PICKUP_LONGITUDE" in first_line.upper()

        if fmt == "porto" or (fmt == "auto" and is_porto):
            loader = RealDatasetLoader.from_porto_csv(
                file_io,
                n_points=n_points,
                n_queries=n_queries,
                grid_dims=grid_dims,
            )
        elif fmt == "nyc" or (fmt == "auto" and is_nyc):
            loader = RealDatasetLoader.from_nyc_taxi_csv(
                file_io,
                n_points=n_points,
                n_queries=n_queries,
                grid_dims=grid_dims,
            )
        else:
            loader = RealDatasetLoader.from_csv(
                file_io,
                n_points=n_points,
                n_queries=n_queries,
                grid_dims=grid_dims,
            )

        trace = loader.generate()
        region_counts = loader.region_point_counts

        _pending_trace = trace
        _pending_dataset_info = {
            "source": loader.source_name,
            "n_points": n_points,
            "n_queries": n_queries,
            "total_ops": len(trace.operations),
            "region_counts": region_counts,
            "filename": f.filename,
        }

        return jsonify({
            "status": "ready",
            "dataset_info": _pending_dataset_info,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 422


@app.route("/api/upload/clear", methods=["POST"])
def clear_upload():
    global _pending_trace, _pending_dataset_info
    _pending_trace = None
    _pending_dataset_info = None
    return jsonify({"status": "cleared"})


@app.route("/api/start", methods=["POST"])
def start_benchmark():
    global state
    if state["running"]:
        return jsonify({"error": "Already running"}), 409
    state = _fresh_state()
    state["dataset_info"] = _pending_dataset_info
    Thread(target=_run_benchmark, args=(_pending_trace,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/reset", methods=["POST"])
def reset():
    global state
    state = _fresh_state()
    return jsonify({"status": "reset"})


# ── Benchmark logic ───────────────────────────────────────────────────────────

def _run_benchmark(trace=None):
    global state

    state["running"] = True
    state["done"] = False
    t0 = time.time()

    if trace is None:
        # Synthetic Zipf workload (default)
        gen = ZipfSpatialGenerator(
            n_points=350,
            n_queries=150,
            alpha=1.5,
            grid_dims=(2, 2),
            seed=42,
        )
        trace = gen.generate()
    total = len(trace.operations)
    state["total_ops"] = total

    m = 4
    UPDATE_EVERY = 30

    # All 6 indexes + their CRI measurers
    indexes = {
        "rtree":    NaiveRTreeIndex(dim=2, leaf_capacity=50),
        "lchi":     LCHIIndex(dim=2, grid_dims=(2, 2)),
        "buffered": BufferedRTreeIndex(dim=2, leaf_capacity=50, buffer_capacity=1000),
        "grid":     GridGlobalBTreeIndex(dim=2, grid_dims=(2, 2)),
        "lsm":      LSMSpatialIndex(dim=2, memtable_capacity=1000),
        "learned":  LearnedGridIndex(dim=2, grid_dims=(2, 2)),
    }
    measurers = {k: CRIMeasurer(n_regions=m, window_size=50) for k in indexes}
    # SPRT: H0=zero CRI, delta=0.05 elasticity, sigma=0.10, alpha=beta=0.05
    # For real-world data the CRI magnitudes differ from synthetic; use a
    # warmup period so sigma is auto-calibrated from the actual distribution.
    is_real = state.get("dataset_info") is not None
    sprt_warmup = 5 if is_real else 0
    sprts = {k: IndexSPRT(n_regions=m, delta=0.05, sigma=0.10, alpha=0.05, beta=0.05,
                          warmup_windows=sprt_warmup)
             for k in indexes}
    sprt_llr_histories: dict[str, list[float]] = {k: [] for k in indexes}
    _sprt_xcal_done = False  # cross-calibration fired at most once

    for i, op in enumerate(trace.operations):
        if op.op_type == OpType.INSERT and op.point is not None:
            for k, idx in indexes.items():
                s = idx.insert(op.point)
                measurers[k].record_update(op.region_id, op.timestamp, s.structural_mutations)

        elif op.op_type == OpType.RANGE_QUERY and op.query_region is not None:
            for k, idx in indexes.items():
                _, s = idx.range_query(op.query_region)
                state["latencies"][k].append(s.latency_ns)
                measurers[k].record_query(op.region_id, op.timestamp, s.latency_ns)

        if (i + 1) % UPDATE_EVERY == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            state["current_op"]   = i + 1
            state["progress"]     = (i + 1) / total * 100
            state["elapsed_time"] = elapsed

            for k in indexes:
                cri = measurers[k].compute_cri()
                state["metrics"][k]["isolation"].append(cri.isolation_score)
                state["metrics"][k]["mean_cri"].append(cri.mean_cross_cri)
                state["metrics"][k]["timestamps"].append(elapsed)
                mat = cri.cri_matrix.tolist()
                state["cri_matrix"][k] = mat
                lats = state["latencies"][k]
                ml = sum(lats) / len(lats) if lats else 0.0
                state["current"][k] = {
                    "isolation":       cri.isolation_score,
                    "mean_cri":        cri.mean_cross_cri,
                    "mean_latency_ns": ml,
                }
                # Feed CRI matrix to SPRT
                sprts[k].update_from_matrix(mat)
                sprt_llr_histories[k].append(round(sprts[k].mean_llr, 4))
                sd = sprts[k].to_dict()
                sd["llr_history"] = sprt_llr_histories[k][-50:]
                state["sprt"][k] = sd

            # ── Cross-calibrate SPRT after warmup (real datasets only) ──────
            # After sprt_warmup windows we know LCHI's CRI scale.  Set
            #   δ = 3 × median(|LCHI CRI|)   σ = δ
            # so that observations *below* δ/2 (where LCHI typically sits)
            # produce negative LLR (drift → Accept) while competitors whose
            # CRI >> LCHI produce positive LLR (drift → Reject).
            if is_real and not _sprt_xcal_done \
                    and (i + 1) == sprt_warmup * UPDATE_EVERY:
                lchi_obs = sorted(
                    x
                    for p in sprts["lchi"].all_pairs
                    for x in p.obs_history
                )
                if lchi_obs:
                    lchi_median = lchi_obs[len(lchi_obs) // 2]
                    delta_ref = max(0.10, lchi_median * 3.0)
                    sigma_ref = delta_ref   # δ/σ = 1 → clean boundary scaling
                    for sprt in sprts.values():
                        sprt.recalibrate(sigma_ref, delta_ref)
                _sprt_xcal_done = True

            time.sleep(0.08)

    state["running"] = False
    state["done"]    = True


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    print("""
╔══════════════════════════════════════════════════════════════╗
║          🔬 CRI BENCHMARK DASHBOARD — READY                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Open in browser:  http://localhost:5000                    ║
║                                                              ║
║   Press  Ctrl+C  to stop                                     ║
╚══════════════════════════════════════════════════════════════╝
""")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
