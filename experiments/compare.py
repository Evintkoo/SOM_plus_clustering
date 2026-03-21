"""
Compare Python vs Rust experiment results side by side.

Usage:
    python experiments/compare.py

Requires both results files to exist:
    experiments/results/python_results.json
    experiments/results/rust_results.json
"""
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def load(name):
    path = os.path.join(RESULTS_DIR, f"{name}_results.json")
    with open(path) as f:
        return json.load(f)

METRICS = [
    ("silhouette",        "Silhouette ↑",          6),
    ("davies_bouldin",    "Davies-Bouldin ↓",       6),
    ("calinski_harabasz", "Calinski-Harabasz ↑",    2),
    ("dunn",              "Dunn ↑",                 6),
    ("fit_time_s",        "Fit time (s)",            4),
    ("predict_time_s",    "Predict time (s)",        4),
]

def fmt(val, decimals):
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)

def speedup(py_t, rs_t):
    try:
        return f"{float(py_t)/float(rs_t):.1f}x"
    except (TypeError, ValueError, ZeroDivisionError):
        return "N/A"

py = load("python")
rs = load("rust")

datasets = list(py.keys())
algos    = ["som", "kmeans"]

for algo in algos:
    title = "SOM" if algo == "som" else "KMeans"
    print(f"\n{'='*90}")
    print(f"  {title}  —  Python vs Rust")
    print(f"{'='*90}")

    # Header
    col_w = 14
    header = f"{'Dataset':<16} {'Metric':<26}"
    header += f"{'Python':>{col_w}} {'Rust':>{col_w}} {'Speedup':>{col_w}}"
    print(header)
    print("-" * 90)

    for ds in datasets:
        py_algo = py.get(ds, {}).get(algo, {})
        rs_algo = rs.get(ds, {}).get(algo, {})
        first = True
        for (key, label, dec) in METRICS:
            py_val = py_algo.get(key)
            rs_val = rs_algo.get(key)
            ds_label = ds if first else ""
            first = False

            sp = ""
            if "time" in key:
                sp = speedup(py_val, rs_val)

            print(
                f"{ds_label:<16} {label:<26}"
                f"{fmt(py_val, dec):>{col_w}} "
                f"{fmt(rs_val, dec):>{col_w}} "
                f"{sp:>{col_w}}"
            )
        print()

# Overall speedup summary
print(f"\n{'='*90}")
print("  Fit-time speedup summary (Python / Rust, higher = Rust faster)")
print(f"{'='*90}")
print(f"{'Dataset':<20} {'SOM speedup':>15} {'KMeans speedup':>16}")
print("-" * 55)
for ds in datasets:
    som_sp = speedup(
        py.get(ds, {}).get("som", {}).get("fit_time_s"),
        rs.get(ds, {}).get("som", {}).get("fit_time_s"),
    )
    km_sp = speedup(
        py.get(ds, {}).get("kmeans", {}).get("fit_time_s"),
        rs.get(ds, {}).get("kmeans", {}).get("fit_time_s"),
    )
    print(f"{ds:<20} {som_sp:>15} {km_sp:>16}")
