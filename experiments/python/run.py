"""
Python experiment runner: SOM + KMeans on standard benchmark datasets.
Outputs results to experiments/results/python_results.json.

Usage:
    cd /path/to/SOM_plus_clustering
    python experiments/python/run.py
"""
import sys, os, time, json
import numpy as np
import pandas as pd

# Allow importing local modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from modules.som import SOM
from modules.kmeans import KMeans

# ---------------------------------------------------------------------------
# Dataset configs: (csv_name, n_classes, som_grid_m, som_grid_n)
# ---------------------------------------------------------------------------
DATASETS = [
    ("iris",          3,  5, 5),
    ("wine",          3,  5, 5),
    ("breast_cancer", 2,  6, 6),
    ("digits",       10,  8, 8),
    ("blobs",         5,  7, 7),
]

SOM_EPOCHS       = 10
SOM_LR           = 0.5
SOM_RADIUS       = 3
SOM_INIT         = "random"
SOM_DIST         = "euclidean"
KMEANS_MAX_ITERS = 300
KMEANS_INIT      = "kmeans++"

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------

def load_dataset(name):
    path = os.path.join(DATASETS_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values.astype(np.float64)
    y = df["label"].values.astype(int)
    return X, y

def run_som(X, n_classes, m, n):
    dim = X.shape[1]
    som = SOM(
        m=m, n=n, dim=dim,
        initiate_method=SOM_INIT,
        learning_rate=SOM_LR,
        neighbour_rad=SOM_RADIUS,
        distance_function=SOM_DIST,
        backend="numpy",
    )

    t0 = time.perf_counter()
    som.fit(X, epoch=SOM_EPOCHS, shuffle=True, batch_size=None)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels = som.predict(X)
    predict_time = time.perf_counter() - t0

    scores = som.evaluate(X, ["all"])
    return {
        "fit_time_s":          round(fit_time, 4),
        "predict_time_s":      round(predict_time, 4),
        "silhouette":          round(float(scores["silhouette"]), 6),
        "davies_bouldin":      round(float(scores["davies_bouldin"]), 6),
        "calinski_harabasz":   round(float(scores["calinski_harabasz"]), 4),
        "dunn":                round(float(scores["dunn"]), 6),
        "n_neurons":           m * n,
    }

def run_kmeans(X, n_classes):
    km = KMeans(n_clusters=n_classes, method=KMEANS_INIT, max_iters=KMEANS_MAX_ITERS)

    t0 = time.perf_counter()
    km.fit(X)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels = km.predict(X)
    predict_time = time.perf_counter() - t0

    # Use the module evals directly
    from modules.evals import (
        silhouette_score, davies_bouldin_index,
        calinski_harabasz_score, dunn_index,
    )
    return {
        "fit_time_s":        round(fit_time, 4),
        "predict_time_s":    round(predict_time, 4),
        "silhouette":        round(float(silhouette_score(x=X, labels=labels)), 6),
        "davies_bouldin":    round(float(davies_bouldin_index(x=X, labels=labels)), 6),
        "calinski_harabasz": round(float(calinski_harabasz_score(x=X, labels=labels)), 4),
        "dunn":              round(float(dunn_index(x=X, labels=labels)), 6),
        "inertia":           round(float(km.inertia_), 4),
    }

# ---------------------------------------------------------------------------
results = {}

for (name, n_classes, m, n) in DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  (k={n_classes}, grid={m}x{n})")
    X, y = load_dataset(name)
    print(f"  Shape: {X.shape}")

    print("  Running SOM ...", end=" ", flush=True)
    try:
        som_res = run_som(X, n_classes, m, n)
        print(f"fit={som_res['fit_time_s']:.3f}s  sil={som_res['silhouette']:.4f}")
    except Exception as e:
        som_res = {"error": str(e)}
        print(f"ERROR: {e}")

    print("  Running KMeans ...", end=" ", flush=True)
    try:
        km_res = run_kmeans(X, n_classes)
        print(f"fit={km_res['fit_time_s']:.3f}s  sil={km_res['silhouette']:.4f}")
    except Exception as e:
        km_res = {"error": str(e)}
        print(f"ERROR: {e}")

    results[name] = {
        "n_samples":  int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes":  n_classes,
        "som":        som_res,
        "kmeans":     km_res,
    }

out_path = os.path.join(RESULTS_DIR, "python_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {out_path}")
