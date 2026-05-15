"""
External metric evaluation: reads Rust predicted labels + ground truth CSVs,
computes ARI, NMI, FMI, V-measure using sklearn, and outputs the full report.

Run after benchmark_quality Rust binary:
    python experiments/benchmark/evaluate.py
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    fowlkes_mallows_score, v_measure_score,
)

BASE      = os.path.dirname(__file__)
DS_DIR    = os.path.join(BASE, "datasets")
RES_DIR   = os.path.join(BASE, "results")
CFG_PATH  = os.path.join(BASE, "dataset_config.json")

# ---------------------------------------------------------------------------

def load_ground_truth(name):
    df = pd.read_csv(os.path.join(DS_DIR, f"{name}.csv"))
    return df["label"].values.astype(int)

def load_labels(name, algo):
    path = os.path.join(RES_DIR, f"{name}_{algo}_labels.csv")
    return pd.read_csv(path, header=None).values.flatten().astype(int)

def ext_metrics(y_true, y_pred):
    return {
        "ari":  round(adjusted_rand_score(y_true, y_pred), 4),
        "nmi":  round(normalized_mutual_info_score(y_true, y_pred), 4),
        "fmi":  round(fowlkes_mallows_score(y_true, y_pred), 4),
        "v_measure": round(v_measure_score(y_true, y_pred), 4),
    }

# Load internal metrics from Rust
with open(os.path.join(RES_DIR, "rust_metrics.json")) as f:
    rust = json.load(f)

with open(CFG_PATH) as f:
    configs = {c["name"]: c for c in json.load(f)}

# Build full results
full = {}
for name, cfg in configs.items():
    if name not in rust:
        continue
    y_true = load_ground_truth(name)
    r = rust[name]

    som_labels    = load_labels(name, "som")
    km_labels     = load_labels(name, "km")
    densom_labels = load_labels(name, "densom")
    auto_labels   = load_labels(name, "auto")

    # DenSOM / AutoSOM labels include -1 for noise.  sklearn metrics treat -1
    # as its own cluster ID, so ARI/NMI still reflect cluster recovery quality.

    # GPU labels (optional — only present when Metal feature was enabled at build time)
    gpu_path = os.path.join(RES_DIR, f"{name}_som_gpu_labels.csv")
    if os.path.exists(gpu_path):
        try:
            gpu_labels = pd.read_csv(gpu_path, header=None).values.flatten().astype(int)
            som_gpu_ext = ext_metrics(y_true, gpu_labels)
        except Exception as e:
            print(f"  warn: GPU labels for {name} unreadable: {e}")
            som_gpu_ext = {"ari": None, "nmi": None, "fmi": None, "v_measure": None}
    else:
        som_gpu_ext = {"ari": None, "nmi": None, "fmi": None, "v_measure": None}

    som_gpu_internal = r.get("som_gpu", {})

    full[name] = {
        "n_samples":       r["n_samples"],
        "n_features":      r["n_features"],
        "n_true_clusters": r["n_true_clusters"],
        "som_grid":        r["som_grid"],
        "som":     {**r["som"],     **ext_metrics(y_true, som_labels)},
        "kmeans":  {**r["kmeans"],  **ext_metrics(y_true, km_labels)},
        "densom":  {**r["densom"],  **ext_metrics(y_true, densom_labels)},
        "autosom": {**r["autosom"], **ext_metrics(y_true, auto_labels)},
        "som_gpu": {**som_gpu_internal, **som_gpu_ext},
    }

out_path = os.path.join(RES_DIR, "full_results.json")
with open(out_path, "w") as f:
    json.dump(full, f, indent=2)
print(f"Full results → {out_path}")
