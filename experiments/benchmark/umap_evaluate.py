"""
External metric evaluation for UMAP-preprocessed results.

Run after:
    cargo run --example benchmark_quality --release -- \\
        experiments/benchmark/umap_datasets \\
        experiments/benchmark/dataset_config_umap.json \\
        experiments/benchmark/umap_results

Then:
    python experiments/benchmark/umap_evaluate.py
"""
import os, json
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    fowlkes_mallows_score, v_measure_score,
)

BASE     = os.path.dirname(__file__)
DS_DIR   = os.path.join(BASE, "datasets")       # ground-truth labels from original datasets
RES_DIR  = os.path.join(BASE, "umap_results")
CFG_PATH = os.path.join(BASE, "dataset_config_umap.json")

def load_ground_truth(name):
    df = pd.read_csv(os.path.join(DS_DIR, f"{name}.csv"))
    return df["label"].values.astype(int)

def load_labels(name, algo):
    path = os.path.join(RES_DIR, f"{name}_{algo}_labels.csv")
    return pd.read_csv(path, header=None).values.flatten().astype(int)

def ext_metrics(y_true, y_pred):
    return {
        "ari":       round(adjusted_rand_score(y_true, y_pred), 4),
        "nmi":       round(normalized_mutual_info_score(y_true, y_pred), 4),
        "fmi":       round(fowlkes_mallows_score(y_true, y_pred), 4),
        "v_measure": round(v_measure_score(y_true, y_pred), 4),
    }

with open(os.path.join(RES_DIR, "rust_metrics.json")) as f:
    rust = json.load(f)

with open(CFG_PATH) as f:
    configs = {c["name"]: c for c in json.load(f)}

full = {}
for name, cfg in configs.items():
    if name not in rust:
        continue
    y_true     = load_ground_truth(name)
    r          = rust[name]
    som_labels = load_labels(name, "som")
    km_labels  = load_labels(name, "km")
    full[name] = {
        "n_samples":       r["n_samples"],
        "n_features":      r["n_features"],
        "n_true_clusters": r["n_true_clusters"],
        "som_grid":        r["som_grid"],
        "som":    {**r["som"],    **ext_metrics(y_true, som_labels)},
        "kmeans": {**r["kmeans"], **ext_metrics(y_true, km_labels)},
    }

out_path = os.path.join(RES_DIR, "umap_full_results.json")
with open(out_path, "w") as f:
    json.dump(full, f, indent=2)
print(f"UMAP results -> {out_path}")
