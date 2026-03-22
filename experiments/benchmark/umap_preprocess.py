"""
UMAP preprocessing: reduces every benchmark dataset to 2-D UMAP embedding,
then saves to umap_datasets/ with an updated dataset_config_umap.json.

2-D datasets are also passed through UMAP (denoising + topology normalisation).

Run:
    python experiments/benchmark/umap_preprocess.py
"""
#!/Users/evintleovonzko/.local/bin/python3.12
import os, json, sys
import numpy as np
import pandas as pd
from umap import UMAP

BASE     = os.path.dirname(__file__)
DS_DIR   = os.path.join(BASE, "datasets")
OUT_DIR  = os.path.join(BASE, "umap_datasets")
CFG_IN   = os.path.join(BASE, "dataset_config.json")
CFG_OUT  = os.path.join(BASE, "dataset_config_umap.json")

os.makedirs(OUT_DIR, exist_ok=True)

with open(CFG_IN) as f:
    configs = json.load(f)

umap_configs = []

for cfg in configs:
    name = cfg["name"]
    print(f"  {name:<28} loading...", end="", flush=True)
    df   = pd.read_csv(os.path.join(DS_DIR, f"{name}.csv"))
    y    = df["label"].values
    X    = df.drop(columns=["label"]).values
    n, d = X.shape

    # choose n_neighbors sensibly for small datasets
    n_neighbors = min(15, max(5, n // 20))
    # low_memory for large N to avoid excessive RAM usage
    low_mem = n > 10_000
    print(f" UMAP (n={n}, d={d}, nn={n_neighbors}, low_mem={low_mem})...", end="", flush=True)
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
        low_memory=low_mem,
    )
    X_umap = reducer.fit_transform(X)

    out_df = pd.DataFrame(X_umap, columns=["u1", "u2"])
    out_df["label"] = y.astype(int)
    out_df.to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False)

    umap_configs.append(cfg)   # keep same k / grid / epochs
    print(f" done", flush=True)

with open(CFG_OUT, "w") as f:
    json.dump(umap_configs, f, indent=2)

print(f"\n{len(umap_configs)} datasets written to {OUT_DIR}")
print(f"Config saved to {CFG_OUT}")
