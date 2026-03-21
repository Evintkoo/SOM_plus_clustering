"""
Print the full benchmark report from full_results.json.

Run:
    python experiments/benchmark/report.py
"""
import os, json
import numpy as np

BASE      = os.path.dirname(__file__)
RES_DIR   = os.path.join(BASE, "results")

with open(os.path.join(RES_DIR, "full_results.json")) as f:
    data = json.load(f)

CATEGORIES = {
    "S-sets (SIPU Gaussian mixtures)": ["s1","s2","s3","s4"],
    "A-sets (SIPU varying k)":         ["a1","a2","a3"],
    "Shape stress tests":              ["moons","circles","spiral","anisotropic","varied_density"],
    "Real-world UCI":                  ["iris","wine","breast_cancer","digits"],
    "Scalability (2-D, k=5)":         ["scale_1k","scale_5k","scale_10k","scale_50k"],
    "High-dimensional (N=1000, k=5)": ["dim_32","dim_64","dim_128","dim_256"],
}

EXT_COLS = [
    ("ari",       "ARI↑",        4),
    ("nmi",       "NMI↑",        4),
    ("fmi",       "FMI↑",        4),
    ("v_measure", "V-meas↑",     4),
]
INT_COLS = [
    ("silhouette",        "Silhouette↑",  4),
    ("davies_bouldin",    "DB↓",          4),
    ("calinski_harabasz", "CH↑",          1),
    ("dunn",              "Dunn↑",        4),
]
SPEED_COLS = [
    ("fit_time_s",     "Fit(s)",  4),
    ("predict_time_s", "Pred(s)", 6),
]

def fmt(d, key, dec):
    v = d.get(key)
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{dec}f}"
    except:
        return str(v)

W = 9  # column width

def header_row(cols):
    return "  ".join(f"{lbl:>{W}}" for _, lbl, _ in cols)

def data_row(d, cols):
    return "  ".join(f"{fmt(d, k, dec):>{W}}" for k, _, dec in cols)

def section(title, names, algo_label, cols):
    label_w = 20
    head = f"{'Dataset':<{label_w}}  {header_row(cols)}"
    print(f"\n  {'─'*len(head)}")
    print(f"  {algo_label}")
    print(f"  {'─'*len(head)}")
    print(f"  {head}")
    print(f"  {'─'*len(head)}")
    for name in names:
        if name not in data:
            continue
        d = data[name]
        print(f"  {name:<{label_w}}  {data_row(d[algo_label.split()[0].lower()], cols)}")


print("\n" + "="*110)
print("  RUST CLUSTERING BENCHMARK REPORT")
print("  SOM + KMeans across 23 datasets  |  Quality (ARI/NMI/FMI/V-meas) + Internal metrics + Speed")
print("="*110)

for cat, names in CATEGORIES.items():
    names_present = [n for n in names if n in data]
    if not names_present:
        continue

    print(f"\n\n{'#'*110}")
    print(f"  {cat.upper()}")
    print(f"{'#'*110}")

    # Info table
    label_w = 20
    print(f"\n  {'Dataset':<{label_w}}  {'N':>7}  {'D':>5}  {'k':>4}  {'SOM grid':>10}")
    print(f"  {'─'*55}")
    for name in names_present:
        d = data[name]
        print(f"  {name:<{label_w}}  {d['n_samples']:>7}  {d['n_features']:>5}  {d['n_true_clusters']:>4}  {d['som_grid']:>10}")

    for algo in ["som", "kmeans"]:
        algo_label = "SOM" if algo == "som" else "KMeans"

        # External metrics
        section(cat, names_present, f"{algo_label} External (vs ground truth)", EXT_COLS)

        # Internal metrics
        section(cat, names_present, f"{algo_label} Internal (no labels needed)", INT_COLS)

        # Speed
        section(cat, names_present, f"{algo_label} Speed", SPEED_COLS)

# ---------------------------------------------------------------------------
# Summary: mean ARI per category
# ---------------------------------------------------------------------------
print(f"\n\n{'='*110}")
print("  SUMMARY — Mean ARI per category (↑ better)")
print(f"{'='*110}")
print(f"  {'Category':<45}  {'SOM ARI':>10}  {'KMeans ARI':>12}")
print(f"  {'─'*70}")
for cat, names in CATEGORIES.items():
    som_aris = [data[n]["som"]["ari"] for n in names if n in data and "ari" in data[n]["som"]]
    km_aris  = [data[n]["kmeans"]["ari"] for n in names if n in data and "ari" in data[n]["kmeans"]]
    s_mean = f"{np.mean(som_aris):.4f}" if som_aris else "N/A"
    k_mean = f"{np.mean(km_aris):.4f}" if km_aris else "N/A"
    print(f"  {cat:<45}  {s_mean:>10}  {k_mean:>12}")

# Scalability fit-time table
print(f"\n\n{'='*110}")
print("  SCALABILITY — Fit time (seconds) vs N  (SOM 10×10, KMeans k=5)")
print(f"{'='*110}")
scale_names = ["scale_1k","scale_5k","scale_10k","scale_50k"]
print(f"  {'Dataset':<16}  {'N':>7}  {'SOM fit(s)':>12}  {'KM fit(s)':>12}")
print(f"  {'─'*55}")
for name in scale_names:
    if name not in data:
        continue
    d = data[name]
    print(f"  {name:<16}  {d['n_samples']:>7}  {fmt(d['som'],'fit_time_s',4):>12}  {fmt(d['kmeans'],'fit_time_s',4):>12}")

# High-dim fit-time table
print(f"\n\n{'='*110}")
print("  HIGH-DIMENSIONAL — Fit time (seconds) vs D  (SOM 8×8, N=1000, k=5)")
print(f"{'='*110}")
dim_names = ["dim_32","dim_64","dim_128","dim_256"]
print(f"  {'Dataset':<16}  {'D':>5}  {'SOM fit(s)':>12}  {'KM fit(s)':>12}")
print(f"  {'─'*50}")
for name in dim_names:
    if name not in data:
        continue
    d = data[name]
    print(f"  {name:<16}  {d['n_features']:>5}  {fmt(d['som'],'fit_time_s',4):>12}  {fmt(d['kmeans'],'fit_time_s',4):>12}")
