"""
Side-by-side comparison: Baseline vs UMAP-preprocessed clustering.

Prints ARI delta for every dataset and algo, plus category summary.

Run:
    python experiments/benchmark/compare_umap.py
"""
import os, json
import numpy as np

BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "results",      "full_results.json")) as f:
    base = json.load(f)

with open(os.path.join(BASE, "umap_results", "umap_full_results.json")) as f:
    umap = json.load(f)

CATEGORIES = {
    "S-sets (SIPU Gaussian mixtures)": ["s1","s2","s3","s4"],
    "A-sets (SIPU varying k)":         ["a1","a2","a3"],
    "Shape stress tests":              ["moons","circles","spiral","anisotropic","varied_density"],
    "Real-world UCI":                  ["iris","wine","breast_cancer","digits"],
    "Scalability (2-D, k=5)":         ["scale_1k","scale_5k","scale_10k","scale_50k"],
    "High-dimensional (N=1000, k=5)": ["dim_32","dim_64","dim_128","dim_256"],
}

def ari(data, name, algo):
    try:
        return data[name][algo]["ari"]
    except (KeyError, TypeError):
        return None

W = 8

print("\n" + "="*110)
print("  UMAP PREPROCESSING IMPACT — Baseline vs UMAP+Cluster  (ARI, higher=better)")
print("="*110)
print(f"  {'Dataset':<22}  {'SOM base':>{W}}  {'SOM umap':>{W}}  {'SOM Δ':>{W}}  |  "
      f"{'KM base':>{W}}  {'KM umap':>{W}}  {'KM Δ':>{W}}")
print("  " + "─"*90)

for cat, names in CATEGORIES.items():
    names_present = [n for n in names if n in base and n in umap]
    if not names_present:
        continue

    print(f"\n  [{cat}]")
    som_deltas, km_deltas = [], []

    for name in names_present:
        sb = ari(base, name, "som");  su = ari(umap, name, "som")
        kb = ari(base, name, "kmeans"); ku = ari(umap, name, "kmeans")

        def fmt(v): return f"{v:.4f}" if v is not None else "  N/A  "
        def dlt(a, b):
            if a is None or b is None: return "  N/A  "
            d = b - a
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.4f}"

        print(f"  {name:<22}  {fmt(sb):>{W}}  {fmt(su):>{W}}  {dlt(sb,su):>{W}}  |  "
              f"{fmt(kb):>{W}}  {fmt(ku):>{W}}  {dlt(kb,ku):>{W}}")

        if sb is not None and su is not None: som_deltas.append(su - sb)
        if kb is not None and ku is not None: km_deltas.append(ku - kb)

    sm = f"{np.mean(som_deltas):+.4f}" if som_deltas else " N/A"
    km = f"{np.mean(km_deltas):+.4f}"  if km_deltas  else " N/A"
    print(f"  {'  Mean Δ':<22}  {'':>{W}}  {'':>{W}}  {sm:>{W}}  |  {'':>{W}}  {'':>{W}}  {km:>{W}}")

# Overall summary
print("\n\n" + "="*110)
print("  SUMMARY — Mean ARI: Baseline vs UMAP  (across all datasets)")
print("="*110)
print(f"  {'Category':<45}  {'SOM base':>10}  {'SOM umap':>10}  {'SOM Δ':>8}"
      f"  |  {'KM base':>10}  {'KM umap':>10}  {'KM Δ':>8}")
print("  " + "─"*108)

for cat, names in CATEGORIES.items():
    sb_list = [ari(base, n, "som")    for n in names if n in base and n in umap and ari(base,n,"som") is not None]
    su_list = [ari(umap, n, "som")    for n in names if n in base and n in umap and ari(umap,n,"som") is not None]
    kb_list = [ari(base, n, "kmeans") for n in names if n in base and n in umap and ari(base,n,"kmeans") is not None]
    ku_list = [ari(umap, n, "kmeans") for n in names if n in base and n in umap and ari(umap,n,"kmeans") is not None]

    def ms(lst): return f"{np.mean(lst):.4f}" if lst else " N/A"
    def md(a,b): return f"{np.mean(b)-np.mean(a):+.4f}" if a and b else " N/A"

    print(f"  {cat:<45}  {ms(sb_list):>10}  {ms(su_list):>10}  {md(sb_list,su_list):>8}"
          f"  |  {ms(kb_list):>10}  {ms(ku_list):>10}  {md(kb_list,ku_list):>8}")
