"""
DenSOM vs SOM vs KMeans benchmark comparison.

Run after:
    cargo run --example benchmark_quality --release
    python experiments/benchmark/evaluate.py

Then:
    python experiments/benchmark/report_densom.py
"""
import os, json
import numpy as np

BASE    = os.path.dirname(__file__)
RES_DIR = os.path.join(BASE, "results")

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

def v(d, key, decimals=4):
    val = d.get(key)
    if val is None:
        return "  N/A  "
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)

def delta(base, new, decimals=4):
    try:
        d = float(new) - float(base)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.{decimals}f}"
    except Exception:
        return "  N/A  "

W = 8

# ---------------------------------------------------------------------------
# Table 1: ARI comparison — SOM / KMeans / DenSOM / AutoSOM / deltas
# ---------------------------------------------------------------------------
print("\n" + "="*130)
print("  AUTOSOM PIPELINE vs SOM vs KMEANS vs DENSOM  —  ARI (Adjusted Rand Index, higher = better)")
print("="*130)
print(f"  {'Dataset':<22}  {'k_true':>6}  {'SOM':>{W}}  {'KMeans':>{W}}  {'DenSOM':>{W}}  "
      f"{'AutoSOM':>{W}}  {'algo':>6}  {'k_det':>5}  {'Δ(Auto-KM)':>10}  {'noise%':>6}  {'fit(s)':>7}")
print("  " + "─"*115)

for cat, names in CATEGORIES.items():
    present = [n for n in names if n in data]
    if not present:
        continue
    print(f"\n  [{cat}]")
    som_aris, km_aris, ds_aris, auto_aris = [], [], [], []
    for name in present:
        d = data[name]
        k_true = d["n_true_clusters"]
        s   = d.get("som",     {})
        km  = d.get("kmeans",  {})
        ds  = d.get("densom",  {})
        au  = d.get("autosom", {})

        s_ari   = s.get("ari")
        km_ari  = km.get("ari")
        ds_ari  = ds.get("ari")
        au_ari  = au.get("ari")
        algo    = au.get("algorithm", "?")[:6]
        k_det   = au.get("k_detected", "?")
        nr      = au.get("noise_ratio")
        noise_pct = f"{nr*100:.1f}" if nr is not None else "N/A"
        fit_s = au.get("fit_time_s")
        fit_str = f"{fit_s:.4f}" if fit_s is not None else "N/A"

        d_auto_km = delta(km_ari, au_ari) if km_ari is not None and au_ari is not None else "  N/A  "

        print(f"  {name:<22}  {k_true:>6}  {v(s,'ari'):>{W}}  {v(km,'ari'):>{W}}  "
              f"{v(ds,'ari'):>{W}}  {v(au,'ari'):>{W}}  {algo:>6}  {str(k_det):>5}  "
              f"{d_auto_km:>10}  {noise_pct:>6}  {fit_str:>7}")

        if s_ari  is not None: som_aris.append(s_ari)
        if km_ari is not None: km_aris.append(km_ari)
        if ds_ari is not None: ds_aris.append(ds_ari)
        if au_ari is not None: auto_aris.append(au_ari)

    def ms(lst): return f"{np.mean(lst):.4f}" if lst else " N/A"
    print(f"  {'  Mean':22}  {'':>6}  {ms(som_aris):>{W}}  {ms(km_aris):>{W}}  "
          f"{ms(ds_aris):>{W}}  {ms(auto_aris):>{W}}")

# ---------------------------------------------------------------------------
# Table 2: NMI comparison
# ---------------------------------------------------------------------------
print("\n\n" + "="*110)
print("  NMI (Normalized Mutual Information, higher = better)")
print("="*110)
print(f"  {'Dataset':<22}  {'SOM':>{W}}  {'KMeans':>{W}}  {'DenSOM':>{W}}  {'AutoSOM':>{W}}  {'Δ(Auto-KM)':>10}")
print("  " + "─"*85)

for cat, names in CATEGORIES.items():
    present = [n for n in names if n in data]
    if not present:
        continue
    print(f"\n  [{cat}]")
    for name in present:
        d  = data[name]
        s  = d.get("som",     {})
        km = d.get("kmeans",  {})
        ds = d.get("densom",  {})
        au = d.get("autosom", {})
        d_str = delta(km.get("nmi"), au.get("nmi"))
        print(f"  {name:<22}  {v(s,'nmi'):>{W}}  {v(km,'nmi'):>{W}}  {v(ds,'nmi'):>{W}}  "
              f"{v(au,'nmi'):>{W}}  {d_str:>10}")

# ---------------------------------------------------------------------------
# Table 3: Auto-k detection accuracy — k_detected vs k_true, algo chosen
# ---------------------------------------------------------------------------
print("\n\n" + "="*100)
print("  AUTO-K DETECTION  —  k_detected (from DenSOM peaks) vs true k  +  algorithm chosen")
print("="*100)
print(f"  {'Dataset':<22}  {'k_true':>6}  {'k_det':>5}  {'match':>5}  {'algo':>6}  "
      f"{'noise%':>6}  {'DenSOM found':>12}  {'AutoSOM ARI':>11}  {'KMeans ARI':>10}")
print("  " + "─"*90)

for cat, names in CATEGORIES.items():
    present = [n for n in names if n in data]
    if not present:
        continue
    print(f"\n  [{cat}]")
    for name in present:
        d       = data[name]
        k_true  = d["n_true_clusters"]
        ds      = d.get("densom",  {})
        au      = d.get("autosom", {})
        km      = d.get("kmeans",  {})
        k_det   = au.get("k_detected", "?")
        algo    = au.get("algorithm", "?")[:6]
        nr      = au.get("noise_ratio")
        noise_pct = f"{nr*100:.1f}%" if nr is not None else "N/A"
        ds_found  = ds.get("n_clusters_found", "?")
        match = "✓" if k_det == k_true else ("~" if isinstance(k_det,int) and abs(k_det-k_true)<=1 else "✗")
        print(f"  {name:<22}  {k_true:>6}  {str(k_det):>5}  {match:>5}  {algo:>6}  "
              f"{noise_pct:>6}  {str(ds_found):>12}  {v(au,'ari'):>11}  {v(km,'ari'):>10}")

# ---------------------------------------------------------------------------
# Table 4: Fit-time comparison (SOM vs DenSOM — same grid, same epochs)
# ---------------------------------------------------------------------------
print("\n\n" + "="*90)
print("  FIT TIME — SOM vs DenSOM (same grid + epochs; DenSOM adds density pipeline)")
print("="*90)
print(f"  {'Dataset':<22}  {'SOM fit(s)':>12}  {'DenSOM fit(s)':>14}  {'overhead':>10}")
print("  " + "─"*65)

for cat, names in CATEGORIES.items():
    present = [n for n in names if n in data]
    if not present:
        continue
    print(f"\n  [{cat}]")
    for name in present:
        d  = data[name]
        s  = d.get("som",    {})
        ds = d.get("densom", {})
        s_t  = s.get("fit_time_s")
        ds_t = ds.get("fit_time_s")
        if s_t and ds_t:
            overhead = f"{(ds_t-s_t)/s_t*100:+.1f}%"
        else:
            overhead = "N/A"
        print(f"  {name:<22}  {v(s,'fit_time_s',4):>12}  {v(ds,'fit_time_s',4):>14}  {overhead:>10}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n\n" + "="*105)
print("  SUMMARY — Mean ARI per category")
print("="*105)
print(f"  {'Category':<45}  {'SOM':>8}  {'KMeans':>8}  {'DenSOM':>8}  {'AutoSOM':>8}  {'Δ(Auto-KM)':>10}")
print("  " + "─"*98)

for cat, names in CATEGORIES.items():
    present  = [n for n in names if n in data]
    s_list   = [data[n]["som"]["ari"]     for n in present if "som"     in data[n] and "ari" in data[n]["som"]]
    km_list  = [data[n]["kmeans"]["ari"]  for n in present if "kmeans"  in data[n] and "ari" in data[n]["kmeans"]]
    ds_list  = [data[n]["densom"]["ari"]  for n in present if "densom"  in data[n] and "ari" in data[n]["densom"]]
    au_list  = [data[n]["autosom"]["ari"] for n in present if "autosom" in data[n] and "ari" in data[n]["autosom"]]
    def ms(lst): return f"{np.mean(lst):.4f}" if lst else "  N/A"
    d_str = f"{np.mean(au_list)-np.mean(km_list):+.4f}" if km_list and au_list else "  N/A"
    print(f"  {cat:<45}  {ms(s_list):>8}  {ms(km_list):>8}  {ms(ds_list):>8}  {ms(au_list):>8}  {d_str:>10}")
