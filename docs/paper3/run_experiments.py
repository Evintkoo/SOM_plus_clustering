#!/usr/bin/env python3
"""
Paper 3 experiment runner: multi-seed KMeans++, statistical tests, figures, V2 comparison.

Run from project root:
    python docs/paper3/run_experiments.py

Outputs:
    docs/paper3/figs/delta_ari.pdf
    docs/paper3/figs/algorithm_comparison.pdf
    docs/paper3/figs/friedman_nemenyi.pdf
    docs/paper3/multi_seed_results.json
    docs/paper3/statistical_tests.json
"""
import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path

# sklearn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
from sklearn.cluster import KMeans

# scipy for statistical tests
from scipy import stats

# matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DS_DIR = PROJECT_ROOT / "experiments" / "benchmark" / "datasets"
RES_DIR = PROJECT_ROOT / "experiments" / "benchmark" / "results"
CFG_PATH = PROJECT_ROOT / "experiments" / "benchmark" / "dataset_config.json"
FULL_RESULTS = RES_DIR / "full_results.json"
PAPER3_DIR = Path(__file__).resolve().parent
FIGS_DIR = PAPER3_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

N_SEEDS = 10  # number of KMeans++ runs per dataset
SEEDS = list(range(42, 42 + N_SEEDS))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_dataset(name):
    df = pd.read_csv(DS_DIR / f"{name}.csv")
    labels = df["label"].values.astype(int)
    features = df.drop(columns=["label"]).values
    return features, labels

def load_config():
    with open(CFG_PATH) as f:
        return json.load(f)

def load_full_results():
    with open(FULL_RESULTS) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Multi-seed KMeans++ experiment
# ---------------------------------------------------------------------------
def run_multiseed_kmeans(configs):
    """Run KMeans++ with multiple seeds on all datasets, return ARI arrays."""
    results = {}
    for cfg in configs:
        name = cfg["name"]
        k = cfg["n_true_clusters"]
        X, y_true = load_dataset(name)
        aris = []
        nmis = []
        fmis = []
        for seed in SEEDS:
            km = KMeans(n_clusters=k, init='k-means++', n_init=1, random_state=seed, max_iter=300)
            y_pred = km.fit_predict(X)
            aris.append(adjusted_rand_score(y_true, y_pred))
            nmis.append(normalized_mutual_info_score(y_true, y_pred))
            fmis.append(fowlkes_mallows_score(y_true, y_pred))
        results[name] = {
            "ari": aris, "nmi": nmis, "fmi": fmis,
            "ari_mean": float(np.mean(aris)), "ari_std": float(np.std(aris)),
            "nmi_mean": float(np.mean(nmis)), "nmi_std": float(np.std(nmis)),
            "fmi_mean": float(np.mean(fmis)), "fmi_std": float(np.std(fmis)),
        }
        print(f"  {name:20s}  ARI={np.mean(aris):.4f}±{np.std(aris):.4f}")
    return results

# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def friedman_test(ari_matrix):
    """
    Friedman test on ARI matrix (datasets x algorithms).
    Returns chi2, p-value.
    """
    stat, p = stats.friedmanchisquare(*[ari_matrix[:, i] for i in range(ari_matrix.shape[1])])
    return float(stat), float(p)

def nemenyi_cd(n_datasets, n_algorithms, alpha=0.05):
    """Critical difference for Nemenyi post-hoc test."""
    # q_alpha values for Nemenyi test (from Demsar 2006, Table 5)
    # For k algorithms: q_alpha at alpha=0.05
    q_values = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}
    q = q_values.get(n_algorithms, 2.569)
    cd = q * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_datasets))
    return cd

def compute_average_ranks(ari_matrix):
    """Compute average ranks across datasets (lower rank = better)."""
    n_datasets, n_algos = ari_matrix.shape
    ranks = np.zeros_like(ari_matrix)
    for i in range(n_datasets):
        # Rank: 1 = best (highest ARI)
        ranks[i] = stats.rankdata(-ari_matrix[i])
    return ranks.mean(axis=0)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_delta_ari(full_results, configs):
    """Lollipop chart of delta ARI (SOM-TSK - KMeans++)."""
    names = []
    deltas = []
    for cfg in configs:
        name = cfg["name"]
        if name not in full_results:
            continue
        r = full_results[name]
        d = r["som"]["ari"] - r["kmeans"]["ari"]
        names.append(name)
        deltas.append(d)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2196F3' if d > 0.005 else '#9E9E9E' for d in deltas]
    y_pos = range(len(names))

    ax.hlines(y_pos, 0, deltas, colors=colors, linewidth=2)
    ax.scatter(deltas, y_pos, color=colors, s=60, zorder=3)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.axvline(0.005, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(-0.005, color='red', linewidth=0.5, linestyle='--', alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(r'$\Delta$ARI (SOM-TSK $-$ KMeans++)')
    ax.set_title('ARI Improvement per Dataset')

    win_patch = mpatches.Patch(color='#2196F3', label=f'Win (Δ > 0.005)')
    tie_patch = mpatches.Patch(color='#9E9E9E', label='Tie (|Δ| ≤ 0.005)')
    ax.legend(handles=[win_patch, tie_patch], loc='lower right')

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "delta_ari.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGS_DIR / 'delta_ari.pdf'}")

def fig_algorithm_comparison(full_results, multiseed, configs):
    """Grouped bar chart comparing all algorithms' ARI."""
    algos = ['som', 'kmeans', 'densom', 'autosom']
    algo_labels = ['SOM-TSK', 'KMeans++', 'DenSOM', 'AutoSOM']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

    # Use only the interesting datasets (not scale/dim which are all 1.0)
    interesting = [c["name"] for c in configs
                   if not c["name"].startswith("scale_") and not c["name"].startswith("dim_")]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(interesting))
    width = 0.2

    for i, (algo, label, color) in enumerate(zip(algos, algo_labels, colors)):
        aris = []
        errs = []
        for name in interesting:
            if name in full_results:
                aris.append(full_results[name][algo]["ari"])
                # Add error bars for KMeans++ from multi-seed
                if algo == "kmeans" and name in multiseed:
                    errs.append(multiseed[name]["ari_std"])
                else:
                    errs.append(0)
            else:
                aris.append(0)
                errs.append(0)
        ax.bar(x + i * width, aris, width, label=label, color=color,
               yerr=errs if algo == "kmeans" else None, capsize=2, alpha=0.85)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(interesting, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('ARI')
    ax.set_title('Algorithm Comparison Across Datasets')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "algorithm_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGS_DIR / 'algorithm_comparison.pdf'}")

def fig_cd_diagram(avg_ranks, algo_labels, cd, n_datasets):
    """Critical difference diagram (Demsar 2006 style)."""
    n_algos = len(algo_labels)
    fig, ax = plt.subplots(figsize=(8, 2.5))

    # Draw axis
    low_rank = 1
    high_rank = n_algos
    ax.set_xlim(low_rank - 0.5, high_rank + 0.5)
    ax.set_ylim(0, 1)
    ax.axhline(0.7, xmin=0, xmax=1, color='black', linewidth=1)

    # Tick marks
    for r in range(1, n_algos + 1):
        ax.axvline(r, ymin=0.65, ymax=0.75, color='black', linewidth=1)
        ax.text(r, 0.78, str(r), ha='center', fontsize=9)

    # Plot algorithms
    sorted_idx = np.argsort(avg_ranks)
    for i, idx in enumerate(sorted_idx):
        rank = avg_ranks[idx]
        y_pos = 0.5 - i * 0.12
        ax.plot(rank, 0.7, 'o', color='black', markersize=6)
        ax.plot([rank, rank], [0.7, y_pos + 0.05], 'k-', linewidth=0.5)
        ax.text(rank, y_pos, f"{algo_labels[idx]} ({rank:.2f})",
                ha='center', fontsize=8, weight='bold')

    # CD bar
    ax.plot([1, 1 + cd], [0.9, 0.9], 'k-', linewidth=2)
    ax.text(1 + cd / 2, 0.93, f'CD = {cd:.2f}', ha='center', fontsize=8)

    ax.set_title(f'Critical Difference Diagram (Nemenyi, α=0.05, {n_datasets} datasets)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(FIGS_DIR / "friedman_nemenyi.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {FIGS_DIR / 'friedman_nemenyi.pdf'}")

def fig_timing_speedup(timing_path):
    """Generate speedup bar chart from timing_comparison.json."""
    if not os.path.exists(timing_path):
        print(f"  Skipping timing figure — {timing_path} not found")
        return
    with open(timing_path) as f:
        timing = json.load(f)

    names, speedup_rayon, speedup_metal, n_samples = [], [], [], []
    for name, d in timing.items():
        s = d.get("serial_ms")
        r = d.get("rayon_ms")
        m = d.get("metal_ms")
        if s is None or r is None or s == 0:
            continue
        names.append(name)
        n_samples.append(d["n_samples"])
        speedup_rayon.append(s / r)
        speedup_metal.append((s / m) if (m and m != 0) else None)

    if not names:
        print("  Skipping timing figure — no valid timing data")
        return

    # Sort by n_samples
    order = sorted(range(len(names)), key=lambda i: n_samples[i])
    names         = [names[i] for i in order]
    speedup_rayon = [speedup_rayon[i] for i in order]
    speedup_metal = [speedup_metal[i] for i in order]
    n_samples     = [n_samples[i] for i in order]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, speedup_rayon, width, label='Rayon / Serial', color='steelblue')
    metal_vals = [v if v is not None else 0.0 for v in speedup_metal]
    if any(v and v > 0 for v in speedup_metal):
        ax.bar(x + width / 2, metal_vals, width, label='Metal / Serial', color='darkorange')

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Dataset (sorted by n_samples)')
    ax.set_ylabel('Speedup over serial (×)')
    ax.set_title('Rayon and Metal Speedup vs. Single-Thread Baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.legend()
    plt.tight_layout()
    out = FIGS_DIR / "timing_speedup.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")

# ---------------------------------------------------------------------------
# V2 comparison
# ---------------------------------------------------------------------------
def compare_v2(full_results, configs):
    """Compare V3 results with V2 (they should be identical since deterministic)."""
    # V2 results from the paper (extracted from the table)
    v2_som_ari = {
        "s1": 0.9762, "s2": 0.6074, "s3": 0.3178, "s4": 0.1886,
        "a1": 0.9770, "a2": 0.8409, "a3": 0.5944,
        "moons": 0.4790, "circles": -0.0033, "spiral": 0.0213,
        "anisotropic": 1.0000, "varied_density": 0.4456,
        "iris": 0.6410, "wine": 0.8975, "breast_cancer": 0.6765, "digits": 0.5795,
        "scale_1k": 0.9779, "scale_5k": 0.9828, "scale_10k": 0.9814, "scale_50k": 0.9839,
        "dim_32": 1.0000, "dim_64": 1.0000, "dim_128": 1.0000, "dim_256": 1.0000,
    }
    v2_km_ari = {
        "s1": 0.9762, "s2": 0.5784, "s3": 0.3105, "s4": 0.1886,
        "a1": 0.9770, "a2": 0.8001, "a3": 0.5790,
        "moons": 0.4790, "circles": -0.0032, "spiral": 0.0184,
        "anisotropic": 1.0000, "varied_density": 0.4452,
        "iris": 0.6451, "wine": 0.8804, "breast_cancer": 0.6765, "digits": 0.3487,
        "scale_1k": 0.9779, "scale_5k": 0.9828, "scale_10k": 0.9814, "scale_50k": 0.9839,
        "dim_32": 1.0000, "dim_64": 1.0000, "dim_128": 1.0000, "dim_256": 1.0000,
    }

    print("\n  V2 vs V3 Comparison (SOM-TSK ARI):")
    print(f"  {'Dataset':<20} {'V2':>8} {'V3':>8} {'Match':>6}")
    print(f"  {'-'*44}")
    all_match = True
    for cfg in configs:
        name = cfg["name"]
        if name not in full_results or name not in v2_som_ari:
            continue
        v2 = v2_som_ari[name]
        v3 = full_results[name]["som"]["ari"]
        match = abs(v2 - v3) < 0.0001
        if not match:
            all_match = False
        print(f"  {name:<20} {v2:>8.4f} {v3:>8.4f} {'✓' if match else '✗':>6}")

    print(f"\n  All results match V2: {'YES ✓' if all_match else 'NO ✗'}")
    return all_match, v2_som_ari, v2_km_ari

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Paper 3: Full Experiment Suite")
    print("=" * 60)

    configs = load_config()
    full_results = load_full_results()

    # --- Step 1: Multi-seed KMeans++ ---
    print("\n[1/5] Running multi-seed KMeans++ (10 seeds per dataset)...")
    multiseed = run_multiseed_kmeans(configs)

    # Save multi-seed results
    out_path = PAPER3_DIR / "multi_seed_results.json"
    # Convert lists to serializable format
    save_data = {}
    for name, data in multiseed.items():
        save_data[name] = {k: v for k, v in data.items()}
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Saved → {out_path}")

    # --- Step 2: V2 Comparison ---
    print("\n[2/5] Comparing with V2 results...")
    all_match, v2_som, v2_km = compare_v2(full_results, configs)

    # --- Step 3: Statistical Tests ---
    print("\n[3/5] Computing Friedman test + Nemenyi post-hoc...")

    # Build ARI matrix: datasets x algorithms (SOM-TSK, KMeans++, DenSOM, AutoSOM)
    algo_names = ['SOM-TSK', 'KMeans++', 'DenSOM', 'AutoSOM']
    algo_keys = ['som', 'kmeans', 'densom', 'autosom']

    # Include GPU column only if ≥50% of datasets have GPU results
    gpu_coverage = sum(
        1 for name in full_results
        if full_results[name].get("som_gpu", {}).get("ari") is not None
    )
    has_gpu = gpu_coverage >= len(full_results) // 2
    if has_gpu:
        algo_names = ['SOM-TSK', 'SOM-TSK GPU', 'KMeans++', 'DenSOM', 'AutoSOM']
        algo_keys  = ['som', 'som_gpu', 'kmeans', 'densom', 'autosom']
        print(f"  GPU results available for {gpu_coverage}/{len(full_results)} datasets — including GPU column")
    else:
        print(f"  GPU results available for {gpu_coverage}/{len(full_results)} datasets — skipping GPU column")

    dataset_names = [c["name"] for c in configs if c["name"] in full_results]
    n_ds = len(dataset_names)

    ari_matrix = np.zeros((n_ds, len(algo_keys)))
    for i, name in enumerate(dataset_names):
        for j, key in enumerate(algo_keys):
            val = full_results[name].get(key, {}).get("ari")
            ari_matrix[i, j] = val if val is not None else 0.0

    chi2, p_value = friedman_test(ari_matrix)
    avg_ranks = compute_average_ranks(ari_matrix)
    cd = nemenyi_cd(n_ds, len(algo_keys), alpha=0.05)

    print(f"  Friedman χ² = {chi2:.4f}, p = {p_value:.6f}")
    print(f"  Average ranks: {dict(zip(algo_names, [f'{r:.2f}' for r in avg_ranks]))}")
    print(f"  Nemenyi CD (α=0.05) = {cd:.4f}")
    print(f"  Significant differences (rank diff > CD):")
    for i in range(len(algo_names)):
        for j in range(i + 1, len(algo_names)):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            sig = "YES" if diff > cd else "no"
            print(f"    {algo_names[i]} vs {algo_names[j]}: |{avg_ranks[i]:.2f} - {avg_ranks[j]:.2f}| = {diff:.2f} → {sig}")

    # Wilcoxon signed-rank test: SOM-TSK vs KMeans++
    som_aris = ari_matrix[:, 0]
    km_aris = ari_matrix[:, 1]
    diffs = som_aris - km_aris
    nonzero = diffs[diffs != 0]
    if len(nonzero) > 0:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(nonzero)
    else:
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    print(f"\n  Wilcoxon signed-rank (SOM-TSK vs KMeans++): W={wilcoxon_stat:.1f}, p={wilcoxon_p:.6f}")

    # Save statistical results
    stat_results = {
        "friedman": {"chi2": chi2, "p_value": p_value, "n_datasets": n_ds, "n_algorithms": len(algo_keys)},
        "average_ranks": dict(zip(algo_names, avg_ranks.tolist())),
        "nemenyi_cd": cd,
        "pairwise_significant": {},
        "wilcoxon_som_vs_km": {"statistic": float(wilcoxon_stat), "p_value": float(wilcoxon_p)},
        "multiseed_km_summary": {
            name: {"mean": d["ari_mean"], "std": d["ari_std"]}
            for name, d in multiseed.items()
        }
    }
    for i in range(len(algo_names)):
        for j in range(i + 1, len(algo_names)):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            stat_results["pairwise_significant"][f"{algo_names[i]}_vs_{algo_names[j]}"] = {
                "rank_diff": float(diff), "significant": bool(diff > cd)
            }

    stat_path = PAPER3_DIR / "statistical_tests.json"
    with open(stat_path, "w") as f:
        json.dump(stat_results, f, indent=2)
    print(f"  Saved → {stat_path}")

    # --- Step 4: Generate Figures ---
    print("\n[4/5] Generating figures...")
    fig_delta_ari(full_results, configs)
    fig_algorithm_comparison(full_results, multiseed, configs)
    fig_cd_diagram(avg_ranks, algo_names, cd, n_ds)
    timing_path = str(PAPER3_DIR / "timing_comparison.json")
    fig_timing_speedup(timing_path)

    # --- Step 5: Summary ---
    print("\n[5/5] Summary")
    print("=" * 60)
    print(f"  Datasets: {n_ds}")
    print(f"  Algorithms: {', '.join(algo_names)}")
    print(f"  V2 match: {'YES' if all_match else 'NO'}")
    print(f"  Friedman p-value: {p_value:.2e} ({'SIGNIFICANT' if p_value < 0.05 else 'not significant'})")
    print(f"  SOM-TSK avg rank: {avg_ranks[0]:.2f} (best=1.0)")
    print(f"  KMeans++ avg rank: {avg_ranks[1]:.2f}")
    print(f"  Wilcoxon p (SOM vs KM): {wilcoxon_p:.4f}")

    # Multi-seed KMeans++ vs deterministic SOM-TSK comparison
    print(f"\n  Multi-seed KMeans++ vs SOM-TSK (best of {N_SEEDS} seeds):")
    wins_vs_best = 0
    for cfg in configs:
        name = cfg["name"]
        if name not in full_results or name not in multiseed:
            continue
        som_ari = full_results[name]["som"]["ari"]
        km_best = max(multiseed[name]["ari"])
        if som_ari - km_best > 0.005:
            wins_vs_best += 1
            print(f"    {name}: SOM-TSK={som_ari:.4f} > best-KM++={km_best:.4f}")
    print(f"  SOM-TSK wins vs best-of-{N_SEEDS} KMeans++: {wins_vs_best}")

    print("\n  Output files:")
    print(f"    {PAPER3_DIR / 'multi_seed_results.json'}")
    print(f"    {stat_path}")
    print(f"    {FIGS_DIR / 'delta_ari.pdf'}")
    print(f"    {FIGS_DIR / 'algorithm_comparison.pdf'}")
    print(f"    {FIGS_DIR / 'friedman_nemenyi.pdf'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
