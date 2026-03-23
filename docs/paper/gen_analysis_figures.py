"""
Generate additional analysis figures for the SOM-TSK paper.
Run from repo root:  python3 docs/paper/gen_analysis_figures.py
Outputs to docs/paper/figs/
"""
import json, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import adjusted_rand_score

HERE   = os.path.dirname(os.path.abspath(__file__))
DSDIR  = os.path.join(HERE, "../../experiments/benchmark/datasets")
RESDIR = os.path.join(HERE, "../../experiments/benchmark/results")
FIGDIR = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)

SOM_COLOR = "#1f77b4"
KM_COLOR  = "#d62728"

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize":7,
    "ytick.labelsize":7,
    "legend.fontsize":7,
    "figure.dpi":     150,
    "pdf.fonttype":   42,
})

# ── helpers ──────────────────────────────────────────────────────────────────

def load_dataset(name):
    path = os.path.join(DSDIR, f"{name}.csv")
    X, y_true = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(v) for k, v in row.items() if k != "label"])
            y_true.append(int(row["label"]))
    return np.array(X), np.array(y_true)

def load_labels(name, algo):
    path = os.path.join(RESDIR, f"{name}_{algo}_labels.csv")
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    return np.array(labels)

def bootstrap_delta_ari(X, y_true, y_som, y_km, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ari_s = adjusted_rand_score(y_true[idx], y_som[idx])
        ari_k = adjusted_rand_score(y_true[idx], y_km[idx])
        deltas.append(ari_s - ari_k)
    return np.array(deltas)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Cluster scatter: ground truth vs SOM-TSK vs KMeans++  (s2 & a2)
# ═══════════════════════════════════════════════════════════════════════════════

PALETTE = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors

def scatter_clusters(ax, X, labels, title, ari):
    ulabels = np.unique(labels)
    for i, lab in enumerate(ulabels):
        mask = labels == lab
        ax.scatter(X[mask, 0], X[mask, 1], s=1.5, color=PALETTE[i % len(PALETTE)],
                   linewidths=0, rasterized=True)
    ax.set_title(f"{title}\nARI = {ari:.3f}", pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4))
for row_i, name in enumerate(["s2", "a2"]):
    X, y_true = load_dataset(name)
    y_som = load_labels(name, "som")
    y_km  = load_labels(name, "km")
    ari_som = adjusted_rand_score(y_true, y_som)
    ari_km  = adjusted_rand_score(y_true, y_km)
    scatter_clusters(axes[row_i, 0], X, y_true, f"{name} — Ground Truth", 1.0)
    scatter_clusters(axes[row_i, 1], X, y_som,  f"{name} — SOM-TSK", ari_som)
    scatter_clusters(axes[row_i, 2], X, y_km,   f"{name} — KMeans++", ari_km)

fig.suptitle("Cluster assignments on two winning datasets (each point = one sample)", fontsize=8)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIGDIR, "cluster_scatter.pdf"), bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved figs/cluster_scatter.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Bootstrap confidence intervals on ΔARI for the 6 winning datasets
# ═══════════════════════════════════════════════════════════════════════════════

WIN_DATASETS = ["s2", "s3", "a2", "a3", "wine", "digits"]
WIN_LABELS   = ["s2\n(k=15)", "s3\n(k=15)", "a2\n(k=35)", "a3\n(k=50)", "Wine\n(k=3)", "Digits\n(k=10)"]

fig, ax = plt.subplots(figsize=(5.5, 2.8))
ax.axhline(0, color="black", linewidth=0.7)
ax.axhline(0.005,  color="gray", linewidth=0.5, linestyle=":", label="Win threshold (0.005)")
ax.axhline(-0.005, color="gray", linewidth=0.5, linestyle=":")

for i, (name, label) in enumerate(zip(WIN_DATASETS, WIN_LABELS)):
    X, y_true = load_dataset(name)
    y_som = load_labels(name, "som")
    y_km  = load_labels(name, "km")
    deltas = bootstrap_delta_ari(X, y_true, y_som, y_km, n_boot=3000)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    med = np.median(deltas)
    obs = adjusted_rand_score(y_true, y_som) - adjusted_rand_score(y_true, y_km)

    # CI bar
    ax.plot([i, i], [lo, hi], color=SOM_COLOR, linewidth=2.5, solid_capstyle="round")
    # Observed point
    ax.plot(i, obs, "o", color=SOM_COLOR, markersize=6, zorder=5)
    # Annotate CI
    ax.annotate(f"[{lo:+.3f},\n {hi:+.3f}]", (i, hi),
                textcoords="offset points", xytext=(6, 2), fontsize=5.5, color="dimgray")

ax.set_xticks(range(len(WIN_DATASETS)))
ax.set_xticklabels(WIN_LABELS)
ax.set_ylabel(r"$\Delta$ARI = SOM-TSK $-$ KMeans++")
ax.set_title(r"Bootstrap 95\% CI on $\Delta$ARI for the six winning datasets ($B=3000$)")
ax.legend(loc="lower right", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "bootstrap_ci.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figs/bootstrap_ci.pdf")

# Print CI table for the paper
print("\nBootstrap 95% CI table:")
print(f"{'Dataset':<10} {'Obs ΔARI':>10} {'CI lo':>8} {'CI hi':>8} {'p(ΔARI>0)':>10}")
for name, label in zip(WIN_DATASETS, WIN_LABELS):
    X, y_true = load_dataset(name)
    y_som = load_labels(name, "som")
    y_km  = load_labels(name, "km")
    deltas = bootstrap_delta_ari(X, y_true, y_som, y_km, n_boot=5000)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    obs = adjusted_rand_score(y_true, y_som) - adjusted_rand_score(y_true, y_km)
    p = np.mean(deltas > 0)
    print(f"{name:<10} {obs:>+10.4f} {lo:>8.4f} {hi:>8.4f} {p:>10.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 – AutoSOM: k_detected vs true k across all datasets
# ═══════════════════════════════════════════════════════════════════════════════

with open(os.path.join(RESDIR, "full_results.json")) as f:
    data = json.load(f)

auto_names = sorted(data.keys())
true_ks  = [data[n]["n_true_clusters"]               for n in auto_names]
det_ks   = [data[n]["autosom"]["k_detected"]          for n in auto_names]
auto_aris= [data[n]["autosom"]["ari"]                 for n in auto_names]
som_aris = [data[n]["som"]["ari"]                     for n in auto_names]

x = np.arange(len(auto_names))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.0), sharex=True)

# top: k_detected vs true k
ax1.scatter(x, true_ks, marker="_", s=50, color="black", linewidths=1.5, label="True $k$", zorder=4)
ax1.scatter(x, det_ks,  marker="o", s=18, color=SOM_COLOR, label="AutoSOM $\\hat{k}$", zorder=5)
for xi, (tk, dk) in enumerate(zip(true_ks, det_ks)):
    ax1.plot([xi, xi], [tk, dk], color="lightsteelblue", linewidth=0.8, zorder=3)
ax1.set_ylabel("$k$")
ax1.set_title("AutoSOM: estimated $\\hat{k}$ vs.\ true $k$ (top) and resulting ARI (bottom)")
ax1.legend(loc="upper right", ncol=2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# bottom: AutoSOM ARI vs SOM-TSK ARI (oracle k)
ax2.bar(x - 0.2, som_aris,  0.38, color=SOM_COLOR, alpha=0.85, label="SOM-TSK (oracle $k$)", linewidth=0)
ax2.bar(x + 0.2, auto_aris, 0.38, color="darkorange", alpha=0.85, label="AutoSOM (est. $k$)", linewidth=0)
ax2.set_ylabel("ARI")
ax2.set_xticks(x)
ax2.set_xticklabels([n.replace("_", r"\_") for n in auto_names], rotation=45, ha="right", fontsize=6)
ax2.legend(loc="upper right", ncol=2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "autosom_analysis.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figs/autosom_analysis.pdf")

print("\nAll analysis figures generated.")
