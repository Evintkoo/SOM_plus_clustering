"""
Generate figures for the SOM-TSK paper.
Run from repo root:  python docs/paper/gen_figures.py
Outputs PDF figures to docs/paper/figs/
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── paths ────────────────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
RES     = os.path.join(HERE, "../../experiments/benchmark/results/full_results.json")
FIGDIR  = os.path.join(HERE, "figs")
os.makedirs(FIGDIR, exist_ok=True)

with open(RES) as f:
    data = json.load(f)

# ── colour / style ────────────────────────────────────────────────────────────
SOM_COLOR = "#1f77b4"   # blue
KM_COLOR  = "#d62728"   # red
matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       150,
    "pdf.fonttype":     42,   # embed fonts for IEEE
})

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – ARI comparison bar chart across all 24 datasets
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORIES = [
    ("SIPU S-sets",         ["s1", "s2", "s3", "s4"]),
    ("SIPU A-sets",         ["a1", "a2", "a3"]),
    ("Shape tests",         ["moons", "circles", "spiral", "anisotropic", "varied_density"]),
    ("UCI real-world",      ["iris", "wine", "breast_cancer", "digits"]),
    ("Scalability",         ["scale_1k", "scale_5k", "scale_10k", "scale_50k"]),
    ("High-dim",            ["dim_32", "dim_64", "dim_128", "dim_256"]),
]

PRETTY = {
    "s1": "s1", "s2": "s2", "s3": "s3", "s4": "s4",
    "a1": "a1", "a2": "a2", "a3": "a3",
    "moons": "moons", "circles": "circles", "spiral": "spiral",
    "anisotropic": "aniso.", "varied_density": "varied",
    "iris": "Iris", "wine": "Wine", "breast_cancer": "Cancer", "digits": "Digits",
    "scale_1k": "1k", "scale_5k": "5k", "scale_10k": "10k", "scale_50k": "50k",
    "dim_32": "d=32", "dim_64": "d=64", "dim_128": "d=128", "dim_256": "d=256",
}

all_names = [n for _, names in CATEGORIES for n in names if n in data]
som_ari  = [data[n]["som"]["ari"] for n in all_names]
km_ari   = [data[n]["kmeans"]["ari"] for n in all_names]
labels   = [PRETTY[n] for n in all_names]

x     = np.arange(len(all_names))
width = 0.38

fig, ax = plt.subplots(figsize=(7.0, 2.4))
bars1 = ax.bar(x - width/2, som_ari, width, label="SOM-TSK", color=SOM_COLOR, linewidth=0)
bars2 = ax.bar(x + width/2, km_ari,  width, label="KMeans++", color=KM_COLOR,  linewidth=0, alpha=0.80)

# Category separators + labels
cat_boundaries = []
pos = 0
for cat_label, names in CATEGORIES:
    n_in = sum(1 for n in names if n in data)
    if n_in == 0:
        continue
    mid = pos + n_in / 2 - 0.5
    ax.axvline(pos - 0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.text(mid, 1.04, cat_label, ha="center", va="bottom",
            fontsize=6, color="dimgray", transform=ax.get_xaxis_transform())
    pos += n_in

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("ARI")
ax.set_ylim(-0.05, 1.15)
ax.axhline(0, color="black", linewidth=0.4)
ax.legend(loc="upper right", framealpha=0.9)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIGDIR, "ari_comparison.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figs/ari_comparison.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Scalability: fit time vs N  (log–log)
# ═══════════════════════════════════════════════════════════════════════════════

scale_names = ["scale_1k", "scale_5k", "scale_10k", "scale_50k"]
ns      = [data[n]["n_samples"]         for n in scale_names]
som_t   = [data[n]["som"]["fit_time_s"] for n in scale_names]
km_t    = [data[n]["kmeans"]["fit_time_s"] for n in scale_names]

fig, ax = plt.subplots(figsize=(3.3, 2.4))
ax.loglog(ns, som_t, "o-",  color=SOM_COLOR, label="SOM-TSK", linewidth=1.2, markersize=4)
ax.loglog(ns, km_t,  "s--", color=KM_COLOR,  label="KMeans++", linewidth=1.2, markersize=4)

# Annotate points
for n_, t_ in zip(ns, som_t):
    ax.annotate(f"{t_:.2f}s", (n_, t_), textcoords="offset points", xytext=(4, 3), fontsize=6)
for n_, t_ in zip(ns, km_t):
    ax.annotate(f"{t_*1000:.1f}ms", (n_, t_), textcoords="offset points", xytext=(4, -9), fontsize=6)

ax.set_xlabel("Dataset size $n$")
ax.set_ylabel("Fit time (s)")
ax.set_title("Scalability: fit time vs $n$  ($d=2$, $k=5$)")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.legend(loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "scalability.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figs/scalability.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Delta ARI lollipop (SOM-TSK − KMeans++ per dataset)
# ═══════════════════════════════════════════════════════════════════════════════

delta = [s - k for s, k in zip(som_ari, km_ari)]
WIN_THR = 0.005

colors = []
for d in delta:
    if d > WIN_THR:
        colors.append(SOM_COLOR)
    elif d < -WIN_THR:
        colors.append(KM_COLOR)
    else:
        colors.append("gray")

fig, ax = plt.subplots(figsize=(7.0, 2.0))
ax.axhline(0, color="black", linewidth=0.6)
ax.axhline( WIN_THR, color="black", linewidth=0.4, linestyle=":")
ax.axhline(-WIN_THR, color="black", linewidth=0.4, linestyle=":")

for i, (d, c) in enumerate(zip(delta, colors)):
    ax.plot([i, i], [0, d], color=c, linewidth=1.2)
    ax.plot(i, d, "o", color=c, markersize=4)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel(r"$\Delta$ARI")
ax.set_title("ARI gain: SOM-TSK $-$ KMeans++ per dataset  (blue = win, gray = tie)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Category separators
pos = 0
for cat_label, names in CATEGORIES:
    n_in = sum(1 for n in names if n in data)
    if n_in == 0:
        continue
    ax.axvline(pos - 0.5, color="gray", linewidth=0.5, linestyle="--")
    pos += n_in

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "delta_ari.pdf"), bbox_inches="tight")
plt.close(fig)
print("Saved figs/delta_ari.pdf")

print("\nAll figures generated in docs/paper/figs/")
