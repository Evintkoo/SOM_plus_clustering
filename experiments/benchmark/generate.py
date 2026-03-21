"""
Generate the full ML QA benchmark dataset suite.

Covers 6 categories that a QA analyst needs to stress-test a clustering library:
  1. SIPU S-sets     — Gaussian mixtures, varying overlap     (s1–s4)
  2. SIPU A-sets     — Gaussian mixtures, increasing k        (a1–a3)
  3. Shape stress    — non-Gaussian topology                  (moons, circles, spiral, anisotropic)
  4. Real-world UCI  — iris, wine, breast_cancer, digits
  5. Scalability     — fixed 2-D, growing N                   (1k–50k)
  6. High-dimensional — fixed N, growing D                    (32–256 dims)

Outputs:
  experiments/benchmark/datasets/<name>.csv   (features + 'label' column, StandardScaled)
  experiments/benchmark/dataset_config.json   (k, som_m, som_n, epochs per dataset)

Run:
    python experiments/benchmark/generate.py
"""
import os, json, sys
import numpy as np
import pandas as pd
from sklearn import datasets as skds
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, make_circles

OUT_DIR    = os.path.join(os.path.dirname(__file__), "datasets")
CFG_PATH   = os.path.join(os.path.dirname(__file__), "dataset_config.json")
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.RandomState(42)
configs = []   # list of dicts written to dataset_config.json

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def save(name, X, y, som_m, som_n, epochs):
    X_sc = StandardScaler().fit_transform(X)
    df = pd.DataFrame(X_sc, columns=[f"f{i}" for i in range(X_sc.shape[1])])
    df["label"] = y.astype(int)
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    k = int(len(np.unique(y)))
    configs.append({
        "name": name,
        "n_true_clusters": k,
        "som_m": som_m,
        "som_n": som_n,
        "epochs": epochs,
    })
    print(f"  {name:<28} {X_sc.shape[0]:>6} x {X_sc.shape[1]:<4} k={k:<4} grid={som_m}x{som_n}")


def s_set(name, std_frac, som_m=8, som_n=8, epochs=5):
    """5000-pt 2-D Gaussian mixture on a 5×3 grid (15 clusters)."""
    k = 15
    n_per = 5000 // k
    # Centers at a 3-col × 5-row grid in [0.1, 0.9]^2
    cx = np.linspace(0.1, 0.9, 3)
    cy = np.linspace(0.1, 0.9, 5)
    centers = np.array([[x, y] for y in cy for x in cx])  # 15 centers
    X, y = [], []
    for i, (cx_, cy_) in enumerate(centers):
        pts = RNG.randn(n_per, 2) * std_frac + [cx_, cy_]
        X.append(pts)
        y.extend([i] * n_per)
    save(name, np.vstack(X), np.array(y), som_m, som_n, epochs)


def a_set(name, k, std_frac=0.04, epochs=5):
    """A-set: ~150 pts/cluster, 2-D, grid-arranged centers."""
    n_per = 150
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    cx = np.linspace(0.1, 0.9, cols)
    cy = np.linspace(0.1, 0.9, rows)
    centers = np.array([[x, y] for y in cy for x in cx])[:k]
    X, y = [], []
    for i, (cx_, cy_) in enumerate(centers):
        pts = RNG.randn(n_per, 2) * std_frac + [cx_, cy_]
        X.append(pts)
        y.extend([i] * n_per)
    m = n = int(np.ceil(np.sqrt(k * 4)))   # ~4 neurons per true cluster
    save(name, np.vstack(X), np.array(y), m, n, epochs)


def make_spiral(n_pts=300, n_spirals=3, noise=0.05):
    """Generate n_spirals interleaved Archimedean spirals."""
    per = n_pts // n_spirals
    X, y = [], []
    for i in range(n_spirals):
        t = np.linspace(0, 1, per)
        angle = 2 * np.pi * i / n_spirals + 3 * np.pi * t
        r = t
        x1 = r * np.cos(angle) + RNG.randn(per) * noise
        x2 = r * np.sin(angle) + RNG.randn(per) * noise
        X.append(np.c_[x1, x2])
        y.extend([i] * per)
    return np.vstack(X), np.array(y)


# ---------------------------------------------------------------------------
# 1. S-sets  (SIPU benchmark, Fränti & Virmajoki 2006)
# ---------------------------------------------------------------------------
print("\n--- S-sets (SIPU) ---")
s_set("s1", std_frac=0.04)   # well-separated
s_set("s2", std_frac=0.09)   # mild overlap
s_set("s3", std_frac=0.14)   # moderate overlap
s_set("s4", std_frac=0.19)   # heavy overlap

# ---------------------------------------------------------------------------
# 2. A-sets  (SIPU benchmark)
# ---------------------------------------------------------------------------
print("\n--- A-sets (SIPU) ---")
a_set("a1", k=20)
a_set("a2", k=35)
a_set("a3", k=50)

# ---------------------------------------------------------------------------
# 3. Shape stress tests
# ---------------------------------------------------------------------------
print("\n--- Shape stress tests ---")

# Two moons (SOM should struggle like KMeans — not linearly separable)
X_m, y_m = make_moons(n_samples=300, noise=0.08, random_state=42)
save("moons",      X_m, y_m, 5, 5, 10)

# Concentric rings
X_c, y_c = make_circles(n_samples=300, noise=0.05, factor=0.4, random_state=42)
save("circles",    X_c, y_c, 5, 5, 10)

# Interleaved spirals (hardest for distance-based methods)
X_sp, y_sp = make_spiral(n_pts=300, n_spirals=3, noise=0.04)
save("spiral",     X_sp, y_sp, 5, 5, 10)

# Anisotropic blobs (elongated — tests distance bias)
X_an, y_an = make_blobs(n_samples=300, centers=3, random_state=42)
T = np.array([[0.6, -0.6], [-0.4, 0.8]])
X_an = X_an @ T
save("anisotropic", X_an, y_an, 5, 5, 10)

# Varied density
X_vd_parts, y_vd = [], []
for i, (n, std) in enumerate([(100, 0.5), (300, 2.0), (600, 4.0)]):
    center = np.array([i * 12.0, 0.0])
    X_vd_parts.append(RNG.randn(n, 2) * std + center)
    y_vd.extend([i] * n)
save("varied_density", np.vstack(X_vd_parts), np.array(y_vd), 5, 5, 10)

# ---------------------------------------------------------------------------
# 4. Real-world UCI benchmarks
# ---------------------------------------------------------------------------
print("\n--- Real-world UCI ---")

iris = skds.load_iris()
save("iris",          iris.data,                     iris.target,   5, 5, 10)

wine = skds.load_wine()
save("wine",          wine.data,                     wine.target,   5, 5, 10)

bc = skds.load_breast_cancer()
save("breast_cancer", bc.data,                       bc.target,     6, 6, 10)

digits = skds.load_digits()
save("digits",        digits.data[:500],             digits.target[:500], 8, 8, 10)

# ---------------------------------------------------------------------------
# 5. Scalability (2-D, k=5, varying N)
# ---------------------------------------------------------------------------
print("\n--- Scalability (varying N) ---")

for n_pts, tag in [(1_000, "1k"), (5_000, "5k"), (10_000, "10k"), (50_000, "50k")]:
    X_s, y_s = make_blobs(n_samples=n_pts, n_features=2, centers=5,
                           cluster_std=0.8, random_state=42)
    epochs = 5 if n_pts >= 5_000 else 10
    save(f"scale_{tag}", X_s, y_s, 10, 10, epochs)

# ---------------------------------------------------------------------------
# 6. High-dimensional (N=1000, k=5, varying D)
# ---------------------------------------------------------------------------
print("\n--- High-dimensional (varying D) ---")

for d in [32, 64, 128, 256]:
    X_d, y_d = make_blobs(n_samples=1_000, n_features=d, centers=5,
                           cluster_std=1.0, random_state=42)
    save(f"dim_{d}", X_d, y_d, 8, 8, 10)

# ---------------------------------------------------------------------------
# Write config
# ---------------------------------------------------------------------------
with open(CFG_PATH, "w") as f:
    json.dump(configs, f, indent=2)

print(f"\n{len(configs)} datasets written.")
print(f"Config saved to {CFG_PATH}")
