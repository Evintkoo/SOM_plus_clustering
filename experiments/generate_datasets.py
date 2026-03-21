"""
Generate normalized benchmark datasets as CSV files for the Python vs Rust experiment.
Each CSV has feature columns followed by a 'label' column.
Data is StandardScaler-normalized so both runners get identical inputs.

Run once before either experiment runner:
    python experiments/generate_datasets.py
"""
import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

OUT_DIR = os.path.join(os.path.dirname(__file__), "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

def save(name, X, y):
    X_scaled = StandardScaler().fit_transform(X)
    df = pd.DataFrame(X_scaled, columns=[f"f{i}" for i in range(X_scaled.shape[1])])
    df["label"] = y.astype(int)
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  {name:20s}  {X_scaled.shape[0]:5d} x {X_scaled.shape[1]:3d}  k={len(np.unique(y))}  -> {path}")


print("Generating datasets...")

# Iris: 150 x 4, 3 classes
iris = datasets.load_iris()
save("iris", iris.data, iris.target)

# Wine: 178 x 13, 3 classes
wine = datasets.load_wine()
save("wine", wine.data, wine.target)

# Breast Cancer: 569 x 30, 2 classes
bc = datasets.load_breast_cancer()
save("breast_cancer", bc.data, bc.target)

# Digits: use first 500 samples, 64 features, 10 classes
digits = datasets.load_digits()
save("digits", digits.data[:500], digits.target[:500])

# Synthetic blobs: 1000 x 10, 5 classes (reproducible)
X_blobs, y_blobs = make_blobs(n_samples=1000, n_features=10, centers=5, random_state=42)
save("blobs", X_blobs, y_blobs)

print("Done.")
