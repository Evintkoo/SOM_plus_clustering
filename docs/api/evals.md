# Evaluation Metrics Module

## Overview

The `evals.py` module provides a comprehensive set of functions to evaluate clustering performance using various metrics. These metrics help assess the quality and effectiveness of different clustering algorithms by comparing distances within clusters and across different clusters.

## Performance Optimizations

- **Vectorized operations** using NumPy for better performance
- **Pre-computed distance matrices** using scipy.spatial.distance
- **Numerical stability improvements** with epsilon values
- **Efficient memory usage** with optimized array operations

## Clustering Evaluation Functions

### `silhouette_score(x: np.ndarray, labels: np.ndarray) -> float`

Calculate the Silhouette Coefficient for a clustering result.

**Parameters:**
- `x` (np.ndarray): Input data points, shape (n_samples, n_features)
- `labels` (np.ndarray): Assigned cluster labels for each data point, shape (n_samples,)

**Returns:**
- `float`: Silhouette Coefficient (range: [-1, 1], higher is better)

**Algorithm:**
1. Pre-compute pairwise distances using `scipy.spatial.distance.pdist`
2. For each point, calculate:
   - `a`: Mean intra-cluster distance
   - `b`: Mean distance to nearest neighboring cluster
3. Silhouette score: `s = (b - a) / max(a, b)`

**Interpretation:**
- Values near +1: Well-separated clusters
- Values near 0: Overlapping clusters  
- Values near -1: Incorrectly assigned points

**Edge Cases:**
- Returns 0.0 for single cluster
- Handles clusters with single samples

### `davies_bouldin_index(x: np.ndarray, labels: np.ndarray) -> float`

Compute the Davies-Bouldin Index for evaluating clustering results.

**Parameters:**
- `x` (np.ndarray): Input data points, shape (n_samples, n_features)
- `labels` (np.ndarray): Cluster labels for each data point, shape (n_samples,)

**Returns:**
- `float`: Davies-Bouldin Index (lower values indicate better clustering)

**Algorithm:**
1. Compute centroids for each cluster
2. Calculate intra-cluster dispersions (mean distance to centroid)
3. Compute pairwise centroid distances
4. For each cluster, find maximum ratio: `(dispersions[i] + dispersions[j]) / centroid_distance[i,j]`
5. Return average of maximum ratios

**Features:**
- Optimized with pre-computed centroids and dispersions
- Vectorized dispersion computation
- Efficient pairwise distance calculation

### `calinski_harabasz_score(x: np.ndarray, labels: np.ndarray) -> float`

Calculate the Calinski-Harabasz Index (Variance Ratio Criterion).

**Parameters:**
- `x` (np.ndarray): Input data matrix, shape (n_samples, n_features)
- `labels` (np.ndarray): Cluster labels for each data point, shape (n_samples,)

**Returns:**
- `float`: Calinski-Harabasz Index (higher values indicate better clustering)

**Algorithm:**
1. Calculate overall centroid of dataset
2. Compute between-cluster dispersion (weighted by cluster size)
3. Compute within-cluster dispersion (sum of squared deviations)
4. Return scaled ratio: `((n_samples - n_clusters) / (n_clusters - 1)) * (between_dispersion / within_dispersion)`

**Mathematical Formula:**
```
CH = ((n - k) / (k - 1)) * (Σ n_i * ||c_i - c||² / Σ ||x_j - c_i||²)
```
where:
- `n`: number of samples
- `k`: number of clusters
- `n_i`: size of cluster i
- `c_i`: centroid of cluster i
- `c`: overall centroid

### `dunn_index(x: np.ndarray, labels: np.ndarray) -> float`

Calculate the Dunn Index for clustering evaluation.

**Parameters:**
- `x` (np.ndarray): Input data points, shape (n_samples, n_features)
- `labels` (np.ndarray): Cluster labels for each data point, shape (n_samples,)

**Returns:**
- `float`: Dunn Index (higher values indicate better clustering)

**Algorithm:**
1. Pre-compute distance matrix using `scipy.spatial.distance.squareform`
2. Find minimum inter-cluster distance (between different clusters)
3. Find maximum intra-cluster distance (within same cluster)
4. Return ratio: `min_inter_cluster_distance / max_intra_cluster_distance`

**Interpretation:**
- Higher values indicate better separation and compactness
- Sensitive to outliers and noise

## Distribution Comparison Functions

### `compare_distribution(data1: np.ndarray, data2: np.ndarray, num_bins: int = 100) -> float`

Compare the distribution of two datasets using histogram analysis.

**Parameters:**
- `data1` (np.ndarray): First dataset, shape (n_features, n_samples1)
- `data2` (np.ndarray): Second dataset, shape (n_features, n_samples2)  
- `num_bins` (int): Number of bins for histogram calculation. Defaults to 100

**Returns:**
- `float`: Mean of average squared differences between normalized histograms

**Algorithm:**
1. For each feature dimension:
   - Compute normalized histograms for both datasets
   - Calculate mean absolute difference between histograms
2. Return average across all features

### `bcubed_precision_recall(clusters: np.ndarray, labels: np.ndarray) -> Tuple[float, float]`

Compute BCubed Precision and Recall for clustering results.

**Parameters:**
- `clusters` (np.ndarray): Cluster assignments, shape (n_samples,)
- `labels` (np.ndarray): Ground truth labels, shape (n_samples,)

**Returns:**
- `Tuple[float, float]`: (BCubed Precision, BCubed Recall)

**Algorithm:**
For each point i:
- Precision: Fraction of points in same cluster that have same true label
- Recall: Fraction of points with same true label that are in same cluster

**Mathematical Formulas:**
```
Precision_i = |C(i) ∩ L(i)| / |C(i)|
Recall_i = |C(i) ∩ L(i)| / |L(i)|
```

## Supervised Learning Metrics

### `accuracy(y_true, y_pred) -> float`

Calculate the accuracy of predictions.

**Parameters:**
- `y_true` (list or array): True labels
- `y_pred` (list or array): Predicted labels

**Returns:**
- `float`: Accuracy as a percentage

**Formula:**
```
Accuracy = (Number of correct predictions / Total predictions) * 100
```

### `f1_score(y_true, y_pred) -> float`

Calculate the F1 score of predictions.

**Parameters:**
- `y_true` (list or array): True labels  
- `y_pred` (list or array): Predicted labels

**Returns:**
- `float`: F1 score

**Algorithm:**
1. Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
2. Compute Precision = TP / (TP + FP)
3. Compute Recall = TP / (TP + FN)  
4. F1 = 2 * (Precision * Recall) / (Precision + Recall)

### `recall(y_true, y_pred) -> float`

Calculate the recall of predictions.

**Parameters:**
- `y_true` (list or array): True labels
- `y_pred` (list or array): Predicted labels

**Returns:**
- `float`: Recall value

**Formula:**
```
Recall = TP / (TP + FN)
```

## Input Validation

### `_validate_clustering_inputs(x: np.ndarray, labels: np.ndarray) -> None`

Validate inputs for clustering evaluation metrics.

**Parameters:**
- `x` (np.ndarray): Input data points
- `labels` (np.ndarray): Cluster labels

**Raises:**
- `ValueError`: If inputs are empty or invalid
- `IndexError`: If dimensions don't match

**Checks:**
- Non-empty arrays
- Matching number of samples
- Minimum data requirements

## Usage Examples

### Basic Clustering Evaluation

```python
import numpy as np
from modules.evals import (
    silhouette_score, davies_bouldin_index, 
    calinski_harabasz_score, dunn_index
)

# Sample data and labels
X = np.random.rand(100, 2)
labels = np.random.randint(0, 3, 100)

# Calculate clustering metrics
silhouette = silhouette_score(X, labels)
db_index = davies_bouldin_index(X, labels)
ch_index = calinski_harabasz_score(X, labels)
dunn = dunn_index(X, labels)

print(f"Silhouette Score: {silhouette:.3f}")
print(f"Davies-Bouldin Index: {db_index:.3f}")
print(f"Calinski-Harabasz Index: {ch_index:.3f}")
print(f"Dunn Index: {dunn:.3f}")
```

### Comprehensive Evaluation

```python
import numpy as np
from modules.evals import *

# Generate clustering results
X = np.random.rand(200, 3)
predicted_labels = np.random.randint(0, 4, 200)
true_labels = np.random.randint(0, 4, 200)

# Unsupervised evaluation
print("=== Unsupervised Metrics ===")
print(f"Silhouette: {silhouette_score(X, predicted_labels):.3f}")
print(f"Davies-Bouldin: {davies_bouldin_index(X, predicted_labels):.3f}")
print(f"Calinski-Harabasz: {calinski_harabasz_score(X, predicted_labels):.3f}")
print(f"Dunn Index: {dunn_index(X, predicted_labels):.3f}")

# Supervised evaluation (if ground truth available)
print("\n=== Supervised Metrics ===")
precision, recall_score = bcubed_precision_recall(predicted_labels, true_labels)
print(f"BCubed Precision: {precision:.3f}")
print(f"BCubed Recall: {recall_score:.3f}")

# Classification metrics (for binary classification)
binary_true = (true_labels > 1).astype(int)
binary_pred = (predicted_labels > 1).astype(int)
print(f"Accuracy: {accuracy(binary_true, binary_pred):.1f}%")
print(f"F1 Score: {f1_score(binary_true, binary_pred):.3f}")
print(f"Recall: {recall(binary_true, binary_pred):.3f}")
```

### Distribution Comparison

```python
import numpy as np
from modules.evals import compare_distribution

# Compare two datasets
data1 = np.random.normal(0, 1, (5, 1000))  # 5 features, 1000 samples
data2 = np.random.normal(0.5, 1.2, (5, 800))  # Different distribution

# Compare distributions
dist_diff = compare_distribution(data1, data2, num_bins=50)
print(f"Distribution difference: {dist_diff:.4f}")
```

## Performance Notes

- **Distance Matrix Caching**: Pre-computed distance matrices improve performance for multiple metrics
- **Vectorized Operations**: NumPy broadcasting reduces computation time
- **Memory Efficiency**: Optimized for large datasets with minimal memory allocation
- **Numerical Stability**: Added epsilon values prevent division by zero
- **Input Validation**: Early validation prevents runtime errors

## Mathematical Properties

### Silhouette Score
- **Range**: [-1, 1]
- **Optimal**: Values close to 1
- **Interpretation**: Measures both cohesion and separation

### Davies-Bouldin Index  
- **Range**: [0, ∞)
- **Optimal**: Values close to 0
- **Interpretation**: Lower values indicate better clustering

### Calinski-Harabasz Index
- **Range**: [0, ∞)
- **Optimal**: Higher values
- **Interpretation**: Ratio of between-cluster to within-cluster variance

### Dunn Index
- **Range**: [0, ∞)
- **Optimal**: Higher values
- **Interpretation**: Ratio of minimum separation to maximum diameter