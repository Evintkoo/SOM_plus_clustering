# KMeans Module

## Overview

The `kmeans.py` module contains an optimized implementation of the KMeans clustering algorithm, which partitions a dataset into a number of clusters by minimizing the variance within each cluster.

## Performance Optimizations

- **Numba JIT compilation** for distance calculations and cluster assignments
- **Vectorized operations** for better performance
- **Improved centroid initialization** and updates
- **Better convergence detection**
- **Reduced memory allocation**

## Classes

### KMeans

Optimized KMeans clustering algorithm with vectorized operations.

#### Constructor

```python
KMeans(n_clusters: int, method: str, tol: float = 1e-6, max_iters: int = 300)
```

**Parameters:**
- `n_clusters` (int): Number of centroids for KMeans
- `method` (str): Method for initializing centroids ("random" or "kmeans++")
- `tol` (float): Tolerance for convergence detection. Defaults to 1e-6
- `max_iters` (int): Maximum number of iterations. Defaults to 300

#### Attributes

- `n_clusters` (int): Number of centroids
- `centroids` (np.ndarray): Array of centroid vectors
- `_trained` (bool): Indicates if the model has been trained
- `method` (str): Method for centroid initialization
- `tol` (float): Tolerance for convergence detection
- `max_iters` (int): Maximum number of iterations
- `inertia_` (float): Within-cluster sum of squared distances after fitting
- `n_iter_` (int): Number of iterations run during fitting

#### Methods

##### `initiate_plus_plus(x: np.ndarray) -> np.ndarray`

Initialize centroids using the optimized KMeans++ algorithm.

**Parameters:**
- `x` (np.ndarray): Input data matrix

**Returns:**
- `np.ndarray`: Array of centroids for KMeans clustering

**Algorithm:**
1. Choose first centroid randomly from data points
2. For each subsequent centroid:
   - Compute squared distances to nearest existing centroid for each point
   - Choose next centroid with probability proportional to squared distance
3. Uses both JIT-optimized and vectorized NumPy implementations

##### `init_centroids(x: np.ndarray) -> None`

Initialize centroids for KMeans clustering using vectorized operations.

**Parameters:**
- `x` (np.ndarray): Input data matrix

**Raises:**
- `ValueError`: If the initialization method is not recognized

**Supported methods:**
- `"random"`: Vectorized random initialization within data bounds
- `"kmeans++"`: Smart initialization using KMeans++ algorithm

##### `fit(x: np.ndarray) -> None`

Train the optimized KMeans model using vectorized operations.

**Parameters:**
- `x` (np.ndarray): Input data matrix

**Raises:**
- `RuntimeError`: If the model has already been trained

**Algorithm:**
1. Initialize centroids using specified method
2. Iteratively assign points to clusters and update centroids
3. Check for convergence using both centroid shift and inertia change
4. Updates `inertia_` and `n_iter_` attributes upon completion

##### `predict(x: np.ndarray) -> np.ndarray`

Predict cluster labels using optimized vectorized operations.

**Parameters:**
- `x` (np.ndarray): Input data matrix

**Returns:**
- `np.ndarray`: Cluster labels for each data point

**Raises:**
- `RuntimeError`: If model must be fitted before prediction

## JIT-Optimized Functions

The module includes several JIT-compiled functions for improved performance:

### `euclidean_distance_squared_jit(x, y)`

JIT-optimized squared Euclidean distance calculation.

**Parameters:**
- `x`: First point
- `y`: Second point

**Returns:**
- `float`: Squared Euclidean distance

### `assign_clusters_jit(data, centroids)`

JIT-optimized cluster assignment.

**Parameters:**
- `data`: Data points array
- `centroids`: Centroids array

**Returns:**
- `np.ndarray`: Cluster assignments

### `update_centroids_jit(data, labels, n_clusters)`

JIT-optimized centroid update.

**Parameters:**
- `data`: Data points array
- `labels`: Current cluster assignments
- `n_clusters`: Number of clusters

**Returns:**
- `np.ndarray`: Updated centroids

### `compute_inertia_jit(data, labels, centroids)`

JIT-optimized inertia computation.

**Parameters:**
- `data`: Data points array
- `labels`: Cluster assignments
- `centroids`: Centroids array

**Returns:**
- `float`: Within-cluster sum of squared distances

### `kmeans_plus_plus_jit(data, n_clusters)`

JIT-optimized KMeans++ initialization.

**Parameters:**
- `data`: Input data matrix
- `n_clusters`: Number of clusters

**Returns:**
- `np.ndarray`: Initialized centroids

## Private Methods

### `_assign_clusters(x: np.ndarray) -> np.ndarray`

Assign each data point to the nearest centroid using optimized operations.

**Parameters:**
- `x` (np.ndarray): Input data matrix

**Returns:**
- `np.ndarray`: Cluster assignments for each data point

**Implementation:**
- Uses JIT-optimized implementation when Numba is available
- Falls back to vectorized NumPy implementation using broadcasting

### `_update_centroids(x: np.ndarray, labels: np.ndarray) -> np.ndarray`

Update centroids using optimized operations.

**Parameters:**
- `x` (np.ndarray): Input data matrix
- `labels` (np.ndarray): Current cluster assignments

**Returns:**
- `np.ndarray`: Updated centroids

**Implementation:**
- Uses JIT-optimized implementation when Numba is available
- Falls back to vectorized NumPy implementation
- Handles empty clusters by keeping old centroid

### `_compute_inertia(x: np.ndarray, labels: np.ndarray) -> float`

Compute within-cluster sum of squared distances (inertia).

**Parameters:**
- `x` (np.ndarray): Input data matrix
- `labels` (np.ndarray): Cluster assignments

**Returns:**
- `float`: Inertia value

## Usage Example

```python
import numpy as np
from modules.kmeans import KMeans

# Generate sample data
X = np.random.rand(1000, 2)

# Create and train KMeans model
kmeans = KMeans(n_clusters=5, method="kmeans++", tol=1e-6, max_iters=300)

# Fit the model
kmeans.fit(X)

# Make predictions
labels = kmeans.predict(X)

# Access model properties
print(f"Centroids: {kmeans.centroids}")
print(f"Inertia: {kmeans.inertia_}")
print(f"Iterations: {kmeans.n_iter_}")

# Use different initialization method
kmeans_random = KMeans(n_clusters=3, method="random")
kmeans_random.fit(X)
labels_random = kmeans_random.predict(X)
```

## Performance Notes

- When Numba is available, all core operations use JIT compilation for significant speedup
- Vectorized NumPy operations provide efficient fallback when Numba is not available
- The KMeans++ initialization uses probabilistic selection for better initial centroids
- Convergence detection uses both centroid movement and inertia change for robustness
- Memory allocation is minimized through in-place operations where possible