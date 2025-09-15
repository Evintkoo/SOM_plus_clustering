# KDE Kernel Module

## Overview

The `kde_kernel.py` module implements Kernel Density Estimation (KDE) with neuron selection functionality. This custom implementation provides KDE calculation without relying on external libraries like scipy, using manually constructed Gaussian kernels for multidimensional density estimation.

## Key Features

- **Custom Gaussian Kernel Implementation**: Manual multidimensional Gaussian kernel computation
- **Local Maxima Detection**: Identifies high-density regions in the estimated distribution
- **Neuron Selection**: Iterative selection of representative points with optimal spatial coverage
- **Adaptive Bandwidth Estimation**: Automatic bandwidth calculation based on data characteristics
- **Parallel Processing Support**: Uses joblib for parallel computations

## Functions

### `gaussian_kernel(x: np.ndarray, xi: np.ndarray, bandwidth: float) -> float`

Multidimensional Gaussian kernel function for density estimation.

**Parameters:**
- `x` (np.ndarray): Point where the kernel is evaluated (D-dimensional vector)
- `xi` (np.ndarray): Data point from the dataset (D-dimensional vector)  
- `bandwidth` (float): Smoothing bandwidth parameter

**Returns:**
- `float`: Kernel value for the point x and data point xi

**Mathematical Formula:**
```
K(x, xi) = (1 / (√(2π)^d * h^d)) * exp(-0.5 * ||x - xi||² / h²)
```

where:
- `d` is the dimensionality
- `h` is the bandwidth
- `||x - xi||²` is the squared Euclidean distance

**Features:**
- Normalized Gaussian kernel ensuring proper probability density
- Numerical stability with appropriate scaling
- Efficient computation for multidimensional data

### `kde_multidimensional(data: np.ndarray, points: np.ndarray, bandwidth: float) -> np.ndarray`

Multidimensional KDE using manually defined Gaussian kernel.

**Parameters:**
- `data` (np.ndarray): Input dataset, shape (N, D) where N is number of points, D is dimensionality
- `points` (np.ndarray): Evaluation points where KDE will be computed, shape (M, D)
- `bandwidth` (float): Bandwidth parameter for smoothing

**Returns:**
- `np.ndarray`: KDE values evaluated at the specified points, shape (M,)

**Algorithm:**
1. For each evaluation point:
   - Sum kernel contributions from all data points
   - Normalize by number of data points
2. Return density estimates at all evaluation points

**Computational Complexity:**
- Time: O(N × M × D) where N is data size, M is evaluation points, D is dimensions
- Space: O(M) for storing results

### `find_local_maxima(kde_values: np.ndarray, points: np.ndarray) -> np.ndarray`

Identifies local maxima in the KDE results representing high-density regions.

**Parameters:**
- `kde_values` (np.ndarray): KDE values computed at each point
- `points` (np.ndarray): Corresponding points where KDE was evaluated

**Returns:**
- `np.ndarray`: Points corresponding to local maxima

**Algorithm:**
1. Iterate through KDE values (excluding boundaries)
2. For each point, check if it's greater than both neighbors
3. Collect points where local maximum condition is satisfied

**Note:**
- Currently implements 1D local maxima detection
- Can be extended for multidimensional local maxima detection

### `bandwidth_estimator(data: np.ndarray) -> float`

Automatic bandwidth estimation based on data characteristics.

**Parameters:**
- `data` (np.ndarray): Dataset for bandwidth estimation, shape (n_samples,)

**Returns:**
- `float`: Estimated bandwidth value

**Raises:**
- `ValueError`: If data contains fewer than 2 points

**Mathematical Formula:**
```
h = (max(data) - min(data)) / (1 + log₂(n))
```

where:
- `n` is the number of data points
- Formula balances between under-smoothing and over-smoothing

**Features:**
- Adaptive to data range and sample size
- Prevents over-smoothing with large datasets
- Ensures reasonable smoothing for small datasets

### `initiate_kde(x: np.ndarray, n_neurons: int, bandwidth: Union[float, None] = None) -> np.ndarray`

Main function that initiates KDE-based neuron selection.

**Parameters:**
- `x` (np.ndarray): Input dataset, shape (n_samples, n_features)
- `n_neurons` (int): Number of neurons to select
- `bandwidth` (Union[float, None]): Bandwidth for KDE. If None, uses automatic estimation

**Returns:**
- `np.ndarray`: Array of selected neurons (representative points)

**Raises:**
- `ValueError`: If maximum number of available neurons is less than or equal to `n_neurons`

**Algorithm:**
1. **Bandwidth Estimation**: Use automatic estimation if not provided
2. **KDE Computation**: Calculate density at all data points
3. **Local Maxima Detection**: Find high-density regions
4. **Neuron Selection**: Use farthest-first traversal for optimal coverage

**Selection Process:**
1. Start with random neuron from local maxima
2. Iteratively select neuron farthest from already selected set
3. Use precomputed distance matrix for efficiency
4. Update minimum distances incrementally

## Implementation Details

### Optimization Strategies

#### Distance Matrix Precomputation
```python
# Precompute all pairwise squared distances
dist_matrix = np.sum((local_max[:, np.newaxis, :] - local_max[np.newaxis, :, :]) ** 2, axis=-1)
```

#### Incremental Distance Updates
```python
# Update minimum distances after each selection
min_dist_to_selected = np.minimum(min_dist_to_selected, dist_matrix[next_neuron])
```

#### Memory Efficient Selection
- Uses boolean masks to track selected neurons
- Avoids redundant distance computations
- Squared distances to avoid expensive square root operations

### Bandwidth Selection Guidelines

- **Small bandwidth**: More detailed density estimation, risk of overfitting
- **Large bandwidth**: Smoother estimation, risk of over-smoothing
- **Automatic estimation**: Balances detail and smoothness based on data characteristics

## Usage Examples

### Basic KDE Usage

```python
import numpy as np
from modules.kde_kernel import kde_multidimensional, gaussian_kernel, bandwidth_estimator

# Generate sample data
data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 200)

# Estimate bandwidth
bandwidth = bandwidth_estimator(data.flatten())
print(f"Estimated bandwidth: {bandwidth:.4f}")

# Evaluate KDE at data points
kde_values = kde_multidimensional(data, data, bandwidth)
print(f"KDE values range: [{kde_values.min():.4f}, {kde_values.max():.4f}]")
```

### Neuron Selection for Clustering

```python
import numpy as np
from modules.kde_kernel import initiate_kde

# Generate clustered data
n_samples = 1000
n_features = 3
X = np.random.rand(n_samples, n_features)

# Select representative neurons
n_neurons = 50
try:
    neurons = initiate_kde(X, n_neurons=n_neurons)
    print(f"Selected {len(neurons)} neurons")
    print(f"Neuron shape: {neurons.shape}")
except ValueError as e:
    print(f"Error: {e}")
```

### Custom Bandwidth Selection

```python
import numpy as np
from modules.kde_kernel import initiate_kde

# Generate data with known structure
X = np.vstack([
    np.random.normal([2, 2], 0.5, (300, 2)),
    np.random.normal([-2, -2], 0.5, (300, 2)),
    np.random.normal([2, -2], 0.5, (300, 2))
])

# Use custom bandwidth
custom_bandwidth = 0.3
neurons = initiate_kde(X, n_neurons=20, bandwidth=custom_bandwidth)
print(f"Selected neurons with custom bandwidth: {neurons.shape}")
```

### Local Maxima Detection

```python
import numpy as np
from modules.kde_kernel import kde_multidimensional, find_local_maxima

# 1D example for local maxima
x_1d = np.random.normal(0, 1, 1000)
x_eval = np.linspace(-4, 4, 100).reshape(-1, 1)
x_1d_reshaped = x_1d.reshape(-1, 1)

# Compute KDE
kde_vals = kde_multidimensional(x_1d_reshaped, x_eval, bandwidth=0.3)

# Find local maxima
maxima = find_local_maxima(kde_vals, x_eval)
print(f"Found {len(maxima)} local maxima")
```

## Mathematical Background

### Kernel Density Estimation

KDE estimates the probability density function of a dataset using:

```
f̂(x) = (1/n) Σᵢ₌₁ⁿ K((x - xᵢ)/h)
```

where:
- `n` is the number of data points
- `K` is the kernel function (Gaussian in this implementation)
- `h` is the bandwidth parameter
- `xᵢ` are the data points

### Gaussian Kernel Properties

- **Symmetry**: K(u) = K(-u)
- **Normalization**: ∫ K(u) du = 1
- **Non-negativity**: K(u) ≥ 0 for all u
- **Unimodality**: Single peak at u = 0

### Bandwidth Selection Theory

The automatic bandwidth estimator uses:

```
h = (max - min) / (1 + log₂(n))
```

This heuristic:
- Scales with data range (max - min)
- Decreases with sample size (log₂(n) term)
- Provides reasonable default for most datasets

## Performance Characteristics

### Time Complexity
- **KDE computation**: O(N × M × D)
- **Local maxima detection**: O(M)
- **Neuron selection**: O(K² × D) where K is number of maxima
- **Overall**: O(N × M × D + K²)

### Memory Complexity
- **Distance matrix**: O(K²) 
- **KDE values**: O(M)
- **Selected neurons**: O(n_neurons × D)

### Optimization Tips

1. **Reduce evaluation points**: Use subset of data for KDE evaluation
2. **Parallel processing**: Leverage joblib for large datasets
3. **Bandwidth tuning**: Experiment with different bandwidth values
4. **Early termination**: Stop if sufficient local maxima found

## Limitations and Considerations

1. **Curse of dimensionality**: KDE performance degrades in high dimensions
2. **Computational cost**: O(N²) complexity for large datasets
3. **Bandwidth sensitivity**: Results highly dependent on bandwidth choice
4. **Local maxima detection**: Currently limited to 1D case
5. **Memory usage**: Distance matrix storage for large neuron sets

## Future Enhancements

1. **Multidimensional local maxima**: Extend detection to higher dimensions
2. **Adaptive bandwidth**: Different bandwidth per dimension or region
3. **Alternative kernels**: Support for other kernel functions
4. **GPU acceleration**: CuPy integration for large-scale computations
5. **Incremental KDE**: Online learning for streaming data