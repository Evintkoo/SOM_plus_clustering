# Initialization Module

## Overview

The `initialization.py` module provides various weight initialization methods for neural networks and clustering algorithms. These methods help improve convergence and performance by setting appropriate initial values for model parameters.

## Functions

### `initiate_zero(P: int, Q: int) -> np.ndarray`

ZerO Initialization for a single layer.

**Parameters:**
- `P` (int): Input dimension of the layer
- `Q` (int): Output dimension of the layer

**Returns:**
- `np.ndarray`: Initialized weight matrix W

**Algorithm:**
- If `P == Q`: Returns identity matrix
- If `P < Q`: Creates partial identity matrix with zeros padding
- If `P > Q`: Applies Hadamard matrix transformation with scaling

**Mathematical Formula:**
For P > Q: `W = c * H_m * I_star`
where:
- `c = 2^(-(m-1)/2)` (scaling factor)
- `H_m` is the Hadamard matrix of size 2^m
- `I_star` is the partial identity matrix

### `hadamard_matrix(m: int) -> np.ndarray`

Generates a Hadamard matrix of size 2^m.

**Parameters:**
- `m` (int): Power of 2 for the size of the Hadamard matrix

**Returns:**
- `np.ndarray`: Hadamard matrix H of size 2^m x 2^m

**Algorithm:**
Recursive construction:
- Base case (m=0): Returns [[1]]
- Recursive case: Constructs from previous Hadamard matrix using block structure

### `initiate_naive_sharding(X: np.ndarray, k: int) -> np.ndarray`

Initialize centroids using the naive sharding method.

**Parameters:**
- `X` (np.ndarray): The dataset, where each row is a data point
- `k` (int): The number of clusters

**Returns:**
- `np.ndarray`: Initialized centroids

**Algorithm:**
1. Compute the sum of each data point's features (composite values)
2. Sort data points by their composite values
3. Split the sorted data into k equal-sized shards
4. Compute the mean of each shard to determine centroids

**Edge Cases:**
- Handles empty shards by random duplication
- Handles k > number of data points
- Handles empty datasets

### `initiate_he(input_dim: int, output_dim: int) -> np.ndarray`

Initializes weights using He initialization.

**Parameters:**
- `input_dim` (int): Number of input units (neurons in previous layer)
- `output_dim` (int): Number of output units (neurons in current layer)

**Returns:**
- `np.ndarray`: Initialized weights of shape (output_dim, input_dim)

**Mathematical Formula:**
```
stddev = sqrt(2.0 / input_dim)
weights ~ N(0, stddev²)
```

**Usage:**
Particularly effective for ReLU activation functions and deep networks.

### `initiate_lecun(input_shape: int, output_shape: int) -> np.ndarray`

Initialize weights using LeCun initialization.

**Parameters:**
- `input_shape` (int): Number of input units
- `output_shape` (int): Number of output units

**Returns:**
- `np.ndarray`: Initialized weights

**Mathematical Formula:**
```
stddev = sqrt(1.0 / fan_in)
weights ~ N(0, stddev²)
```

**Usage:**
Suitable for tanh and sigmoid activation functions.

### `svd_orthonormal(shape: Tuple[int, int]) -> np.ndarray`

Generate an orthonormal matrix using Singular Value Decomposition (SVD).

**Parameters:**
- `shape` (Tuple[int, int]): Shape must have exactly 2 dimensions (input_dim, output_dim)

**Returns:**
- `np.ndarray`: Orthonormal matrix

**Raises:**
- `ValueError`: If shape doesn't have exactly 2 dimensions

**Algorithm:**
1. Generate random matrix with standard normal distribution
2. Apply SVD decomposition
3. Return U or V matrix depending on shape requirements

### `initiate_lsuv(input_dim: int, output_dim: int, X_batch: np.ndarray, tol: float = 0.1, max_iter: int = 10) -> np.ndarray`

Initialize weights using LSUV (Layer-wise Sequential Unit-Variance) initialization.

**Parameters:**
- `input_dim` (int): Number of input units
- `output_dim` (int): Number of output units
- `X_batch` (np.ndarray): Batch of input data for variance adjustment
- `tol` (float): Tolerance for variance convergence. Defaults to 0.1
- `max_iter` (int): Maximum number of iterations for variance adjustment. Defaults to 10

**Returns:**
- `np.ndarray`: Initialized weight matrix with LSUV

**Algorithm:**
1. Initialize weights with orthonormal matrix using SVD
2. Iteratively adjust weights to achieve unit variance in activations
3. Check convergence within tolerance
4. Handle degenerate cases with He initialization fallback

**Features:**
- Ensures unit variance in layer activations
- Handles degenerate data cases
- Convergence-based iteration control

## Usage Examples

### Basic Initialization

```python
import numpy as np
from modules.initialization import initiate_he, initiate_lecun, initiate_zero

# He initialization for ReLU networks
weights_he = initiate_he(input_dim=784, output_dim=256)

# LeCun initialization for tanh networks  
weights_lecun = initiate_lecun(input_shape=256, output_shape=128)

# Zero initialization
weights_zero = initiate_zero(P=100, Q=50)
```

### Clustering Initialization

```python
import numpy as np
from modules.initialization import initiate_naive_sharding

# Generate sample data
X = np.random.rand(1000, 2)

# Initialize centroids using naive sharding
centroids = initiate_naive_sharding(X, k=5)
print(f"Initialized {len(centroids)} centroids")
```

### LSUV Initialization

```python
import numpy as np
from modules.initialization import initiate_lsuv

# Prepare batch data
X_batch = np.random.randn(100, 784)

# LSUV initialization with variance control
weights_lsuv = initiate_lsuv(
    input_dim=784, 
    output_dim=256, 
    X_batch=X_batch,
    tol=0.05,
    max_iter=15
)

# Check resulting activation variance
activations = X_batch @ weights_lsuv
print(f"Activation variance: {np.var(activations):.4f}")
```

### Advanced Usage with Shape Handling

```python
import numpy as np
from modules.initialization import svd_orthonormal, hadamard_matrix

# Generate orthonormal matrix
ortho_matrix = svd_orthonormal((512, 256))

# Generate Hadamard matrix for specific power of 2
hadamard = hadamard_matrix(m=8)  # 256x256 matrix
print(f"Hadamard matrix shape: {hadamard.shape}")
```

## Mathematical Background

### He Initialization
Designed for ReLU activations to maintain signal propagation:
- Variance: `Var(W) = 2/n_in`
- Prevents vanishing/exploding gradients in deep networks

### LeCun Initialization  
Designed for symmetric activations (tanh, sigmoid):
- Variance: `Var(W) = 1/n_in`
- Maintains signal strength through layers

### LSUV Initialization
Ensures unit variance in pre-activations:
- Iteratively scales weights: `W = W / sqrt(Var(activations))`
- Combines orthogonal initialization with variance normalization

### Naive Sharding
Distributes initial centroids based on feature sums:
- Sorts data by composite feature values
- Ensures even distribution across feature space
- Reduces initialization bias in clustering

## Performance Considerations

- **He/LeCun**: Fast single-pass initialization
- **LSUV**: Requires forward passes, slower but more precise
- **Naive Sharding**: O(n log n) due to sorting, but robust for clustering
- **SVD Orthonormal**: Uses SVD decomposition, moderate computational cost
- **Hadamard**: Recursive construction, exponential size growth