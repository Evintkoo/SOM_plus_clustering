# Utils Module

## Overview

The `utils.py` module provides utility functions for various mathematical operations and data processing tasks commonly used in machine learning and data analysis. The module includes performance optimizations through Numba JIT compilation and CuPy GPU acceleration.

## Performance Optimizations

- **Numba JIT compilation** for mathematical operations
- **CuPy GPU acceleration** when available
- **Optimized algorithms** for better numerical stability
- **Efficient memory usage** with in-place operations where possible

## Functions

### `find_most_edge_point(points: Any) -> Any`

Find the point farthest from the center of the dataset.

**Parameters:**
- `points` (cp.ndarray or np.ndarray): Array of points

**Returns:**
- `cp.ndarray` or `np.ndarray`: The point farthest from the center

**Algorithm:**
1. Calculate the center (mean) of all points
2. Compute distances from each point to the center
3. Return the point with maximum distance

**Performance Features:**
- Uses JIT-optimized implementation when Numba is available
- Automatically handles CuPy/NumPy array conversion
- Vectorized distance computation for efficiency

**Mathematical Formula:**
```
center = (1/n) Σᵢ₌₁ⁿ xᵢ
edge_point = argmax(||xᵢ - center||²)
```

### `cos_distance(vector1: List[float], vector2: List[float]) -> float`

Calculate the cosine distance between two vectors.

**Parameters:**
- `vector1` (List[float]): First vector
- `vector2` (List[float]): Second vector

**Returns:**
- `float`: Cosine distance between the vectors

**Raises:**
- `ValueError`: If vectors have different lengths

**Algorithm:**
1. Compute dot product of the vectors
2. Calculate L2 norms of both vectors
3. Compute cosine similarity: `dot_product / (norm1 * norm2)`
4. Return cosine distance: `1 - cosine_similarity`

**Mathematical Formula:**
```
cos_distance = 1 - (v1 · v2) / (||v1|| * ||v2||)
```

**Features:**
- Numerical stability with epsilon values (1e-12)
- Cosine similarity clipping to valid range [-1, 1]
- JIT optimization available with Numba
- Automatic backend selection (CuPy/NumPy)

### `random_initiate(dim: int, min_val: float, max_val: float) -> Any`

Generate an array of random numbers within a specified range.

**Parameters:**
- `dim` (int): Dimension of the array
- `min_val` (float): Minimum value of random numbers
- `max_val` (float): Maximum value of random numbers

**Returns:**
- `cp.ndarray` or `np.ndarray`: Array of randomly generated numbers

**Usage:**
Commonly used for initializing weights or generating test data within specific bounds.

### `euc_distance(point1: Any, point2: Any) -> float`

Calculate the Euclidean distance between two points in n-dimensional space.

**Parameters:**
- `point1` (cp.ndarray or np.ndarray): Coordinates of the first point
- `point2` (cp.ndarray or np.ndarray): Coordinates of the second point

**Returns:**
- `float`: Euclidean distance between the points

**Raises:**
- `ValueError`: If point dimensions are not equal

**Algorithm:**
- Uses JIT-optimized implementation when Numba is available
- Falls back to `cp.linalg.norm()` for CuPy/NumPy arrays
- Automatic device handling for GPU/CPU arrays

**Mathematical Formula:**
```
euclidean_distance = √(Σᵢ₌₁ⁿ (p1ᵢ - p2ᵢ)²)
```

### `one_hot_encode(y: Any) -> Any`

One-hot encode an array of labels with GPU optimization.

**Parameters:**
- `y` (cp.ndarray or np.ndarray): Array of integer-like labels to encode

**Returns:**
- `cp.ndarray` or `np.ndarray`: One-hot encoded array with shape (N, C)

**Algorithm:**
1. Flatten input if necessary
2. Find unique classes and create inverse mapping
3. Create output array of zeros
4. Use advanced indexing for efficient assignment

**Features:**
- Fully vectorized GPU implementation
- Memory efficient with appropriate dtype (float32)
- Handles arbitrary label values through unique mapping
- Optimized advanced indexing for assignment

**Example:**
```python
y = np.array([0, 1, 2, 1, 0])
encoded = one_hot_encode(y)
# Result: [[1,0,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0]]
```

### `normalize_column(data: Any, column_index: int) -> Any`

Normalize a specific column in an array using min-max scaling.

**Parameters:**
- `data` (cp.ndarray or np.ndarray): The data array
- `column_index` (int): Index of the column to normalize

**Returns:**
- `cp.ndarray` or `np.ndarray`: The normalized column

**Raises:**
- `IndexError`: If column_index is out of bounds

**Algorithm:**
1. Extract the specified column
2. Find minimum and maximum values
3. Apply min-max normalization: `(x - min) / (max - min)`
4. Handle degenerate cases (zero range)

**Mathematical Formula:**
```
normalized = (column - min(column)) / (max(column) - min(column))
```

**Features:**
- Numerical stability check for very small ranges
- Returns zeros for degenerate data (range < 1e-9)
- Bounds checking for column index validity

## JIT-Optimized Functions

The module includes several JIT-compiled functions for improved performance:

### `euclidean_distance_jit(point1, point2)`

JIT-optimized Euclidean distance calculation.

**Features:**
- `nopython=True` for maximum performance
- `fastmath=True` for aggressive optimization
- Manual loop implementation for speed

### `cosine_distance_jit(vector1, vector2)`

JIT-optimized cosine distance calculation.

**Features:**
- Combined dot product and norm computation
- Numerical stability with epsilon values
- Cosine similarity clipping for valid range

### `find_most_edge_point_jit(points)`

JIT-optimized function to find the point farthest from center.

**Features:**
- Manual center calculation for speed
- Squared distance comparison (avoids square root)
- Efficient loop structure for large datasets

## Backend Selection

The module automatically selects the best available backend:

1. **Numba JIT**: Used when available for CPU optimization
2. **CuPy**: GPU acceleration for large-scale operations
3. **NumPy**: Fallback pure Python implementation

## Usage Examples

### Basic Distance Calculations

```python
import numpy as np
from modules.utils import euc_distance, cos_distance

# Euclidean distance
point1 = np.array([1.0, 2.0, 3.0])
point2 = np.array([4.0, 5.0, 6.0])
dist = euc_distance(point1, point2)
print(f"Euclidean distance: {dist:.3f}")

# Cosine distance
vector1 = [1.0, 2.0, 3.0]
vector2 = [2.0, 4.0, 6.0]
cos_dist = cos_distance(vector1, vector2)
print(f"Cosine distance: {cos_dist:.3f}")
```

### Data Preprocessing

```python
import numpy as np
from modules.utils import one_hot_encode, normalize_column

# One-hot encoding
labels = np.array([0, 1, 2, 1, 0, 2])
encoded = one_hot_encode(labels)
print(f"Encoded shape: {encoded.shape}")
print(f"Encoded data:\n{encoded}")

# Column normalization
data = np.random.rand(100, 5) * 100  # Random data 0-100
normalized_col = normalize_column(data, column_index=2)
print(f"Column 2 range after normalization: [{normalized_col.min():.3f}, {normalized_col.max():.3f}]")
```

### Edge Point Detection

```python
import numpy as np
from modules.utils import find_most_edge_point

# Generate clustered data
cluster1 = np.random.normal([0, 0], 1, (50, 2))
cluster2 = np.random.normal([5, 5], 1, (50, 2))
data = np.vstack([cluster1, cluster2])

# Find edge point
edge_point = find_most_edge_point(data)
print(f"Edge point: {edge_point}")

# Verify it's actually an edge point
center = np.mean(data, axis=0)
distances = np.linalg.norm(data - center, axis=1)
max_dist_idx = np.argmax(distances)
print(f"Verification - max distance point: {data[max_dist_idx]}")
```

### Random Initialization

```python
import numpy as np
from modules.utils import random_initiate

# Initialize weights for a neural network layer
input_dim = 784
hidden_dim = 256

# Random initialization between -1 and 1
weights = random_initiate(dim=(input_dim, hidden_dim), min_val=-1.0, max_val=1.0)
print(f"Weight matrix shape: {weights.shape}")
print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
```

### Performance Comparison

```python
import numpy as np
import time
from modules.utils import euc_distance, cos_distance

# Generate large test data
n_points = 10000
dim = 100
points1 = np.random.rand(n_points, dim)
points2 = np.random.rand(n_points, dim)

# Time Euclidean distance calculation
start_time = time.time()
distances = [euc_distance(p1, p2) for p1, p2 in zip(points1[:1000], points2[:1000])]
euclidean_time = time.time() - start_time

print(f"Euclidean distance computation time: {euclidean_time:.3f}s")

# Time cosine distance calculation
start_time = time.time()
cos_distances = [cos_distance(p1.tolist(), p2.tolist()) for p1, p2 in zip(points1[:1000], points2[:1000])]
cosine_time = time.time() - start_time

print(f"Cosine distance computation time: {cosine_time:.3f}s")
```

### GPU Acceleration Example

```python
import numpy as np
try:
    import cupy as cp
    from modules.utils import find_most_edge_point, one_hot_encode
    
    # Generate data on GPU
    gpu_data = cp.random.rand(10000, 3)
    
    # Find edge point on GPU
    edge_point = find_most_edge_point(gpu_data)
    print(f"GPU edge point: {edge_point}")
    
    # One-hot encoding on GPU
    labels = cp.random.randint(0, 5, 1000)
    encoded = one_hot_encode(labels)
    print(f"GPU encoded shape: {encoded.shape}")
    
except ImportError:
    print("CuPy not available, using CPU implementation")
```

## Performance Notes

### JIT Compilation Benefits
- **First Call**: Compilation overhead (slower)
- **Subsequent Calls**: Significant speedup (2-10x typical)
- **Large Data**: Maximum benefit with substantial computations

### GPU Acceleration
- **Memory Transfer**: Consider CPU-GPU transfer costs
- **Batch Size**: Larger batches better utilize GPU
- **Problem Size**: GPU beneficial for large-scale operations

### Numerical Stability
- **Epsilon Values**: Prevent division by zero
- **Range Clipping**: Maintain valid mathematical ranges
- **Precision**: Use appropriate floating-point precision

## Error Handling

The module includes comprehensive error handling:

- **Dimension Validation**: Check array shapes before operations
- **Bounds Checking**: Validate array indices
- **Type Checking**: Ensure compatible data types
- **Range Validation**: Check for valid numerical ranges

## Memory Considerations

- **In-place Operations**: Used where possible to reduce memory
- **Array Copying**: Automatic handling for different backends
- **Large Arrays**: Consider chunking for memory-limited systems
- **GPU Memory**: Monitor VRAM usage for large-scale operations