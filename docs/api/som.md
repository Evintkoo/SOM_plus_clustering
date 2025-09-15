# SOM Module

## Overview

The `som.py` module implements the Self-Organizing Map (SOM) algorithm for unsupervised learning. This is a highly optimized implementation that includes GPU acceleration with CuPy, JIT compilation with Numba, and parallel processing with Taichi.

## Performance Optimizations

- **Numba JIT compilation** for distance calculations and non-parallel operations
- **Taichi kernels** for parallel SOM operations
- **CuPy GPU acceleration** for large-scale computations
- **Adaptive backend selection** based on data size and hardware availability

## Classes

### SOM

Self-Organizing Map (SOM) implementation with GPU acceleration using CuPy.

#### Constructor

```python
SOM(m: int, n: int, dim: int, initiate_method: str, learning_rate: float, 
    neighbour_rad: int, distance_function: str, max_iter: Union[int, float] = np.inf,
    backend: str = "auto")
```

**Parameters:**
- `m` (int): Height of the grid
- `n` (int): Width of the grid  
- `dim` (int): Dimensionality of input data
- `initiate_method` (str): Method for neuron initialization (see Variables module for options)
- `learning_rate` (float): Initial learning rate (should be in (0,1])
- `neighbour_rad` (int): Initial neighbourhood radius
- `distance_function` (str): Distance function ("euclidean" or "cosine")
- `max_iter` (int, optional): Maximum number of iterations. Defaults to np.inf
- `backend` (str): Backend to use ("auto", "cupy", "taichi", "numba", "numpy")

#### Methods

##### `initiate_neuron(data: np.ndarray) -> Any`

Initialize neuron weights using the specified method.

**Parameters:**
- `data` (np.ndarray): Input data on CPU

**Returns:**
- `cp.ndarray`: Neuron weights on GPU

**Supported initialization methods:**
- `"random"`: Random uniform initialization
- `"kde"`: Kernel Density Estimation initialization
- `"kmeans"`, `"kde_kmeans"`, `"kmeans++"`: K-means based initialization
- `"som++"`: SOM++ initialization (farthest-first traversal)
- `"zero"`: Zero initialization
- `"he"`: He initialization
- `"naive_sharding"`: Naive sharding initialization
- `"lecun"`: LeCun initialization
- `"lsuv"`: Layer-wise Sequential Unit-Variance initialization

##### `index_bmu(x: Any) -> Tuple[int, int]`

Find the index of the best matching unit (BMU) among all neurons.

**Parameters:**
- `x` (cp.ndarray): Input data point (on GPU/CPU)

**Returns:**
- `Tuple[int, int]`: The indices (row, column) of the BMU

##### `fit(x: np.ndarray, epoch: int, shuffle: bool = True, batch_size: int = None) -> None`

Fit the SOM to the input data using optimized batch processing.

**Parameters:**
- `x` (np.ndarray): Input data (CPU, NumPy array)
- `epoch` (int): Number of epochs
- `shuffle` (bool, optional): Whether to shuffle data each epoch. Defaults to True
- `batch_size` (int, optional): Batch size for mini-batch processing. If None, uses adaptive sizing

**Features:**
- Mini-batch processing for better memory management
- Adaptive batch sizing
- Exponential decay for learning rate and neighborhood radius
- Input validation for NaN and infinite values

##### `predict(x: np.ndarray) -> np.ndarray`

Predict cluster labels for the input data.

**Parameters:**
- `x` (np.ndarray): Input data (CPU, NumPy array)

**Returns:**
- `np.ndarray`: Predicted cluster labels

##### `fit_predict(x: np.ndarray, epoch: int, shuffle: bool = True, batch_size: int = None) -> np.ndarray`

Fit the SOM and then predict cluster labels for the input data.

**Parameters:**
- `x` (np.ndarray): Input data (CPU, NumPy array)
- `epoch` (int): Number of epochs
- `shuffle` (bool, optional): Whether to shuffle data each epoch
- `batch_size` (int, optional): Batch size for mini-batch processing

**Returns:**
- `np.ndarray`: Predicted cluster labels

##### `evaluate(x: np.ndarray, method: List[str]) -> Union[List[float], dict]`

Evaluate the SOM clustering using various metrics.

**Parameters:**
- `x` (np.ndarray): Input data (CPU, NumPy array)
- `method` (List[str]): List of evaluation methods

**Returns:**
- `Union[List[float], dict]`: Evaluation scores

**Available evaluation methods:**
- `"silhouette"`: Silhouette coefficient
- `"davies_bouldin"`: Davies-Bouldin index
- `"calinski_harabasz"`: Calinski-Harabasz index  
- `"dunn"`: Dunn index
- `"all"`: Returns all metrics as a dictionary

##### `save(path: str) -> None`

Save the SOM model to a file using pickle.

**Parameters:**
- `path` (str): File path to save the model

##### `load(path: str) -> 'SOM'` (class method)

Load a SOM model from a file.

**Parameters:**
- `path` (str): File path to load the model from

**Returns:**
- `SOM`: Loaded SOM instance

#### Properties

##### `cluster_center_`

Get the cluster centers as a NumPy array.

**Returns:**
- `np.ndarray`: Cluster centers with shape (m*n, dim)

## Utility Functions

### `validate_configuration(initiate_method: str, learning_rate: float, distance_function: str) -> None`

Validate input parameters for SOM initialization.

**Parameters:**
- `initiate_method` (str): Initialization method
- `learning_rate` (float): Learning rate 
- `distance_function` (str): Distance function

**Raises:**
- `ValueError`: If parameters are invalid

### `initiate_plus_plus(m: int, n: int, x: np.ndarray) -> np.ndarray`

Initialize centroids using SOM++-style farthest-first traversal.

**Parameters:**
- `m` (int): Grid height
- `n` (int): Grid width
- `x` (np.ndarray): Input data

**Returns:**
- `np.ndarray`: Initialized centroids

## JIT-Optimized Functions

The module includes several JIT-compiled functions for improved performance:

- `euclidean_distance_jit`: Fast Euclidean distance calculation
- `euclidean_distance_squared_jit`: Fast squared Euclidean distance
- `cosine_distance_jit`: Fast cosine distance calculation
- `batch_euclidean_distances_jit`: Batch distance computation
- `batch_cosine_distances_jit`: Batch cosine distance computation
- `neighborhood_function_jit`: Neighborhood function computation

## Taichi Kernels

For parallel processing, the module includes Taichi kernels:

- `taichi_batch_euclidean_distances`: Parallel batch distance computation
- `taichi_neighborhood_update`: Parallel neighborhood update

## Backend Selection

The SOM implementation automatically selects the best available backend:

1. **CuPy**: GPU acceleration (preferred if available)
2. **Taichi**: Parallel CPU/GPU processing
3. **Numba**: JIT compilation for CPU
4. **NumPy**: Fallback pure Python implementation

## Usage Example

```python
import numpy as np
from modules.som import SOM

# Generate sample data
X = np.random.rand(1000, 10)

# Create and train SOM
som = SOM(m=10, n=10, dim=10, initiate_method="kmeans++", 
          learning_rate=0.5, neighbour_rad=3, distance_function="euclidean")

# Fit the model
som.fit(X, epoch=100)

# Make predictions
labels = som.predict(X)

# Evaluate the model
scores = som.evaluate(X, method=["silhouette", "davies_bouldin"])
print(f"Silhouette Score: {scores[0]}")
print(f"Davies-Bouldin Index: {scores[1]}")

# Save the model
som.save("som_model.pkl")
```