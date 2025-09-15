"""
This module implements the Self-Organizing Map (SOM) algorithm for unsupervised learning.
It includes methods for initializing the SOM, training it on input data,
and evaluating the results.

Performance optimizations:
- Numba JIT compilation for distance calculations and non-parallel operations
- Taichi kernels for parallel SOM operations
- CuPy GPU acceleration
- Adaptive backend selection based on data size and hardware availability
"""
import math
import pickle
from typing import List, Tuple, Union, Any, Optional

import numpy as np
import importlib
try:
    cp = importlib.import_module('cupy')  # type: ignore
    _USING_CUPY = True
except Exception:  # pragma: no cover - allow CPU-only environments
    cp = np  # type: ignore
    _USING_CUPY = False
    # Provide small compatibility shims
    if not hasattr(cp, "asarray"):
        cp.asarray = np.asarray  # type: ignore
    if not hasattr(cp, "asnumpy"):
        cp.asnumpy = np.asarray  # type: ignore

# JIT and parallel processing imports
try:
    from numba import jit, prange
    import numba
    _USING_NUMBA = True
except ImportError:
    _USING_NUMBA = False
    # Fallback decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(x):
        return range(x)

try:
    import taichi as ti
    ti.init(arch=ti.cpu)  # Initialize with CPU backend, can be changed to GPU if available
    _USING_TAICHI = True
except ImportError:
    _USING_TAICHI = False

from .initialization import *
from .evals import (
    silhouette_score, davies_bouldin_index, calinski_harabasz_score,
    dunn_index #, compare_distribution, bcubed_precision_recall
)
from .kde_kernel import initiate_kde
from .kmeans import KMeans
from .variables import INITIATION_METHOD_LIST, DISTANCE_METHOD_LIST, EVAL_METHOD_LIST

# JIT-optimized distance calculation functions
@jit(nopython=True, fastmath=True)
def euclidean_distance_jit(x, y):
    """JIT-optimized Euclidean distance calculation."""
    diff = x - y
    return np.sqrt(np.sum(diff * diff))

@jit(nopython=True, fastmath=True)
def euclidean_distance_squared_jit(x, y):
    """JIT-optimized squared Euclidean distance calculation."""
    diff = x - y
    return np.sum(diff * diff)

@jit(nopython=True, fastmath=True)
def cosine_distance_jit(x, y):
    """JIT-optimized cosine distance calculation."""
    norm_x = np.sqrt(np.sum(x * x)) + 1e-12
    norm_y = np.sqrt(np.sum(y * y)) + 1e-12
    dot_product = np.sum(x * y)
    cosine_sim = dot_product / (norm_x * norm_y)
    return 1.0 - cosine_sim

@jit(nopython=True, fastmath=True)
def batch_euclidean_distances_jit(data_batch, neurons_flat):
    """JIT-optimized batch distance computation for Euclidean distance."""
    batch_size = data_batch.shape[0]
    n_neurons = neurons_flat.shape[0]
    distances = np.empty((batch_size, n_neurons))
    
    for i in prange(batch_size):
        for j in range(n_neurons):
            distances[i, j] = euclidean_distance_squared_jit(data_batch[i], neurons_flat[j])
    
    return distances

@jit(nopython=True, fastmath=True)
def batch_cosine_distances_jit(data_batch, neurons_flat):
    """JIT-optimized batch distance computation for cosine distance."""
    batch_size = data_batch.shape[0]
    n_neurons = neurons_flat.shape[0]
    distances = np.empty((batch_size, n_neurons))
    
    for i in prange(batch_size):
        for j in range(n_neurons):
            distances[i, j] = cosine_distance_jit(data_batch[i], neurons_flat[j])
    
    return distances

@jit(nopython=True, fastmath=True)
def neighborhood_function_jit(dist_squared, learning_rate, radius_squared):
    """JIT-optimized neighborhood function computation."""
    return learning_rate * np.exp(-dist_squared / (2.0 * radius_squared))

# Taichi kernels for parallel operations
if _USING_TAICHI:
    @ti.kernel
    def taichi_batch_euclidean_distances(data_batch: ti.template(), 
                                       neurons_flat: ti.template(),
                                       distances: ti.template()):
        """Taichi kernel for parallel batch Euclidean distance computation."""
        batch_size, dim = data_batch.shape
        n_neurons = neurons_flat.shape[0]
        
        for i, j in ti.ndrange(batch_size, n_neurons):
            dist_sq = 0.0
            for k in range(dim):
                diff = data_batch[i, k] - neurons_flat[j, k]
                dist_sq += diff * diff
            distances[i, j] = dist_sq
    
    @ti.kernel
    def taichi_neighborhood_update(neurons: ti.template(),
                                 data_point: ti.template(),
                                 bmu_row: ti.i32,
                                 bmu_col: ti.i32,
                                 learning_rate: ti.f32,
                                 radius_sq: ti.f32):
        """Taichi kernel for parallel neighborhood update."""
        m, n, dim = neurons.shape
        
        for i, j in ti.ndrange(m, n):
            # Compute distance to BMU
            dist_sq = (i - bmu_row) ** 2 + (j - bmu_col) ** 2
            
            # Compute neighborhood influence
            h = learning_rate * ti.exp(-dist_sq / (2.0 * radius_sq))
            
            # Update neuron weights
            for k in range(dim):
                neurons[i, j, k] += h * (data_point[k] - neurons[i, j, k])
else:
    # Dummy functions if Taichi is not available
    def taichi_batch_euclidean_distances(*args, **kwargs):
        pass
    def taichi_neighborhood_update(*args, **kwargs):
        pass

def validate_configuration(initiate_method: str,
                        learning_rate: float,
                        distance_function: str) -> None:
    """Validate input parameters."""
    if learning_rate > 1.76:
        raise ValueError("Learning rate should be less than 1.76")
    if initiate_method not in INITIATION_METHOD_LIST:
        raise ValueError(f"Invalid initiation method: {initiate_method}")
    if distance_function not in DISTANCE_METHOD_LIST:
        raise ValueError(f"Invalid distance function: {distance_function}")

def initiate_plus_plus(m: int, n: int, x: np.ndarray) -> np.ndarray:
    """Initialize centroids using SOM++-style farthest-first traversal on CPU (NumPy).

    Notes:
        - Stays fully in NumPy to avoid device churn during init.
        - Starts from the point farthest from the mean (robust edge start).
    """
    k: int = m * n
    if x.ndim != 2:
        raise ValueError("x must be 2D array")
    # Start from farthest from mean (NumPy only)
    mean_pt = np.mean(x, axis=0)
    dist0 = np.sum((x - mean_pt) ** 2, axis=1)
    centroids: List[np.ndarray] = [x[int(np.argmax(dist0))]]
    dist_sq: np.ndarray = np.sum((x - centroids[0]) ** 2, axis=1)

    for _ in range(1, k):
        furthest_data_idx: int = int(np.argmax(dist_sq))
        c = x[furthest_data_idx]
        centroids.append(c)
        # Update min distance to the set of chosen centroids
        dist_sq = np.minimum(dist_sq, np.sum((x - c) ** 2, axis=1))

    return np.asarray(centroids)


class SOM:
    """
    Self-Organizing Map (SOM) implementation with GPU acceleration using CuPy.
    This version does not use Joblib for parallelism; all computation is performed using CuPy.
    """

    def __init__(self, m: int, n: int,
                 dim: int, initiate_method: str,
                 learning_rate: float, neighbour_rad: int,
                 distance_function: str, max_iter: Union[int, float] = np.inf,
                 backend: str = "auto"
                 ) -> None:
        """
        Initialize the SOM.

        Args:
            m (int): Height of the grid.
            n (int): Width of the grid.
            dim (int): Dimensionality of input data.
            initiate_method (str): Method for neuron initialization.
            learning_rate (float): Initial learning rate (should be in (0,1]).
            neighbour_rad (int): Initial neighbourhood radius.
            distance_function (str): Distance function ("euclidean" or "cosine").
            max_iter (int, optional): Maximum number of iterations. Defaults to np.inf.
            backend (str): Backend to use ("auto", "cupy", "taichi", "numba", "numpy").
        """
        # Validate configuration parameters (this should raise ValueError for invalid configurations)
        validate_configuration(initiate_method, learning_rate, distance_function)

        self.m: int = m
        self.n: int = n
        self.dim: int = dim
        self.shape: Tuple[int, int, int] = (m, n, dim)
        self.max_iter: Union[int, float] = max_iter if max_iter is not None else np.inf
        self.init_method: str = initiate_method
        self.dist_func: str = distance_function
        
        # Backend selection
        self.backend = self._select_backend(backend)

        self.cur_learning_rate: float = learning_rate
        self.initial_learning_rate: float = learning_rate

        self.cur_neighbour_rad: float = neighbour_rad
        self.initial_neighbour_rad: float = neighbour_rad

        # Neurons will be stored as a CuPy/NumPy array.
        self.neurons = cp.empty((0,))
        self.initial_neurons = cp.empty((0,))
        self._trained: bool = False
        # Precomputed grid coordinates for neighborhood calculations
        self._grid_rows: Optional[Any] = None
        self._grid_cols: Optional[Any] = None

    def _select_backend(self, backend: str) -> str:
        """Select the optimal backend based on availability and data characteristics."""
        if backend == "auto":
            if _USING_CUPY:
                return "cupy"
            elif _USING_TAICHI:
                return "taichi"
            elif _USING_NUMBA:
                return "numba"
            else:
                return "numpy"
        else:
            # Validate specific backend choice
            backend_available = {
                "cupy": _USING_CUPY,
                "taichi": _USING_TAICHI,
                "numba": _USING_NUMBA,
                "numpy": True
            }
            if backend not in backend_available:
                raise ValueError(f"Unknown backend: {backend}")
            if not backend_available[backend]:
                raise ValueError(f"Backend {backend} is not available")
            return backend

    def initiate_neuron(self, data: np.ndarray) -> Any:
        """
        Initialize neuron weights using the specified method.

        Args:
            data (np.ndarray): Input data on CPU.

        Returns:
            cp.ndarray: Neuron weights on GPU.
        """
        # Convert input data to GPU/CPU array.
        data_gpu: Any = cp.asarray(data)
        min_val: float = float(cp.min(data_gpu).item())
        max_val: float = float(cp.max(data_gpu).item())

        if self.init_method == "random":
            neurons = cp.random.uniform(min_val, max_val, self.shape)
            return neurons
        elif self.init_method == "kde":
            # Assume initiate_kde returns a NumPy array.
            neurons = initiate_kde(x=data, n_neurons=self.m * self.n)
            return cp.asarray(neurons).reshape(self.shape)
        elif self.init_method in ["kmeans", "kde_kmeans", "kmeans++"]:
            model = KMeans(n_clusters=(self.m * self.n), method=self.init_method)
            model.fit(x=data)
            return cp.asarray(model.centroids).reshape(self.shape)
        elif self.init_method == "som++":
            plus_plus_neurons = initiate_plus_plus(m=self.m, n=self.n, x=data)
            return cp.asarray(plus_plus_neurons).reshape(self.shape)
        elif self.init_method == "zero":
            neurons = initiate_zero(P=self.m * self.n, Q=self.dim)
            return cp.asarray(neurons).reshape(self.shape)
        elif self.init_method == "he":
            neurons = initiate_he(input_dim=self.dim, output_dim=self.m * self.n)
            return cp.asarray(neurons).reshape(self.shape)
        elif self.init_method == "naive_sharding":
            neurons = initiate_naive_sharding(X=data, k=self.m * self.n)
            return cp.asarray(neurons).reshape(self.shape)
        elif self.init_method == "lecun":
            neurons = initiate_lecun(input_shape=self.dim, output_shape=self.m * self.n)
            return cp.asarray(neurons).reshape(self.shape)
        elif self.init_method == "lsuv":
            neurons = initiate_lsuv(input_dim=self.dim, output_dim=self.m * self.n, X_batch=data)
            return cp.asarray(neurons).reshape(self.shape)

        raise ValueError(f"Invalid initiation method: {self.init_method}")
    def index_bmu(self, x: Any) -> Tuple[int, int]:
        """
        Find the index of the best matching unit (BMU) among all neurons.
        Optimized with multiple backends for better performance.

        Args:
            x (cp.ndarray): Input data point (on GPU/CPU).

        Returns:
            Tuple[int, int]: The indices (row, column) of the BMU.
        """
        neurons_flat = self.neurons.reshape(-1, self.dim)
        
        if self.backend == "cupy" and _USING_CUPY:
            # Use existing CuPy implementation
            if self.dist_func == "euclidean":
                diff = neurons_flat - x
                d2 = cp.sum(diff * diff, axis=1)
                min_index_gpu = cp.argmin(d2)
            elif self.dist_func == "cosine":
                norm_neurons = cp.linalg.norm(neurons_flat, axis=1, keepdims=True) + 1e-12
                norm_x = cp.linalg.norm(x) + 1e-12
                normalized_neurons = neurons_flat / norm_neurons
                normalized_x = x / norm_x
                cosine_sim = cp.sum(normalized_neurons * normalized_x, axis=1)
                distances = 1.0 - cosine_sim
                min_index_gpu = cp.argmin(distances)
            min_index = int(min_index_gpu.item())
            
        elif self.backend == "numba" and _USING_NUMBA:
            # Use Numba JIT-optimized implementation
            x_cpu = cp.asnumpy(x) if hasattr(x, 'get') else x
            neurons_cpu = cp.asnumpy(neurons_flat) if hasattr(neurons_flat, 'get') else neurons_flat
            
            min_dist = np.inf
            min_index = 0
            
            for i in range(neurons_cpu.shape[0]):
                if self.dist_func == "euclidean":
                    dist = euclidean_distance_squared_jit(x_cpu, neurons_cpu[i])
                else:  # cosine
                    dist = cosine_distance_jit(x_cpu, neurons_cpu[i])
                
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
                    
        else:
            # Fallback to NumPy implementation
            x_cpu = cp.asnumpy(x) if hasattr(x, 'get') else x
            neurons_cpu = cp.asnumpy(neurons_flat) if hasattr(neurons_flat, 'get') else neurons_flat
            
            if self.dist_func == "euclidean":
                diff = neurons_cpu - x_cpu
                d2 = np.sum(diff * diff, axis=1)
                min_index = int(np.argmin(d2))
            elif self.dist_func == "cosine":
                norm_neurons = np.linalg.norm(neurons_cpu, axis=1, keepdims=True) + 1e-12
                norm_x = np.linalg.norm(x_cpu) + 1e-12
                normalized_neurons = neurons_cpu / norm_neurons
                normalized_x = x_cpu / norm_x
                cosine_sim = np.sum(normalized_neurons * normalized_x, axis=1)
                distances = 1.0 - cosine_sim
                min_index = int(np.argmin(distances))
            else:
                raise ValueError(f"Unsupported distance function: {self.dist_func}")

        # Convert linear index to 2D indices
        row = min_index // self.n
        col = min_index % self.n
        return row, col

    def _ensure_grid(self) -> None:
        """Precompute and cache grid coordinates used for neighborhood computation."""
        if self._grid_rows is None or self._grid_cols is None:
            self._grid_rows = cp.arange(self.m).reshape(self.m, 1)
            self._grid_cols = cp.arange(self.n).reshape(1, self.n)

    def _bmu_indices_batch(self, data_batch: Any) -> Any:
        """Compute BMU linear indices for a batch of samples with optimized backends.

        Args:
            data_batch: shape (B, dim)

        Returns:
            Array of shape (B,) with linear BMU indices in range [0, m*n).
        """
        neurons_flat = self.neurons.reshape(-1, self.dim)  # (K, dim)
        batch_size = data_batch.shape[0]
        
        if self.backend == "cupy" and _USING_CUPY:
            # Use existing optimized CuPy implementation
            if self.dist_func == "euclidean":
                neuron_norms = cp.sum(neurons_flat * neurons_flat, axis=1, keepdims=True).T
                batch_norms = cp.sum(data_batch * data_batch, axis=1, keepdims=True)
                cross_terms = data_batch @ neurons_flat.T
                distances_sq = batch_norms + neuron_norms - 2.0 * cross_terms
                idx = cp.argmin(distances_sq, axis=1)
            elif self.dist_func == "cosine":
                neuron_norms = cp.linalg.norm(neurons_flat, axis=1, keepdims=True).T + 1e-12
                batch_norms = cp.linalg.norm(data_batch, axis=1, keepdims=True) + 1e-12
                dot_products = data_batch @ neurons_flat.T
                cosine_similarities = dot_products / (batch_norms * neuron_norms)
                distances = 1.0 - cosine_similarities
                idx = cp.argmin(distances, axis=1)
                
        elif self.backend == "taichi" and _USING_TAICHI:
            # Use Taichi parallel implementation
            data_cpu = cp.asnumpy(data_batch) if hasattr(data_batch, 'get') else data_batch
            neurons_cpu = cp.asnumpy(neurons_flat) if hasattr(neurons_flat, 'get') else neurons_flat
            
            # Create Taichi fields
            data_field = ti.field(dtype=ti.f32, shape=data_cpu.shape)
            neurons_field = ti.field(dtype=ti.f32, shape=neurons_cpu.shape)
            distances_field = ti.field(dtype=ti.f32, shape=(batch_size, neurons_cpu.shape[0]))
            
            # Copy data to Taichi fields
            data_field.from_numpy(data_cpu.astype(np.float32))
            neurons_field.from_numpy(neurons_cpu.astype(np.float32))
            
            if self.dist_func == "euclidean":
                taichi_batch_euclidean_distances(data_field, neurons_field, distances_field)
            
            # Get results and find minimum indices
            distances_result = distances_field.to_numpy()
            idx = np.argmin(distances_result, axis=1)
            
            # Convert back to appropriate backend
            if self.backend == "cupy":
                idx = cp.asarray(idx)
                
        elif self.backend == "numba" and _USING_NUMBA:
            # Use Numba JIT-optimized implementation
            data_cpu = cp.asnumpy(data_batch) if hasattr(data_batch, 'get') else data_batch
            neurons_cpu = cp.asnumpy(neurons_flat) if hasattr(neurons_flat, 'get') else neurons_flat
            
            if self.dist_func == "euclidean":
                distances = batch_euclidean_distances_jit(data_cpu, neurons_cpu)
            else:  # cosine
                distances = batch_cosine_distances_jit(data_cpu, neurons_cpu)
            
            idx = np.argmin(distances, axis=1)
            
            # Convert back to appropriate backend
            if _USING_CUPY and hasattr(data_batch, 'get'):
                idx = cp.asarray(idx)
                
        else:
            # Fallback NumPy implementation
            data_cpu = cp.asnumpy(data_batch) if hasattr(data_batch, 'get') else data_batch
            neurons_cpu = cp.asnumpy(neurons_flat) if hasattr(neurons_flat, 'get') else neurons_flat
            
            if self.dist_func == "euclidean":
                distances = np.linalg.norm(data_cpu[:, np.newaxis] - neurons_cpu, axis=2)
            elif self.dist_func == "cosine":
                neuron_norms = np.linalg.norm(neurons_cpu, axis=1, keepdims=True) + 1e-12
                batch_norms = np.linalg.norm(data_cpu, axis=1, keepdims=True) + 1e-12
                dot_products = data_cpu @ neurons_cpu.T
                cosine_similarities = dot_products / (batch_norms * neuron_norms.T)
                distances = 1.0 - cosine_similarities
            
            idx = np.argmin(distances, axis=1)
            
            # Convert back to appropriate backend
            if _USING_CUPY and hasattr(data_batch, 'get'):
                idx = cp.asarray(idx)
        
        return idx  # (B,)


    def _vectorized_update(self, neurons: Any, data_point: Any) -> Any:
        """
        Perform a vectorized update of the neurons given one data point.
        Optimized with multiple backends including parallel Taichi kernels.

        Args:
            neurons (cp.ndarray): Neuron weights (shape: (m, n, dim)).
            data_point (cp.ndarray): Single input data point (shape: (dim,)).

        Returns:
            cp.ndarray: Updated neuron weights.
        """
        # Find the BMU for the data point.
        bmu_row, bmu_col = self.index_bmu(x=data_point)
        
        if self.backend == "taichi" and _USING_TAICHI and self.dist_func == "euclidean":
            # Use Taichi parallel kernel for neighborhood update
            neurons_cpu = cp.asnumpy(neurons) if hasattr(neurons, 'get') else neurons
            data_cpu = cp.asnumpy(data_point) if hasattr(data_point, 'get') else data_point
            
            # Create Taichi fields
            neurons_field = ti.field(dtype=ti.f32, shape=neurons_cpu.shape)
            data_field = ti.field(dtype=ti.f32, shape=data_cpu.shape)
            
            # Copy data to Taichi fields
            neurons_field.from_numpy(neurons_cpu.astype(np.float32))
            data_field.from_numpy(data_cpu.astype(np.float32))
            
            # Perform parallel neighborhood update
            taichi_neighborhood_update(
                neurons_field, data_field, bmu_row, bmu_col,
                self.cur_learning_rate, 
                max(self.cur_neighbour_rad * self.cur_neighbour_rad, 1e-18)
            )
            
            # Get results
            result = neurons_field.to_numpy()
            
            # Convert back to appropriate backend
            if _USING_CUPY and hasattr(neurons, 'get'):
                return cp.asarray(result)
            else:
                return result
                
        else:
            # Use existing optimized implementation for CuPy/NumPy backends
            self._ensure_grid()
            grid_rows = self._grid_rows
            grid_cols = self._grid_cols
            
            # Compute squared distances from each neuron to the BMU.
            dist_squared = (grid_rows - bmu_row) ** 2 + (grid_cols - bmu_col) ** 2
            
            # Avoid division by zero and use more stable computation.
            nr_sq = max(self.cur_neighbour_rad * self.cur_neighbour_rad, 1e-18)
            
            if self.backend == "numba" and _USING_NUMBA:
                # Use JIT-optimized neighborhood function
                neurons_cpu = cp.asnumpy(neurons) if hasattr(neurons, 'get') else neurons
                data_cpu = cp.asnumpy(data_point) if hasattr(data_point, 'get') else data_point
                dist_sq_cpu = cp.asnumpy(dist_squared) if hasattr(dist_squared, 'get') else dist_squared
                
                # Compute neighborhood influence using JIT
                h_cpu = neighborhood_function_jit(dist_sq_cpu, self.cur_learning_rate, nr_sq)
                h_expanded_cpu = h_cpu[:, :, np.newaxis]
                
                if self.dist_func == "euclidean":
                    diff = data_cpu - neurons_cpu
                    neurons_cpu += h_expanded_cpu * diff
                elif self.dist_func == "cosine":
                    neuron_norm = np.linalg.norm(neurons_cpu, axis=2, keepdims=True) + 1e-12
                    x_norm = np.linalg.norm(data_cpu) + 1e-12
                    normalized_neurons = neurons_cpu / neuron_norm
                    normalized_x = data_cpu / x_norm
                    dot_product = np.sum(normalized_neurons * normalized_x, axis=2, keepdims=True)
                    update_direction = dot_product * normalized_neurons - normalized_neurons
                    neurons_cpu += h_expanded_cpu * update_direction * neuron_norm
                
                # Convert back to appropriate backend
                if _USING_CUPY and hasattr(neurons, 'get'):
                    return cp.asarray(neurons_cpu)
                else:
                    return neurons_cpu
            else:
                # Use existing CuPy/NumPy implementation
                # Compute the neighborhood function over the grid with numerical stability.
                h = self.cur_learning_rate * cp.exp(-dist_squared / (2.0 * nr_sq))
                # Expand h to match neuron shape.
                h_expanded = h[:, :, cp.newaxis]  # shape (m, n, 1)
                
                if self.dist_func == "euclidean":
                    # In-place update to reduce allocations - more numerically stable
                    diff = data_point - neurons
                    neurons += h_expanded * diff
                elif self.dist_func == "cosine":
                    # Optimized cosine update with better numerical stability
                    neuron_norm = cp.linalg.norm(neurons, axis=2, keepdims=True) + 1e-12
                    x_norm = cp.linalg.norm(data_point) + 1e-12
                    # Normalize both vectors for more stable cosine computation
                    normalized_neurons = neurons / neuron_norm
                    normalized_x = data_point / x_norm
                    dot_product = cp.sum(normalized_neurons * normalized_x, axis=2, keepdims=True)
                    # Use stable cosine similarity computation
                    update_direction = dot_product * normalized_neurons - normalized_neurons
                    neurons += h_expanded * update_direction * neuron_norm
                    
                return neurons

    def fit(self, x: np.ndarray, epoch: int, shuffle: bool = True, batch_size: int = None) -> None:
        """
        Fit the SOM to the input data using optimized batch processing.
        
        This method now supports mini-batch processing for better memory management
        and improved convergence properties.
        
        Args:
            x (np.ndarray): Input data (CPU, NumPy array).
            epoch (int): Number of epochs.
            shuffle (bool, optional): Whether to shuffle data each epoch.
            batch_size (int, optional): Batch size for mini-batch processing. 
                                      If None, uses full batch processing.
        """
        # Validate input data for NaN or infinite values
        if np.any(np.isnan(x)):
            raise ValueError("Input data contains NaN values, which are not supported")
        if np.any(np.isinf(x)):
            raise ValueError("Input data contains infinite values, which are not supported")
        
        # Initialize neurons if not already trained.
        if not self._trained:
            self.neurons = self.initiate_neuron(data=x)
            self.initial_neurons = self.neurons.copy()

        if x.shape[1] != self.dim:
            raise ValueError(f"X.shape[1] should be {self.dim}, but found {x.shape[1]}")

        # Convert input data to GPU/CPU array.
        data_gpu: Any = cp.asarray(x)
        n_sample: int = data_gpu.shape[0]
        total_iterations = int(min(epoch * n_sample, self.max_iter))

        # Determine batch size - use adaptive sizing if not specified
        if batch_size is None:
            batch_size = min(n_sample, max(32, n_sample // 100))  # Adaptive batch size
        
        global_iter = 0
        for epoch_num in range(epoch):
            # Optionally shuffle the data entirely on GPU
            if shuffle:
                perm = cp.random.permutation(n_sample)
                data_gpu = data_gpu[perm]

            # Process data in mini-batches for better memory management
            for batch_start in range(0, n_sample, batch_size):
                if global_iter > total_iterations:
                    break
                    
                batch_end = min(batch_start + batch_size, n_sample)
                batch_data = data_gpu[batch_start:batch_end]
                
                # Process each sample in the batch
                for idx in range(batch_data.shape[0]):
                    global_iter += 1
                    if global_iter > total_iterations:
                        break

                    data_point = batch_data[idx]
                    # Update neurons using vectorized update.
                    self.neurons = self._vectorized_update(self.neurons, data_point)

                    # Update learning rate and neighborhood radius with exponential decay
                    # for better convergence properties
                    progress = global_iter / max(total_iterations, 1)
                    # Use exponential decay for more stable convergence
                    self.cur_learning_rate = self.initial_learning_rate * cp.exp(-5.0 * progress)
                    self.cur_neighbour_rad = max(1e-12, self.initial_neighbour_rad * cp.exp(-3.0 * progress))

        self._trained = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for the input data.

        Args:
            x (np.ndarray): Input data (CPU, NumPy array).

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if not self._trained:
            raise RuntimeError("SOM must be fitted before predicting")

        if len(x.shape) != 2:
            raise ValueError(f"X should have two dimensions, not {len(x.shape)}")
        if x.shape[1] != self.dim:
            raise ValueError(f"This SOM has dimension {self.dim} but received {x.shape[1]}")

        data_gpu = cp.asarray(x)
        idx = self._bmu_indices_batch(data_gpu)  # (N,)
        # Convert linear BMU index to 2D then to label = m * row + col (same as linear)
        labels_gpu = idx  # already linear index in row-major
        return cp.asnumpy(labels_gpu)

    def fit_predict(self, x: np.ndarray, epoch: int, shuffle: bool = True, batch_size: int = None) -> np.ndarray:
        """
        Fit the SOM and then predict cluster labels for the input data.

        Args:
            x (np.ndarray): Input data (CPU, NumPy array).
            epoch (int): Number of epochs.
            shuffle (bool, optional): Whether to shuffle data each epoch.
            batch_size (int, optional): Batch size for mini-batch processing.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        self.fit(x=x, epoch=epoch, shuffle=shuffle, batch_size=batch_size)
        return self.predict(x=x)

    def evaluate(self, x: np.ndarray, method: List[str]) -> Union[List[float], dict]:
        """
        Evaluate the SOM clustering.

        Args:
            x (np.ndarray): Input data (CPU, NumPy array).
            method (List[str]): List of evaluation methods.

        Returns:
            Union[List[float], dict]: Evaluation scores.
        """
        if not set(method).issubset(set(EVAL_METHOD_LIST)):
            raise ValueError(f'Invalid evaluation method(s): {set(method) - set(EVAL_METHOD_LIST)}')

        pred: np.ndarray = self.predict(x)
        score_functions = {
            "silhouette": silhouette_score,
            "davies_bouldin": davies_bouldin_index,
            "calinski_harabasz": calinski_harabasz_score,
            "dunn": dunn_index
        }
        if "all" not in method:
            return [score_functions[m](x=x, labels=pred) for m in method if m in score_functions]
        return {
            "silhouette": silhouette_score(x=x, labels=pred),
            "davies_bouldin": davies_bouldin_index(x=x, labels=pred),
            "calinski_harabasz": calinski_harabasz_score(x=x, labels=pred),
            "dunn": dunn_index(x=x, labels=pred)
        }

    @property
    def cluster_center_(self) -> np.ndarray:
        """
        Get the cluster centers as a NumPy array.
        """
        return cp.asnumpy(self.neurons.reshape(-1, self.dim))

    def save(self, path: str) -> None:
        """
        Save the SOM model to a file.
        """
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str) -> 'SOM':
        """
        Load a SOM model from a file.
        """
        with open(path, 'rb') as file:
            return pickle.load(file)
