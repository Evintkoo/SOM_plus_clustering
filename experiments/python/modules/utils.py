"""
utils.py

This module provides utility functions for various mathematical operations and 
data processing tasks commonly used in machine learning and data analysis.

Performance optimizations:
- Numba JIT compilation for mathematical operations
- CuPy GPU acceleration when available
- Optimized algorithms for better numerical stability

Functions:
    find_most_edge_point(points: np.ndarray) -> np.ndarray:
        Finds the point farthest from the center of a given dataset.

    cos_distance(vector1: List[float], vector2: List[float]) -> float:
        Calculates the cosine distance between two vectors.

    random_initiate(dim: int, min_val: float, max_val: float) -> np.ndarray:
        Generates an array of random numbers within a specified range.

    euc_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        Calculates the Euclidean distance between two points in n-dimensional space.

    one_hot_encode(y: np.ndarray) -> np.ndarray:
        Performs one-hot encoding of an array of labels.

    normalize_column(data: np.ndarray, column_index: int) -> np.ndarray:
        Normalizes a specific column in a numpy array.
"""

import math
from typing import List, Any
import importlib
import numpy as np
try:
    cp = importlib.import_module('cupy')  # type: ignore
except Exception:  # pragma: no cover - allow CPU-only environments
    cp = np  # type: ignore
    if not hasattr(cp, 'asarray'):
        cp.asarray = np.asarray  # type: ignore
    if not hasattr(cp, 'arccos'):
        cp.arccos = np.arccos  # type: ignore

# JIT optimization imports
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

# JIT-optimized utility functions
@jit(nopython=True, fastmath=True)
def euclidean_distance_jit(point1, point2):
    """JIT-optimized Euclidean distance calculation."""
    total = 0.0
    for i in range(len(point1)):
        diff = point1[i] - point2[i]
        total += diff * diff
    return math.sqrt(total)

@jit(nopython=True, fastmath=True)
def cosine_distance_jit(vector1, vector2):
    """JIT-optimized cosine distance calculation."""
    dot_product = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    
    for i in range(len(vector1)):
        dot_product += vector1[i] * vector2[i]
        norm1_sq += vector1[i] * vector1[i]
        norm2_sq += vector2[i] * vector2[i]
    
    norm1 = math.sqrt(norm1_sq) + 1e-12
    norm2 = math.sqrt(norm2_sq) + 1e-12
    
    cosine_sim = dot_product / (norm1 * norm2)
    cosine_sim = max(-1.0, min(1.0, cosine_sim))  # Clip to valid range
    
    return 1.0 - cosine_sim

@jit(nopython=True, fastmath=True)
def find_most_edge_point_jit(points):
    """JIT-optimized function to find the point farthest from center."""
    n_points, n_features = points.shape
    
    # Calculate center
    center = np.zeros(n_features)
    for i in range(n_points):
        for j in range(n_features):
            center[j] += points[i, j]
    
    for j in range(n_features):
        center[j] /= n_points
    
    # Find farthest point
    max_distance = 0.0
    max_index = 0
    
    for i in range(n_points):
        distance_sq = 0.0
        for j in range(n_features):
            diff = points[i, j] - center[j]
            distance_sq += diff * diff
        
        if distance_sq > max_distance:
            max_distance = distance_sq
            max_index = i
    
    return points[max_index]

def find_most_edge_point(points: Any) -> Any:
    """
    Find the point farthest from the center of the dataset.
    Optimized with JIT compilation for better performance.

    Args:
        points (cp.ndarray): An array of points.

    Returns:
        cp.ndarray: The point farthest from the center.
    """
    if _USING_NUMBA:
        # Use JIT-optimized implementation
        points_cpu = cp.asnumpy(points) if hasattr(points, 'get') else points
        result = find_most_edge_point_jit(points_cpu)
        
        # Convert back to appropriate backend
        if hasattr(points, 'get'):  # CuPy array
            return cp.asarray(result)
        else:
            return result
    else:
        # Use existing CuPy/NumPy implementation
        center = cp.mean(points, axis=0)
        distances = cp.sqrt(cp.sum((points - center) ** 2, axis=1))
        most_edge_index = cp.argmax(distances)
        return points[most_edge_index]


def cos_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate the cosine distance between two vectors.
    Optimized with JIT compilation for better performance.

    Args:
        vector1 (List[float]): The first vector.
        vector2 (List[float]): The second vector.

    Returns:
        float: The cosine distance between the two vectors.

    Raises:
        ValueError: If the vectors are not of the same length.
    """
    if len(vector1) != len(vector2):
        raise ValueError("Input vectors must have the same length, " +
                         f"but got {len(vector1)} != {len(vector2)}.")
    
    if _USING_NUMBA:
        # Use JIT-optimized implementation
        v1_array = np.array(vector1, dtype=np.float64)
        v2_array = np.array(vector2, dtype=np.float64)
        return cosine_distance_jit(v1_array, v2_array)
    else:
        # Use existing CuPy/NumPy implementation
        v1 = cp.asarray(vector1)
        v2 = cp.asarray(vector2)
        
        # Compute norms with numerical stability
        norm_v1 = cp.linalg.norm(v1) + 1e-12
        norm_v2 = cp.linalg.norm(v2) + 1e-12
        
        # Compute cosine similarity using dot product
        dot_product = cp.dot(v1, v2)
        cosine_similarity = dot_product / (norm_v1 * norm_v2)
        
        # Ensure cosine similarity is in valid range [-1, 1]
        cosine_similarity = cp.clip(cosine_similarity, -1.0, 1.0)
        
        # Convert to cosine distance
        cosine_distance = 1.0 - cosine_similarity
        
        return float(cosine_distance.item())


def random_initiate(dim: int, min_val: float, max_val: float) -> Any:
    """
    Initiate an array of random numbers in the range (min_val, max_val).

    Args:
        dim (int): Dimension of the array.
        min_val (float): Minimum value of the random numbers.
        max_val (float): Maximum value of the random numbers.

    Returns:
        cp.ndarray: Array of randomly generated numbers.
    """
    return cp.random.uniform(min_val, max_val, dim)


def euc_distance(point1: Any, point2: Any) -> float:
    """
    Calculate the Euclidean distance between two points in n-dimensional space.
    Optimized with JIT compilation for better performance.

    Args:
        point1 (cp.ndarray): The coordinates of the first point.
        point2 (cp.ndarray): The coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.

    Raises:
        ValueError: If the dimensions of the two points are not equal.
    """
    if point1.shape != point2.shape:
        raise ValueError("The dimensions of the two points must be equal.")
    
    if _USING_NUMBA:
        # Use JIT-optimized implementation
        p1_cpu = cp.asnumpy(point1) if hasattr(point1, 'get') else point1
        p2_cpu = cp.asnumpy(point2) if hasattr(point2, 'get') else point2
        return euclidean_distance_jit(p1_cpu, p2_cpu)
    else:
        # Use existing CuPy/NumPy implementation
        return cp.linalg.norm(point1 - point2).item()


def one_hot_encode(y: Any) -> Any:
    """
    One-hot encode a CuPy array of labels (fully vectorized on GPU).
    Optimized for better performance and memory efficiency.

    Args:
        y (cp.ndarray): Array of integer-like labels to be encoded.

    Returns:
        cp.ndarray: The one-hot encoded array with shape (N, C).
    """
    # Flatten input if necessary and ensure it's 1D
    y_flat = y.flatten() if y.ndim > 1 else y
    
    # Compute unique classes and map labels to indices
    classes, inverse = cp.unique(y_flat, return_inverse=True)  # inverse in [0, C)
    num_samples = y_flat.size
    num_classes = classes.size

    # Allocate output array with appropriate dtype
    encoded = cp.zeros((num_samples, num_classes), dtype=cp.float32)
    
    # Use advanced indexing for efficient assignment
    rows = cp.arange(num_samples)
    encoded[rows, inverse] = 1.0
    
    return encoded


def normalize_column(data: Any, column_index: int) -> Any:
    """
    Normalize a specific column in a CuPy array.
    Optimized for better numerical stability.

    Args:
        data (cp.ndarray): The data array.
        column_index (int): The index of the column to normalize.

    Returns:
        cp.ndarray: The normalized column.
        
    Raises:
        IndexError: If column_index is out of bounds.
    """
    if column_index >= data.shape[1] or column_index < 0:
        raise IndexError(f"Column index {column_index} is out of bounds for array with {data.shape[1]} columns")
    
    column = data[:, column_index]
    min_val = cp.min(column)
    max_val = cp.max(column)
    
    # Add numerical stability check
    range_val = max_val - min_val
    if range_val < 1e-9:
        # If range is very small, return zeros or original values
        return cp.zeros_like(column)
    
    normalized_column = (column - min_val) / range_val
    return normalized_column
