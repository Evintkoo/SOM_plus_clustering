"""
utils.py

This module provides utility functions for various mathematical operations and 
data processing tasks commonly used in machine learning and data analysis.

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

Usage Example:
    ```python
    import numpy as np
    from utils import (
        find_most_edge_point, cos_distance, random_initiate, 
        euc_distance, one_hot_encode, normalize_column
    )

    # Example usage of find_most_edge_point
    points = np.array([[1, 2], [3, 4], [5, 6]])
    edge_point = find_most_edge_point(points)
    print("Farthest point from center:", edge_point)

    # Example usage of cos_distance
    vector1 = [1, 0, 0]
    vector2 = [0, 1, 0]
    distance = cos_distance(vector1, vector2)
    print("Cosine distance between vectors:", distance)

    # Example usage of random_initiate
    random_values = random_initiate(3, 0.0, 1.0)
    print("Randomly initiated values:", random_values)

    # Example usage of euc_distance
    point1 = np.array([1, 2, 3])
    point2 = np.array([4, 5, 6])
    euclidean_distance = euc_distance(point1, point2)
    print("Euclidean distance between points:", euclidean_distance)

    # Example usage of one_hot_encode
    labels = np.array([0, 1, 2, 1, 0])
    encoded_labels = one_hot_encode(labels)
    print("One-hot encoded labels:\n", encoded_labels)

    # Example usage of normalize_column
    data = np.array([[1, 2], [3, 4], [5, 6]])
    normalized_col = normalize_column(data, 0)
    print("Normalized column:", normalized_col)
    ```

Dependencies:
    - numpy: Used for numerical computations and data manipulation.
    - math: Provides access to mathematical functions.
    - typing: Used for type hinting in function definitions.
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

def find_most_edge_point(points: Any) -> Any:
    """
    Find the point farthest from the center of the dataset.

    Args:
        points (cp.ndarray): An array of points.

    Returns:
        cp.ndarray: The point farthest from the center.
    """
    # Calculate the center of the dataset
    center = cp.mean(points, axis=0)
    
    # Calculate the Euclidean distance between each point and the center
    distances = cp.sqrt(cp.sum((points - center) ** 2, axis=1))
    
    # Find the index of the point with the maximum distance
    most_edge_index = cp.argmax(distances)
    
    return points[most_edge_index]


def cos_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate the cosine distance between two vectors.
    Optimized for better numerical stability and performance.

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
    
    # Convert lists to CuPy arrays
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
    
    # You can also use cp.linalg.norm for a concise implementation.
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
    if range_val < 1e-12:
        # If range is very small, return zeros or original values
        return cp.zeros_like(column)
    
    normalized_column = (column - min_val) / range_val
    return normalized_column
