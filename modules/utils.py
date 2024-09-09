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
from typing import List
import numpy as np

def find_most_edge_point(points: np.ndarray) -> np.ndarray:
    """
    Find the point farthest from the center of the dataset.

    Args:
        points (np.ndarray): An array of points.

    Returns:
        np.ndarray: The point farthest from the center.
    """
    # Calculate the center of the dataset
    center = np.mean(points, axis=0)

    # Calculate the Euclidean distance between each point and the center
    distances = np.sqrt(np.sum((points - center) ** 2, axis=1))

    # Find the index of the point with the maximum distance
    most_edge_index = np.argmax(distances)

    return points[most_edge_index]

def cos_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate the cosine distance between two vectors.

    Args:
        vector1 (List[float]): The first vector.
        vector2 (List[float]): The second vector.

    Returns:
        float: The cosine distance between the two vectors.

    Raises:
        ValueError: If the vectors are not of the same length.
    """
    if len(vector1) != len(vector2):
        raise ValueError("Input vectors must have the same length," +
                         f"but got {len(vector1)} != {len(vector2)}.")

    mag_a = np.linalg.norm(vector1)
    mag_b = np.linalg.norm(vector2)
    d_cos = 1 - mag_a * mag_b / (mag_a ** 2 + mag_b ** 2)
    return math.acos(d_cos)

def random_initiate(dim: int, min_val: float, max_val: float) -> np.ndarray:
    """
    Initiate an array of random numbers in the range (min_val, max_val).

    Args:
        dim (int): Dimension of the array.
        min_val (float): Minimum value of the random numbers.
        max_val (float): Maximum value of the random numbers.

    Returns:
        np.ndarray: Array of randomly generated numbers.
    """
    return np.random.uniform(min_val, max_val, dim)

def euc_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points in n-dimensional space.

    Args:
        point1 (list or tuple): The coordinates of the first point.
        point2 (list or tuple): The coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.

    Raises:
        ValueError: If the dimensions of the two points are not equal.
    """
    if len(point1) != len(point2):
        raise ValueError("The dimensions of the two points must be equal.")

    squared_diff_sum = sum((x1 - x2) ** 2 for x1, x2 in zip(point1, point2))
    return math.sqrt(squared_diff_sum)

def one_hot_encode(y: np.ndarray) -> np.ndarray:
    """
    One-hot encode a numpy array of labels.

    Args:
        y (np.ndarray): Array of labels to be encoded.

    Returns:
        np.ndarray: The one-hot encoded array.
    """
    classes = np.unique(y)
    encoded = np.zeros((y.size, classes.size))
    for idx, label in enumerate(y):
        encoded[idx, np.where(classes == label)[0][0]] = 1
    return encoded

def normalize_column(data: np.ndarray, column_index: int) -> np.ndarray:
    """
    Normalize a specific column in a numpy array.

    Args:
        data (np.ndarray): The data array.
        column_index (int): The index of the column to normalize.

    Returns:
        np.ndarray: The normalized column.
    """
    column = data[:, column_index]
    min_val = np.min(column)
    max_val = np.max(column)
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column
