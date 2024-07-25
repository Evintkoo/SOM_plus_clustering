import numpy as np
import math
import sys

def find_most_edge_point(points):
    # Calculate the center of the dataset
    center = np.mean(points, axis=0)
    
    # Calculate the Euclidean distance between each point and the center
    distances = np.sqrt(np.sum((points - center)**2, axis=1))
    
    # Find the index of the point with the maximum distance
    most_edge_index = np.argmax(distances)
    
    return points[most_edge_index]

def cos_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("input value has different length,", len(vector1), "!=", len(vector2))
    else:
        mag_a = np.linalg.norm(vector1)
        mag_b = np.linalg.norm(vector2)
        d_cos = 1-mag_a*mag_b/(mag_a**2+mag_b**2)
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

    Time Complexity: O(1)
    Space Complexity: O(dim)
    """
    return np.random.uniform(min_val, max_val, dim)

import math

def euc_distance(point1, point2):
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
    distance = math.sqrt(squared_diff_sum)
    return distance

def one_hot_encode(y):
    classes = np.unique(y)
    encoded = np.zeros((y.size, classes.size))
    for idx, label in enumerate(y):
        encoded[idx, np.where(classes == label)[0][0]] = 1
    return encoded

def normalize_column(data, column_index):
    column = data[:, column_index]
    min_val = np.min(column)
    max_val = np.max(column)
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column