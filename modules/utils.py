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