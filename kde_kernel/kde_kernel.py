"""
This module implements the KMeans clustering algorithm for unsupervised learning.

The KMeans algorithm clusters data into a specified number of clusters (n_clusters)
by iteratively minimizing the distance between data points and cluster centroids.

Key Components:
- `KMeans` Class: 
  The main class that handles the KMeans clustering process. It supports different
  methods for centroid initialization ('random' and 'kmeans++') and provides
  functionalities to fit the model to input data, predict cluster labels for new data,
  and update centroids during training.

Functions:
- `initiate_plus_plus`: Initializes centroids using the KMeans++ algorithm.
- `init_centroids`: Initializes centroids using the specified method.
- `update_centroids`: Updates the centroid values based on the input data points.
- `fit`: Trains the KMeans model to find the optimal centroid values.
- `predict`: Predicts the cluster label for each data point in the input matrix.

Utility Functions:
- The module relies on utility functions (`random_initiate`, `euc_distance`) 
  from an external `utils` module to handle centroid initialization and distance
  calculation.

Usage:
- Import this module and create an instance of the `KMeans` class with the desired
  number of clusters and initialization method.
- Use the `fit` method to train the model on input data.
- Use the `predict` method to assign cluster labels to new data points.

Overall, this module provides a flexible and efficient implementation of the KMeans
clustering algorithm, suitable for a variety of unsupervised learning tasks.
"""


from typing import Callable, List, Union
import numpy as np
import numpy as np
from joblib import Parallel, delayed

import numpy as np

def gaussian_kernel(x, xi, bandwidth):
    """
    Multidimensional Gaussian kernel function.
    
    Parameters:
    x (numpy.ndarray): A point where the kernel is evaluated (D-dimensional vector).
    xi (numpy.ndarray): A data point from the dataset (D-dimensional vector).
    bandwidth (float): The smoothing bandwidth.

    Returns:
    float: Kernel value for the point x and data point xi.
    """
    d = len(x)
    norm_factor = (1 / (np.sqrt(2 * np.pi) ** d * bandwidth ** d))  # Normalization constant for Gaussian kernel
    diff = x - xi
    exponent = -0.5 * np.dot(diff, diff) / (bandwidth ** 2)
    return norm_factor * np.exp(exponent)

def kde_multidimensional(data, points, bandwidth):
    """
    Multidimensional KDE using a manually defined Gaussian kernel.

    Parameters:
    data (numpy.ndarray): An NxD array where N is the number of data points and D is the dimensionality.
    points (numpy.ndarray): Points where the KDE will be evaluated (MxD array).
    bandwidth (float): The bandwidth parameter for smoothing.

    Returns:
    kde_values (numpy.ndarray): KDE evaluated at the points.
    """
    N = data.shape[0]
    kde_values = np.zeros(points.shape[0])
    
    for i, point in enumerate(points):
        kernel_sum = 0
        for xi in data:
            kernel_sum += gaussian_kernel(point, xi, bandwidth)
        kde_values[i] = kernel_sum / N
    
    return kde_values

def find_local_maxima(kde_values, points):
    """
    Finds local maxima of the KDE in the grid of points.
    
    Parameters:
    kde_values (numpy.ndarray): KDE values computed at each point.
    points (numpy.ndarray): The corresponding points where KDE was evaluated.

    Returns:
    numpy.ndarray: Points corresponding to local maxima.
    """
    local_maxima = []
    for i in range(1, len(kde_values) - 1):
        if kde_values[i - 1] < kde_values[i] > kde_values[i + 1]:
            local_maxima.append(points[i])
    
    return np.array(local_maxima)

def bandwidth_estimator(data: np.ndarray) -> float:
    """
    Bandwidth estimator based on the given formula.

    Args:
        data: A numpy array of shape (n_samples,) representing the dataset.

    Returns:
        Bandwidth (h) calculated from the data.
    """
    n = len(data)
    if n < 2:
        raise ValueError("Data must contain at least two points to compute bandwidth.")
    
    # Calculate the bandwidth using the formula
    h = (np.max(data) - np.min(data)) / (1 + np.log2(n))
    
    return h

def initiate_kde(x: np.ndarray, n_neurons: int, bandwidth: Union[float, None]=None) -> np.ndarray:
    """
    Initiates KDE by finding local maxima and selecting a subset of neurons based on the furthest distance.

    Args:
        x: A numpy array of shape (n_samples, n_features) representing the dataset.
        n_neurons: An integer specifying the number of neurons to select.
        bandwidth: A float specifying the bandwidth for the KDE.

    Returns:
        A numpy array of selected neurons.

    Raises:
        ValueError: If the maximum number of neurons is less than or equal to `n_neurons`.
    """
    
    if bandwidth == None:
        bandwidth = bandwidth_estimator(x)
    
    kde_values = kde_multidimensional(x, x, bandwidth)
    local_max = find_local_maxima(kde_values, x)

    
    max_neurons = local_max.shape[0]
    if max_neurons <= n_neurons:
        raise ValueError(f"Maximum number of neurons is {max_neurons}")
    
    # Select the first neuron randomly
    selected = [np.random.choice(max_neurons)]
    
    # Boolean mask to keep track of selected neurons
    selected_mask = np.zeros(max_neurons, dtype=bool)
    selected_mask[selected[0]] = True
    
    # Precompute all pairwise distances (squared distances to avoid costly sqrt operations)
    dist_matrix = np.sum((local_max[:, np.newaxis, :] - local_max[np.newaxis, :, :]) ** 2, axis=-1)
    
    # Track the minimum distance to the selected set for each unselected point
    min_dist_to_selected = dist_matrix[selected[0]]
    
    for _ in range(1, n_neurons):
        # Find the unselected point with the maximum minimum distance to selected points
        unselected_indices = np.where(~selected_mask)[0]
        next_neuron = unselected_indices[np.argmax(min_dist_to_selected[unselected_indices])]
        
        # Mark this neuron as selected
        selected.append(next_neuron)
        selected_mask[next_neuron] = True
        
        # Update the minimum distances to the selected points
        min_dist_to_selected = np.minimum(min_dist_to_selected, dist_matrix[next_neuron])
    
    return local_max[selected]