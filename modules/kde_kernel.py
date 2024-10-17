"""
Kernel Density Estimation (KDE) with Neuron Selection
This script implements a custom Kernel Density Estimation (KDE) algorithm in Python, without relying on external libraries like scipy. The KDE function is constructed using a Gaussian kernel manually, and it evaluates the density on a set of points in a multidimensional space.

The process is further extended to select a subset of neurons (local maxima) based on their spatial distribution. The neurons are chosen by iteratively finding the point that is furthest from the already selected set, ensuring an even spread across the space.

Key Components:
Custom Gaussian Kernel Function:

A multidimensional Gaussian kernel is implemented to compute kernel values for any given data point and grid point.
KDE Calculation:

The kde_multidimensional function manually computes the Kernel Density Estimation over a set of grid points, without using external packages like scipy. This function is used to estimate the density distribution based on input data.
Local Maxima Detection:

The find_local_maxima function identifies local maxima in the KDE result, representing the highest-density areas. These maxima are used as candidate neurons.
Neuron Selection:

The script includes functionality to select a subset of neurons from the detected local maxima. The selection is performed iteratively, choosing points that are farthest from previously selected neurons, ensuring optimal coverage of the space.
Bandwidth Estimation:

A custom bandwidth estimator based on the formula provided is used to determine the smoothing parameter for the Gaussian kernel. This formula is a heuristic based on the range of the data and the number of points, ensuring an appropriate balance between under-smoothing and over-smoothing.
Usage:
The script can be used to:

Estimate the probability density function for a set of data points in multiple dimensions.
Identify and select a subset of neurons based on local maxima in the KDE.
Control the smoothness of the KDE using a bandwidth parameter, calculated manually or passed as an argument.
Example Workflow:
Generate or load your dataset.
Use the custom KDE function to estimate the density across a grid.
Detect local maxima in the estimated density.
Select the most representative neurons from these maxima, ensuring they are spread across the dataset.
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