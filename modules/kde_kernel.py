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


from typing import Callable, List
import numpy as np

def gaussian_kernel(x: np.ndarray) -> float:
    """
    Computes the Gaussian kernel for a given vector `x`.

    Args:
        x: A numpy array representing the input vector.

    Returns:
        A float value representing the Gaussian kernel result.
    """
    d: int = x.shape[0]
    return (1 / (2 * np.pi) ** (d / 2)) * np.exp(-0.5 * np.dot(x, x))

def multivariate_kde(x: np.ndarray, h: np.ndarray) -> Callable[[np.ndarray], float]:
    """
    Returns a KDE function using a multivariate Gaussian kernel.

    Args:
        X: A numpy array of shape (n_samples, n_features) representing the dataset.
        h: A numpy array representing the bandwidth matrix.

    Returns:
        A function that computes the KDE value for a given input point.
    """
    n : int = x.shape[0]
    h_inv: np.ndarray = np.linalg.inv(h)
    h_det: float = np.linalg.det(h)
    h_det_sqrt: float = h_det ** 0.5
    const: float = 1 / (n * h_det_sqrt)

    def kde(x_data: np.ndarray) -> float:
        x_data = np.asarray(x)
        if x_data.ndim == 1:
            x_data = x_data.reshape(1, -1)

        diff: np.ndarray = x_data[:, np.newaxis, :] - x
        tdiff: np.ndarray = np.einsum('ijk,kl->ijl', diff, h_inv)
        energy: np.ndarray = np.einsum('ijk,ijk->ij', diff, tdiff)
        result: np.ndarray = const * np.sum(np.exp(-0.5 * energy), axis=1)

        return float(result[0])

    return kde

def scotts_rule(x: np.ndarray) -> np.ndarray:
    """
    Computes the bandwidth matrix using Scott's rule.

    Args:
        X: A numpy array of shape (n_samples, n_features) representing the dataset.

    Returns:
        A diagonal bandwidth matrix based on the dataset.
    """
    n: int
    d: int
    n, d = x.shape
    sigma: np.ndarray = np.std(x, axis=0)
    h: np.ndarray = n ** (-1 / (d + 4)) * sigma
    return np.diag(h)

def numerical_gradient(f: Callable[[np.ndarray], float],
                       x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Computes the numerical gradient of a function `f` at point `x`.

    Args:
        f: A callable function that takes a numpy array and returns a float.
        x: A numpy array representing the point at which to compute the gradient.
        h: A float representing the step size for the finite difference approximation.

    Returns:
        A numpy array representing the gradient at point `x`.
    """
    grad: np.ndarray = np.zeros_like(x)
    for i in range(len(x)):
        x_plus: np.ndarray = x.copy()
        x_minus: np.ndarray = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def find_local_maxima(kde: Callable[[np.ndarray], float], x: np.ndarray,
                      num_restarts: int = 100, max_iter: int = 100, tol: float = 1e-5,
                      step_size: float = 0.1) -> np.ndarray:
    """
    Finds local maxima of the KDE function.

    Args:
        kde: A callable KDE function.
        X: A numpy array of shape (n_samples, n_features) representing the dataset.
        num_restarts: An integer specifying the number of restarts for finding local maxima.
        max_iter: An integer specifying the maximum number of iterations for each restart.
        tol: A float representing the tolerance for convergence.
        step_size: A float representing the step size for gradient ascent.

    Returns:
        A numpy array of local maxima points.
    """

    n: int= x.shape[0]
    local_maxima: List[np.ndarray] = []
    decimals: int = -int(np.log10(tol))

    for _ in range(num_restarts):
        random_x: np.ndarray = x[np.random.choice(n)]
        for _ in range(max_iter):
            grad: np.ndarray = numerical_gradient(kde, random_x)
            x_new: np.ndarray = random_x + step_size * grad
            if np.linalg.norm(x_new - random_x) < tol:
                local_maxima.append(x_new.round(decimals))
                break
            random_x = x_new

    return np.unique(np.array(local_maxima), axis=0)

def initiate_kde(x: np.ndarray, n_neurons: int) -> np.ndarray:
    """
    Initiates KDE by finding local maxima and selecting a subset of neurons.

    Args:
        X: A numpy array of shape (n_samples, n_features) representing the dataset.
        n_neurons: An integer specifying the number of neurons to select.

    Returns:
        A numpy array of selected neurons.

    Raises:
        ValueError: If the maximum number of neurons is less than or equal to `n_neurons`.
    """
    kde: Callable[[np.ndarray], float] = multivariate_kde(x, scotts_rule(x))
    local_max: np.ndarray = find_local_maxima(kde, x)
    max_neurons: int = local_max.shape[0]
    if max_neurons <= n_neurons:
        raise ValueError(f"Maximum number of neurons is {max_neurons}")
    return local_max[np.random.choice(max_neurons, size=n_neurons, replace=False)]
