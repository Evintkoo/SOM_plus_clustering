"""
kmeans.py

This module contains an optimized implementation of the KMeans clustering algorithm, 
which partitions a dataset into a number of clusters by minimizing the variance within each cluster.

Performance optimizations:
- Numba JIT compilation for distance calculations and cluster assignments
- Vectorized operations for better performance
- Improved centroid initialization and updates
- Better convergence detection
- Reduced memory allocation

Classes:
    KMeans: An optimized class that implements the KMeans clustering algorithm.

Functions:
    None.
"""

from typing import List, Optional
import random
import numpy as np
from .utils import random_initiate, euc_distance

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

# JIT-optimized functions for KMeans
@jit(nopython=True, fastmath=True)
def euclidean_distance_squared_jit(x, y):
    """JIT-optimized squared Euclidean distance calculation."""
    total = 0.0
    for i in range(len(x)):
        diff = x[i] - y[i]
        total += diff * diff
    return total

@jit(nopython=True, fastmath=True)
def assign_clusters_jit(data, centroids):
    """JIT-optimized cluster assignment."""
    n_samples = data.shape[0]
    n_clusters = centroids.shape[0]
    labels = np.zeros(n_samples, dtype=np.int32)
    
    for i in prange(n_samples):
        min_dist = np.inf
        min_idx = 0
        
        for j in range(n_clusters):
            dist = euclidean_distance_squared_jit(data[i], centroids[j])
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        
        labels[i] = min_idx
    
    return labels

@jit(nopython=True, fastmath=True)
def update_centroids_jit(data, labels, n_clusters):
    """JIT-optimized centroid update."""
    n_features = data.shape[1]
    new_centroids = np.zeros((n_clusters, n_features))
    cluster_counts = np.zeros(n_clusters)
    
    # Sum points for each cluster
    for i in range(data.shape[0]):
        cluster_id = labels[i]
        cluster_counts[cluster_id] += 1
        for j in range(n_features):
            new_centroids[cluster_id, j] += data[i, j]
    
    # Compute means
    for i in range(n_clusters):
        if cluster_counts[i] > 0:
            for j in range(n_features):
                new_centroids[i, j] /= cluster_counts[i]
    
    return new_centroids

@jit(nopython=True, fastmath=True)
def compute_inertia_jit(data, labels, centroids):
    """JIT-optimized inertia computation."""
    inertia = 0.0
    for i in range(data.shape[0]):
        cluster_id = labels[i]
        inertia += euclidean_distance_squared_jit(data[i], centroids[cluster_id])
    return inertia

@jit(nopython=True, fastmath=True)
def kmeans_plus_plus_jit(data, n_clusters):
    """JIT-optimized KMeans++ initialization."""
    n_samples, n_features = data.shape
    centroids = np.zeros((n_clusters, n_features))
    
    # Choose first centroid randomly
    first_idx = np.random.randint(0, n_samples)
    for j in range(n_features):
        centroids[0, j] = data[first_idx, j]
    
    # Choose remaining centroids
    for c_id in range(1, n_clusters):
        distances = np.full(n_samples, np.inf)
        
        # Compute distance to nearest existing centroid
        for i in range(n_samples):
            for j in range(c_id):
                dist = euclidean_distance_squared_jit(data[i], centroids[j])
                if dist < distances[i]:
                    distances[i] = dist
        
        # Choose next centroid with probability proportional to squared distance
        total_dist = np.sum(distances)
        if total_dist > 0:
            r = np.random.random() * total_dist
            cumsum = 0.0
            next_centroid_idx = 0
            for i in range(n_samples):
                cumsum += distances[i]
                if cumsum >= r:
                    next_centroid_idx = i
                    break
        else:
            next_centroid_idx = np.random.randint(0, n_samples)
        
        for j in range(n_features):
            centroids[c_id, j] = data[next_centroid_idx, j]
    
    return centroids

class KMeans:
    """
    Optimized KMeans clustering algorithm with vectorized operations.

    Attributes:
        n_clusters (int): Number of centroids.
        centroids (np.ndarray): Array of centroid vectors.
        _trained (bool): Indicates if the model has been trained.
        method (str): Method for centroid initialization.
        tol (float): Tolerance for convergence detection.
        max_iters (int): Maximum number of iterations.
    """

    def __init__(self, n_clusters: int, method: str, tol: float = 1e-6, max_iters: int = 300) -> None:
        """
        Initialize the KMeans clustering parameters.

        Args:
            n_clusters (int): Number of centroids for KMeans.
            method (str): Method for initializing centroids.
            tol (float): Tolerance for convergence detection.
            max_iters (int): Maximum number of iterations.
        """
        self.n_clusters: int = n_clusters
        self.centroids: np.ndarray = np.array([])
        self._trained: bool = False
        self.method: str = method
        self.tol: float = tol
        self.max_iters: int = max_iters
        self.inertia_: float = 0.0
        self.n_iter_: int = 0

    def initiate_plus_plus(self, x: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using the optimized KMeans++ algorithm.

        Args:
            x (np.ndarray): Input data matrix.

        Returns:
            np.ndarray: Array of centroids for KMeans clustering.
        """
        if _USING_NUMBA:
            # Use JIT-optimized implementation
            return kmeans_plus_plus_jit(x, self.n_clusters)
        else:
            # Use original NumPy implementation
            n_samples, n_features = x.shape
            centroids = np.empty((self.n_clusters, n_features))
            
            # Choose first centroid randomly
            centroids[0] = x[np.random.randint(n_samples)]
            
            # Choose remaining centroids
            for c_id in range(1, self.n_clusters):
                # Compute squared distances to nearest existing centroid for each point
                # Vectorized computation for all points at once
                distances = np.full(n_samples, np.inf)
                for i, centroid in enumerate(centroids[:c_id]):
                    # Compute squared distances efficiently
                    current_distances = np.sum((x - centroid)**2, axis=1)
                    distances = np.minimum(distances, current_distances)
                
                # Choose next centroid with probability proportional to squared distance
                probabilities = distances / distances.sum()
                cumulative_probs = probabilities.cumsum()
                r = np.random.rand()
                next_centroid_idx = np.searchsorted(cumulative_probs, r)
                centroids[c_id] = x[next_centroid_idx]
            
            return centroids

    def init_centroids(self, x: np.ndarray) -> None:
        """
        Initialize centroids for KMeans clustering using vectorized operations.

        Args:
            x (np.ndarray): Input data matrix.

        Raises:
            ValueError: If the initialization method is not recognized.
        """
        if self.method == "random":
            # Vectorized random initialization
            min_vals = np.min(x, axis=0)
            max_vals = np.max(x, axis=0)
            self.centroids = np.random.uniform(
                min_vals, max_vals, size=(self.n_clusters, x.shape[1])
            )
        elif self.method == "kmeans++":
            self.centroids = self.initiate_plus_plus(x)
        else:
            raise ValueError(f"Unrecognized method: {self.method}")

    def _assign_clusters(self, x: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the nearest centroid using optimized operations.

        Args:
            x (np.ndarray): Input data matrix.

        Returns:
            np.ndarray: Cluster assignments for each data point.
        """
        if _USING_NUMBA:
            # Use JIT-optimized implementation
            return assign_clusters_jit(x, self.centroids)
        else:
            # Use vectorized NumPy implementation
            # Compute all distances at once using broadcasting
            # x: (n_samples, n_features), centroids: (n_clusters, n_features)
            # distances: (n_samples, n_clusters)
            distances = np.linalg.norm(x[:, np.newaxis] - self.centroids, axis=2)
            return np.argmin(distances, axis=1)

    def _update_centroids(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids using optimized operations.

        Args:
            x (np.ndarray): Input data matrix.
            labels (np.ndarray): Current cluster assignments.

        Returns:
            np.ndarray: Updated centroids.
        """
        if _USING_NUMBA:
            # Use JIT-optimized implementation
            return update_centroids_jit(x, labels, self.n_clusters)
        else:
            # Use vectorized NumPy implementation
            new_centroids = np.empty_like(self.centroids)
            for k in range(self.n_clusters):
                # Find points assigned to cluster k
                cluster_mask = labels == k
                if np.any(cluster_mask):
                    # Compute mean of assigned points
                    new_centroids[k] = np.mean(x[cluster_mask], axis=0)
                else:
                    # Keep old centroid if no points assigned
                    new_centroids[k] = self.centroids[k]
            return new_centroids

    def _compute_inertia(self, x: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute within-cluster sum of squared distances (inertia).

        Args:
            x (np.ndarray): Input data matrix.
            labels (np.ndarray): Cluster assignments.

        Returns:
            float: Inertia value.
        """
        if _USING_NUMBA:
            # Use JIT-optimized implementation
            return compute_inertia_jit(x, labels, self.centroids)
        else:
            # Use vectorized NumPy implementation
            inertia = 0.0
            for k in range(self.n_clusters):
                cluster_mask = labels == k
                if np.any(cluster_mask):
                    cluster_points = x[cluster_mask]
                    centroid = self.centroids[k]
                    inertia += np.sum((cluster_points - centroid) ** 2)
            return inertia

    def fit(self, x: np.ndarray) -> None:
        """
        Train the optimized KMeans model using vectorized operations.

        Args:
            x (np.ndarray): Input data matrix.

        Raises:
            RuntimeError: If the model has already been trained.
        """
        if self._trained:
            raise RuntimeError("Cannot fit an already trained model.")

        # Initialize centroids
        self.init_centroids(x)
        
        prev_inertia = np.inf
        
        # Main optimization loop
        for iteration in range(self.max_iters):
            # Assign clusters
            labels = self._assign_clusters(x)
            
            # Update centroids
            new_centroids = self._update_centroids(x, labels)
            
            # Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            
            # Compute inertia for convergence checking
            current_inertia = self._compute_inertia(x, labels)
            
            # Update centroids
            self.centroids = new_centroids
            
            # Check convergence conditions
            if centroid_shift < self.tol or abs(prev_inertia - current_inertia) < self.tol:
                break
                
            prev_inertia = current_inertia
        
        self.inertia_ = current_inertia
        self.n_iter_ = iteration + 1
        self._trained = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels using optimized vectorized operations.

        Args:
            x (np.ndarray): Input data matrix.

        Returns:
            np.ndarray: Cluster labels for each data point.
        """
        if not self._trained:
            raise RuntimeError("Model must be fitted before prediction.")
        
        return self._assign_clusters(x)
