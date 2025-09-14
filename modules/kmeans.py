"""
kmeans.py

This module contains an optimized implementation of the KMeans clustering algorithm, 
which partitions a dataset into a number of clusters by minimizing the variance within each cluster.

This version includes significant performance optimizations:
- Vectorized operations for better performance
- Improved centroid initialization and updates
- Better convergence detection
- Reduced memory allocation

Classes:
    KMeans: An optimized class that implements the KMeans clustering algorithm.

Functions:
    None.

Usage Example:
    ```python
    import numpy as np
    from kmeans import KMeans

    # Sample data
    data = np.array([[1.0, 2.0], [2.0, 1.0], [4.0, 5.0], [5.0, 4.0]])

    # Initialize KMeans with 2 clusters and the 'kmeans++' initialization method
    kmeans = KMeans(n_clusters=2, method='kmeans++')

    # Train the KMeans model on the data
    kmeans.fit(data)

    # Predict the cluster of each data point
    labels = kmeans.predict(data)

    print("Cluster Labels:", labels)
    ```

Details:
    - The KMeans algorithm aims to partition n observations into k clusters 
    in which each observation belongs to the cluster with the nearest mean.
    - The optimized `KMeans` class provides vectorized operations for better performance
    and includes convergence detection for early stopping.

Module Dependencies:
    - numpy: Used for numerical operations and data manipulation.
    - random: Used for random selection and initialization.
    - typing: Provides support for type hints.
    - utils (user-defined module): Contains utility functions such as
    `random_initiate` and `euc_distance`.
"""

from typing import List, Optional
import random
import numpy as np
from .utils import random_initiate, euc_distance

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
        Assign each data point to the nearest centroid using vectorized operations.

        Args:
            x (np.ndarray): Input data matrix.

        Returns:
            np.ndarray: Cluster assignments for each data point.
        """
        # Compute all distances at once using broadcasting
        # x: (n_samples, n_features), centroids: (n_clusters, n_features)
        # distances: (n_samples, n_clusters)
        distances = np.linalg.norm(x[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Update centroids using vectorized operations.

        Args:
            x (np.ndarray): Input data matrix.
            labels (np.ndarray): Current cluster assignments.

        Returns:
            np.ndarray: Updated centroids.
        """
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
