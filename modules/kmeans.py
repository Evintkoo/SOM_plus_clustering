"""
kmeans.py

This module contains the implementation of the KMeans clustering algorithm, 
which partitions a dataset into a number of clusters by minimizing the variance within each cluster.

Classes:
    KMeans: A class that implements the KMeans clustering algorithm.

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
    - The `KMeans` class provides functionality for initializing centroids 
    using either random or KMeans++ methods, updating centroids during training,
    and predicting the clusters for new data points.

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
from joblib import Parallel, delayed
from .utils import random_initiate, euc_distance

class KMeans:
    """
    KMeans clustering algorithm with parallel computing support.

    Attributes:
        n_clusters (int): Number of centroids.
        centroids (Optional[List[np.ndarray]]): List of centroid vectors.
        _trained (bool): Indicates if the model has been trained.
        method (str): Method for centroid initialization.
    """

    def __init__(self, n_clusters: int, method: str) -> None:
        """
        Initialize the KMeans clustering parameters.

        Args:
            n_clusters (int): Number of centroids for KMeans.
            method (str): Method for initializing centroids.
        """
        self.n_clusters: int = n_clusters
        self.centroids: np.ndarray = np.ndarray([])
        self._trained: bool = False
        self.method: str = method

    def initiate_plus_plus(self, x: np.ndarray, n_jobs: int) -> List[np.ndarray]:
        """
        Initialize centroids using the KMeans++ algorithm.

        Args:
            x (np.ndarray): Input data matrix.
            n_jobs (int): Number of parallel jobs.

        Returns:
            List[np.ndarray]: List of centroids for KMeans clustering.
        """
        centroids: List[np.ndarray] = [random.choice(x)]
        k: int = self.n_clusters

        for _ in range(k - 1):
            # Parallelize distance calculation
            dist_arr: List[float] = Parallel(n_jobs=n_jobs)(
                delayed(lambda i, centroids: min(euc_distance(i, c) ** 2 for c in centroids))(i, centroids)
                for i in x
            )
            furthest_data: np.ndarray = x[np.argmax(dist_arr)]
            centroids.append(furthest_data)

        return centroids

    def init_centroids(self, x: np.ndarray, n_jobs: int) -> None:
        """
        Initialize centroids for KMeans clustering.

        Args:
            x (np.ndarray): Input data matrix.
            n_jobs (int): Number of parallel jobs.

        Raises:
            ValueError: If the initialization method is not recognized.
        """
        if self.method == "random":
            self.centroids = Parallel(n_jobs=n_jobs)(
                delayed(random_initiate)(dim=x.shape[1], min_val=x.min(), max_val=x.max())
                for _ in range(self.n_clusters)
            )
        elif self.method == "kmeans++":
            self.centroids = self.initiate_plus_plus(x, n_jobs)
        else:
            raise ValueError(f"Unrecognized method: {self.method}")

    def update_centroids(self, x: np.ndarray, n_jobs: int) -> None:
        """
        Update the centroid values in a parallel manner.

        Args:
            x (np.ndarray): Input data matrix.
            n_jobs (int): Number of parallel jobs.
        """
        if self.centroids is None or x is None:
            raise ValueError("Centroids or X have not been initiated.")

        # Parallelize distance calculation
        distances: List[float] = Parallel(n_jobs=n_jobs)(
            delayed(euc_distance)(x, c) for c in self.centroids
        )
        
        # Find the index of the closest centroid
        closest_centroid_index: int = int(np.argmin(distances))
        
        # Get the closest centroid using the index
        closest_centroid: np.ndarray = self.centroids[closest_centroid_index]

        # Update centroid
        updated_centroid: np.ndarray = np.mean([closest_centroid, x], axis=0)
        self.centroids[closest_centroid_index] = updated_centroid

    def fit(self, x: np.ndarray, n_jobs: int = -1, epochs: int = 3000, shuffle: bool = True) -> None:
        """
        Train the KMeans model to find the best centroid values.

        Args:
            x (np.ndarray): Input data matrix.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (all cores).
            epochs (int, optional): Number of training iterations. Defaults to 3000.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        Raises:
            RuntimeError: If the model has already been trained.
        """
        if self._trained:
            raise RuntimeError("Cannot fit an already trained model.")

        # Initialize centroids
        self.init_centroids(x, n_jobs)

        # Train the model
        for _ in range(epochs):
            if shuffle:
                np.random.shuffle(x)

            # Parallelize centroid updates
            Parallel(n_jobs=n_jobs)(
                delayed(self.update_centroids)(i, n_jobs) for i in x
            )

        self._trained = True

    def predict(self, x: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        """
        Predict the cluster number for each data point in the input matrix using parallel processing.

        Args:
            x (np.ndarray): Input data matrix.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1.

        Returns:
            np.ndarray: Cluster labels for each data point.
        """
        # Compute the distance of each data point to all centroids in parallel
        cluster_labels = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(lambda i: int(np.argmin([euc_distance(i, c) for c in self.centroids])))(i)
                for i in x
            )
        )
        
        return cluster_labels
