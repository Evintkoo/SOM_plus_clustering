"""
This module implements the Self-Organizing Map (SOM) algorithm for unsupervised learning.
It includes methods for initializing the SOM, training it on input data,
and visualizing the results.
"""
import math
import pickle
from typing import List, Tuple, Union
import numpy as np
import joblib
from joblib import Parallel, delayed

from .evals import (
    accuracy, f1_score, recall
)
from .utils import find_most_edge_point, cos_distance
from .kde_kernel import initiate_kde
from .kmeans import KMeans
from .variables import INITIATION_METHOD_LIST, DISTANCE_METHOD_LIST, CLASSIFICATION_EVAL_METHOD_LIST

def validate_configuration(initiate_method: str,
                        learning_rate: float,
                        distance_function: str) -> None:
    """Validate input parameters."""
    if learning_rate > 1.76:
        raise ValueError("Learning rate should be less than 1.76")
    if initiate_method not in INITIATION_METHOD_LIST:
        raise ValueError(f"Invalid initiation method: {initiate_method}")
    if distance_function not in DISTANCE_METHOD_LIST:
        raise ValueError(f"Invalid distance function: {distance_function}")

def initiate_plus_plus(m: int, n:int, x: np.ndarray) -> np.ndarray:
    """Initialize centroids using SOM++ algorithm."""
    centroids: List[np.ndarray] = [find_most_edge_point(x)]
    k: int = m * n
    dist_sq: np.ndarray = np.sum((x - centroids[0])**2, axis=1)

    for _ in range(1, k):
        furthest_data_idx: int = int(np.argmax(dist_sq))
        centroids.append(x[furthest_data_idx])
        dist_sq = np.minimum(dist_sq, np.sum((x - centroids[-1])**2, axis=1))

    return np.array(centroids)

class SOM:
    """
    Self-Organizing Map (SOM) implementation.
    """

    def __init__(self, m: int, n: int,
                dim: int, initiate_method: str,
                learning_rate: float, neighbour_rad: int,
                distance_function: str, max_iter: Union[int, float] = np.inf
                ) -> None:
        """
        Initialize the SOM.

        Args:
            m (int): Height of the matrix.
            n (int): Width of the matrix.
            dim (int): Input dimension of matrix.
            initiate_method (str): Neurons initiation method.
            learning_rate (float): Initial learning rate.
            neighbour_rad (int): Initial neighbourhood radius.
            distance_function (str): Distance function for BMU calculation.
            max_iter (int, optional): Maximum number of iterations. Defaults to None.
        """
        validate_configuration(initiate_method, learning_rate, distance_function)

        self.m: int = m
        self.n: int = n
        self.dim: int = dim
        self.shape: Tuple[int, int, int] = (m, n, dim)
        self.max_iter: Union[int, float] = max_iter if max_iter is not None else np.inf
        self.init_method: str = initiate_method
        self.dist_func: str = distance_function

        self.cur_learning_rate: float = learning_rate
        self.initial_learning_rate: float = learning_rate

        self.cur_neighbour_rad: float = neighbour_rad
        self.initial_neighbour_rad: float = neighbour_rad

        self.neurons: Union[np.ndarray] = np.ndarray([])
        self.neuron_label: np.array = np.ndarray([]) 
        self.initial_neurons: Union[np.ndarray] = np.ndarray([])
        self._trained: bool = False

    def initiate_neuron(self, data: np.ndarray) -> np.ndarray:
        """Initiate initial value of the neurons."""
        min_val: float = data.min()
        max_val: float = data.max()

        if self.init_method == "random":
            return np.random.uniform(min_val, max_val, self.shape)
        if self.init_method == "kde":
            neurons: np.ndarray = initiate_kde(x=data, n_neurons=self.m * self.n)
            return neurons.reshape(self.shape)
        if self.init_method in ["kmeans", "kde_kmeans", "kmeans++"]:
            model: KMeans = KMeans(n_clusters=(self.m * self.n), method=self.init_method)
            model.fit(x=data)
            return np.array(np.array(model.centroids).reshape(self.shape))
        if self.init_method == "som++":
            plus_plus_neurons: np.array = initiate_plus_plus(m = self.m, n = self.n, x = data)
            return plus_plus_neurons.reshape(self.shape)
        raise ValueError(f"Invalid initiation method: {self.init_method}")

    def index_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """Find the index of best matching unit among all neurons."""
        neurons: np.ndarray = self.neurons.reshape(-1, self.dim)
        distances: np.ndarray
        if self.dist_func == "euclidean":
            distances = np.linalg.norm(neurons - x, axis=1)
        elif self.dist_func == "cosine":
            distances = (1 - np.dot(neurons, x) /
                         (np.linalg.norm(neurons, axis=1) * np.linalg.norm(x)))
        min_index: int = int(np.argmin(distances))
        return tuple(np.unravel_index(min_index, (self.m, self.n)))

    def gaussian_neighbourhood(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate the Gaussian neighbourhood function."""
        omega: float = 1e-9
        lr: float = self.cur_learning_rate
        nr: float = max(self.cur_neighbour_rad, omega)  # Avoid division by zero
        dist_squared: int = (x1 - x2) ** 2 + (y1 - y2) ** 2
        return lr * math.exp(-0.5 * dist_squared / (nr * nr))

    def update_neuron(self, x: np.ndarray) -> None:
        """Update neurons based on the input data."""
        index_bmu = self.index_bmu(x=x)
        col_bmu: int = index_bmu[0]
        row_bmu: int = index_bmu[1]

        for cur_col in range(self.m):
            for cur_row in range(self.n):
                h: float = self.gaussian_neighbourhood(col_bmu, row_bmu, cur_col, cur_row)
                if h > 0:
                    if self.dist_func == "euclidean":
                        self.neurons[cur_col, cur_row] += h * (x - self.neurons[cur_col, cur_row])
                    elif self.dist_func == "cosine":
                        angle: float = cos_distance(x, self.neurons[cur_col, cur_row])
                        self.neurons[cur_col, cur_row] += (h *
                                                           (np.cos(angle) *
                                                            self.neurons[cur_col, cur_row]
                                                            - self.neurons[cur_col, cur_row]))

    def _worker(self, x: np.ndarray, epoch: int, shuffle: bool) -> np.ndarray:
        """Worker function for parallel processing."""
        neurons: np.ndarray = self.neurons.copy()
        n_sample: int = x.shape[0]
        total_iteration: int = int(min(epoch * n_sample, self.max_iter))

        for epoch_num in range(epoch):
            if shuffle:
                np.random.shuffle(x)

            for idx, data_point in enumerate(x):
                global_iter: int = epoch_num * n_sample + idx + 1
                if global_iter > self.max_iter:
                    break

                # Update neuron
                index_bmu = self.index_bmu(data_point)
                col_bmu : int = index_bmu[0]
                row_bmu : int = index_bmu[1]

                for cur_col in range(self.m):
                    for cur_row in range(self.n):
                        h: float = self.gaussian_neighbourhood(col_bmu, row_bmu, cur_col, cur_row)
                        if h > 0:
                            if self.dist_func == "euclidean":
                                neurons[cur_col, cur_row] += (h *
                                                              (data_point -
                                                               neurons[cur_col, cur_row]))
                            elif self.dist_func == "cosine":
                                angle: float = cos_distance(data_point, neurons[cur_col, cur_row])
                                neurons[cur_col, cur_row] += (h *
                                                              (np.cos(angle) *
                                                               neurons[cur_col, cur_row]
                                                               - neurons[cur_col, cur_row]))

                # Update learning rate and neighbourhood radius
                power: float = global_iter / total_iteration
                self.cur_learning_rate = (self.initial_learning_rate *
                                          (1 - power)
                                          * math.exp(-global_iter / self.initial_learning_rate))
                self.cur_neighbour_rad = (self.initial_neighbour_rad *
                                          (1 - power) *
                                          math.exp(-global_iter / self.initial_neighbour_rad))

        return neurons

    def fit(self, x: np.ndarray, y: np.ndarray,
            epoch: int, shuffle: bool = True,
            verbose: bool = True, n_jobs: int = -1) -> None:
        """
        Fit the SOM to the input data.

        Args:
            X (np.ndarray): Input data.
            epoch (int): Number of training iterations.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            verbose (bool, optional): Whether to show progress bar. Defaults to True.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (all available cores).
        """
        if not self._trained:
            self.neurons = self.initiate_neuron(data=x)
            self.initial_neurons = self.neurons.copy()

        if x.shape[1] != self.dim:
            raise ValueError(f"X.shape[1] should be {self.dim}, but found {x.shape[1]}")

        # Split data for parallel processing
        n_splits: int = n_jobs if n_jobs > 0 else joblib.cpu_count()
        data_splits: List[np.ndarray] = np.array_split(x, n_splits)

        # Parallel processing
        results: List[np.ndarray] = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self._worker)(split, epoch, shuffle)
            for split in data_splits
        )

        # Combine results
        self.neurons = np.mean(results, axis=0)
        
        # Update learning rate and radius to reflect the final training state
        n_sample: int = x.shape[0]
        total_iteration: int = int(min(epoch * n_sample, self.max_iter))
        final_iter = min(total_iteration, self.max_iter)
        power: float = final_iter / total_iteration if total_iteration > 0 else 1.0
        
        self.cur_learning_rate = (self.initial_learning_rate *
                                  (1 - power) *
                                  math.exp(-final_iter / self.initial_learning_rate))
        self.cur_neighbour_rad = (self.initial_neighbour_rad *
                                  (1 - power) *
                                  math.exp(-final_iter / self.initial_neighbour_rad))
        
        # Calculate distances between neurons and data points in a vectorized way using numpy
        distances = np.linalg.norm(self.cluster_center_[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2)

        # Find the index of the closest data point for each neuron
        closest_data_indices = np.argmin(distances, axis=1)

        # Assign labels to neurons based on the closest data points
        self.neuron_label = np.array([y[idx] for idx in closest_data_indices]).reshape((self.m, self.n))

        # assign the neurons to classification
        self._trained = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for the input data.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if not self._trained:
            raise RuntimeError("SOM must be fitted before predicting")

        assert len(x.shape) == 2, f'X should have two dimensions, not {len(x.shape)}'
        assert x.shape[1] == self.dim, f'This SOM has dimension {self.dim} but Received {x.shape[1]}'

        labels: np.ndarray = np.array([self.index_bmu(i) for i in x])
        return [self.neuron_label[x, y] for x,y in labels]

    def fit_predict(self, x: np.ndarray, y:np.array, epoch: int,
                    shuffle: bool = True, verbose: bool = True,
                    n_jobs: int = -1) -> np.ndarray:
        """
        Fit the SOM and predict cluster labels for the input data.

        Args:
            X (np.ndarray): Input data.
            epoch (int): Number of training iterations.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            verbose (bool, optional): Whether to show progress bar. Defaults to True.
            n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (all available cores).

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        self.fit(x=x, y=y, epoch=epoch, shuffle=shuffle, verbose=verbose, n_jobs=n_jobs)
        return self.predict(x=x)

    def evaluate(self, x: np.ndarray, y:np.ndarray,
                 method: List[str]) -> Union[List[float], dict]:
        """
        Evaluate the SOM clustering.

        Args:
            X (np.ndarray): Input data.
            y_true (np.ndarray, optional): True labels. Defaults to None.
            method (List[str], optional): Evaluation methods. Defaults to ["silhouette"].

        Returns:
            Union[List[float], dict]: Evaluation scores.
        """
        if not set(method).issubset(set(CLASSIFICATION_EVAL_METHOD_LIST)):
            raise ValueError(f'Invalid evaluation method(s): {set(method) - set(CLASSIFICATION_EVAL_METHOD_LIST)}')

        pred: np.ndarray = self.predict(x)
        # Map method names to their corresponding scoring functions
        score_functions = {
            "accuracy": accuracy,
            "f1_score": f1_score,
            "recall": recall
        }
        if "all" not in method:
            return [
                score_functions[m](y_true= y, y_pred=pred)
                for m in method if m in ["accuracy", "f1_score",
                                         "recall"]
            ]
        return {
            "accuracy": accuracy(y_true= y, y_pred=pred),
            "f1_score": f1_score(y_true= y, y_pred=pred),
            "recall": recall(y_true= y, y_pred=pred),
        }


    @property
    def cluster_center_(self) -> np.ndarray:
        """Get the cluster centers."""
        return self.neurons.reshape(-1, self.dim)

    def save(self, path: str) -> None:
        """Save the SOM model to a file."""
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str) -> 'SOM':
        """Load a SOM model from a file."""
        with open(path, 'rb') as file:
            return pickle.load(file)
