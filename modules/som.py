"""
This module implements the Self-Organizing Map (SOM) algorithm for unsupervised learning.
It includes methods for initializing the SOM, training it on input data,
and visualizing the results.
"""
import math
import pickle
from typing import List, Tuple, Union
import numpy as np
import cupy as cp
import joblib
from joblib import Parallel, delayed
from .initialization import *

from .evals import (
    silhouette_score, davies_bouldin_index, calinski_harabasz_score,
    dunn_index #, compare_distribution, bcubed_precision_recall
)
from .utils import find_most_edge_point, cos_distance
from .kde_kernel import initiate_kde
from .kmeans import KMeans
from .variables import INITIATION_METHOD_LIST, DISTANCE_METHOD_LIST, EVAL_METHOD_LIST

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
import math
import pickle
from typing import List, Tuple, Union

import cupy as cp
import numpy as np  # For interfacing with CPU-based evaluation functions

# Assumed external functions; ensure these either work with or convert to/from CuPy as needed.
# from external_inits import (initiate_kde, initiate_plus_plus, initiate_zero,
#                             initiate_he, initiate_naive_sharding, initiate_lecun, initiate_lsuv)
# from external_kmeans import KMeans
# from external_eval import (silhouette_score, davies_bouldin_index,
#                            calinski_harabasz_score, dunn_index)
# from external_utils import validate_configuration, cos_distance, EVAL_METHOD_LIST

# For demonstration, here is a simple cosine distance implementation that works on GPU arrays.


class SOM:
    """
    Self-Organizing Map (SOM) implementation with GPU acceleration using CuPy.
    This version does not use Joblib for parallelism; all computation is performed using CuPy.
    """

    def __init__(self, m: int, n: int,
                 dim: int, initiate_method: str,
                 learning_rate: float, neighbour_rad: int,
                 distance_function: str, max_iter: Union[int, float] = np.inf
                 ) -> None:
        """
        Initialize the SOM.

        Args:
            m (int): Height of the grid.
            n (int): Width of the grid.
            dim (int): Dimensionality of input data.
            initiate_method (str): Method for neuron initialization.
            learning_rate (float): Initial learning rate (should be in (0,1]).
            neighbour_rad (int): Initial neighbourhood radius.
            distance_function (str): Distance function ("euclidean" or "cosine").
            max_iter (int, optional): Maximum number of iterations. Defaults to np.inf.
        """
        # Validate configuration parameters (this should raise ValueError for invalid configurations)
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

        # Neurons will be stored as a CuPy array.
        self.neurons: cp.ndarray = cp.empty((0,))
        self.initial_neurons: cp.ndarray = cp.empty((0,))
        self._trained: bool = False

    def initiate_neuron(self, data: np.ndarray) -> cp.ndarray:
        """
        Initialize neuron weights using the specified method.

        Args:
            data (np.ndarray): Input data on CPU.

        Returns:
            cp.ndarray: Neuron weights on GPU.
        """
        # Convert input data to GPU array.
        data_gpu: cp.ndarray = cp.asarray(data)
        min_val: float = cp.min(data_gpu).item()
        max_val: float = cp.max(data_gpu).item()

        if self.init_method == "random":
            neurons = cp.random.uniform(min_val, max_val, self.shape)
            return neurons
        elif self.init_method == "kde":
            # Assume initiate_kde returns a NumPy array.
            neurons = initiate_kde(x=data, n_neurons=self.m * self.n)
            return cp.asarray(neurons).reshape(self.shape)
        elif self.init_method in ["kmeans", "kde_kmeans", "kmeans++"]:
            model = KMeans(n_clusters=(self.m * self.n), method=self.init_method)
            model.fit(x=data)
            return cp.asarray(model.centroids).reshape(self.shape)
        elif self.init_method == "som++":
            plus_plus_neurons = initiate_plus_plus(m=self.m, n=self.n, x=data)
            return cp.asarray(plus_plus_neurons).reshape(self.shape)
        elif self.init_method == "zero":
            neurons = initiate_zero(P=self.m * self.n, Q=self.dim)
            return cp.asarray(neurons).reshape(self.shape)
        elif self.init_method == "he":
            neurons = initiate_he(input_dim=self.dim, output_dim=self.m * self.n)
            return cp.asarray(neurons).reshape(self.shape)
        elif self.init_method == "naive_sharding":
            neurons = initiate_naive_sharding(X=data, k=self.m * self.n)
            return cp.asarray(neurons).reshape(self.shape)
        elif self.init_method == "lecun":
            neurons = initiate_lecun(input_shape=self.dim, output_shape=self.m * self.n)
            return cp.asarray(neurons).reshape(self.shape)
        elif self.init_method == "lsuv":
            neurons = initiate_lsuv(input_dim=self.dim, output_dim=self.m * self.n, X_batch=data)
            return cp.asarray(neurons).reshape(self.shape)

        raise ValueError(f"Invalid initiation method: {self.init_method}")
    def index_bmu(self, x: cp.ndarray) -> Tuple[int, int]:
        """
        Find the index of the best matching unit (BMU) among all neurons.

        Args:
            x (cp.ndarray): Input data point (on GPU).

        Returns:
            Tuple[int, int]: The indices (row, column) of the BMU.
        """
        neurons_flat = self.neurons.reshape(-1, self.dim)
        if self.dist_func == "euclidean":
            distances = cp.linalg.norm(neurons_flat - x, axis=1)
        elif self.dist_func == "cosine":
            norm_neurons = cp.linalg.norm(neurons_flat, axis=1)
            norm_x = cp.linalg.norm(x)
            distances = 1 - (cp.dot(neurons_flat, x) / (norm_neurons * norm_x + 1e-9))
        else:
            raise ValueError(f"Unsupported distance function: {self.dist_func}")

        # Ensure the index is wrapped as a CuPy array before calling cp.unravel_index.
        min_index = int(cp.argmin(distances).item())
        bmu_indices = cp.unravel_index(cp.array(min_index), (self.m, self.n))
        # Convert the resulting indices to Python integers.
        return int(bmu_indices[0].item()), int(bmu_indices[1].item())


    def _vectorized_update(self, neurons: cp.ndarray, data_point: cp.ndarray) -> cp.ndarray:
        """
        Perform a vectorized update of the neurons given one data point.

        Args:
            neurons (cp.ndarray): Neuron weights (shape: (m, n, dim)).
            data_point (cp.ndarray): Single input data point (shape: (dim,)).

        Returns:
            cp.ndarray: Updated neuron weights.
        """
        # Find the BMU for the data point.
        bmu_row, bmu_col = self.index_bmu(x=data_point)
        # Create a grid of coordinates.
        grid_rows = cp.arange(self.m).reshape(self.m, 1)
        grid_cols = cp.arange(self.n).reshape(1, self.n)
        # Compute squared distances from each neuron to the BMU.
        dist_squared = (grid_rows - bmu_row) ** 2 + (grid_cols - bmu_col) ** 2
        # Avoid division by zero.
        nr = max(self.cur_neighbour_rad, 1e-9)
        # Compute the neighborhood function over the grid.
        h = self.cur_learning_rate * cp.exp(-0.5 * dist_squared / (nr * nr))
        # Expand h to match neuron shape.
        h_expanded = h[:, :, cp.newaxis]  # shape (m, n, 1)
        if self.dist_func == "euclidean":
            neurons = neurons + h_expanded * (data_point - neurons)
        elif self.dist_func == "cosine":
            # For cosine distance, update rule: h * (cos(angle) * neuron - neuron)
            neuron_norm = cp.linalg.norm(neurons, axis=2, keepdims=True) + 1e-9
            x_norm = cp.linalg.norm(data_point) + 1e-9
            dot_product = cp.sum(neurons * data_point, axis=2, keepdims=True)
            cos_angle = dot_product / (neuron_norm * x_norm)
            neurons = neurons + h_expanded * (cos_angle * neurons - neurons)
        return neurons

    def fit(self, x: np.ndarray, epoch: int, shuffle: bool = True) -> None:
        """
        Fit the SOM to the input data using only CuPy for computation.
        
        This method operates sequentially without parallelization.
        
        Args:
            x (np.ndarray): Input data (CPU, NumPy array).
            epoch (int): Number of epochs.
            shuffle (bool, optional): Whether to shuffle data each epoch.
        """
        # Initialize neurons if not already trained.
        if not self._trained:
            self.neurons = self.initiate_neuron(data=x)
            self.initial_neurons = self.neurons.copy()

        if x.shape[1] != self.dim:
            raise ValueError(f"X.shape[1] should be {self.dim}, but found {x.shape[1]}")

        # Convert input data to GPU array.
        data_gpu: cp.ndarray = cp.asarray(x)
        n_sample: int = data_gpu.shape[0]
        total_iterations = int(min(epoch * n_sample, self.max_iter))

        global_iter = 0
        for epoch_num in range(epoch):
            # Optionally shuffle the data (shuffle on GPU via CPU conversion)
            if shuffle:
                data_cpu = cp.asnumpy(data_gpu)
                np.random.shuffle(data_cpu)
                data_gpu = cp.asarray(data_cpu)

            for idx in range(n_sample):
                global_iter += 1
                if global_iter > total_iterations:
                    break

                data_point = data_gpu[idx]
                # Update neurons using vectorized update.
                self.neurons = self._vectorized_update(self.neurons, data_point)

                # Update learning rate and neighborhood radius with decay.
                power = global_iter / total_iterations
                self.cur_learning_rate = self.initial_learning_rate * (1 - power) * math.exp(-global_iter / self.initial_learning_rate)
                self.cur_neighbour_rad = self.initial_neighbour_rad * (1 - power) * math.exp(-global_iter / self.initial_neighbour_rad)

        self._trained = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for the input data.

        Args:
            x (np.ndarray): Input data (CPU, NumPy array).

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if not self._trained:
            raise RuntimeError("SOM must be fitted before predicting")

        if len(x.shape) != 2:
            raise ValueError(f"X should have two dimensions, not {len(x.shape)}")
        if x.shape[1] != self.dim:
            raise ValueError(f"This SOM has dimension {self.dim} but received {x.shape[1]}")

        labels = []
        for sample in x:
            sample_gpu = cp.asarray(sample)
            bmu = self.index_bmu(sample_gpu)
            # Compute a unique label from the 2D BMU index.
            labels.append(self.m * bmu[0] + bmu[1])
        return np.array(labels)

    def fit_predict(self, x: np.ndarray, epoch: int, shuffle: bool = True) -> np.ndarray:
        """
        Fit the SOM and then predict cluster labels for the input data.

        Args:
            x (np.ndarray): Input data (CPU, NumPy array).
            epoch (int): Number of epochs.
            shuffle (bool, optional): Whether to shuffle data each epoch.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        self.fit(x=x, epoch=epoch, shuffle=shuffle)
        return self.predict(x=x)

    def evaluate(self, x: np.ndarray, method: List[str]) -> Union[List[float], dict]:
        """
        Evaluate the SOM clustering.

        Args:
            x (np.ndarray): Input data (CPU, NumPy array).
            method (List[str]): List of evaluation methods.

        Returns:
            Union[List[float], dict]: Evaluation scores.
        """
        if not set(method).issubset(set(EVAL_METHOD_LIST)):
            raise ValueError(f'Invalid evaluation method(s): {set(method) - set(EVAL_METHOD_LIST)}')

        pred: np.ndarray = self.predict(x)
        score_functions = {
            "silhouette": silhouette_score,
            "davies_bouldin": davies_bouldin_index,
            "calinski_harabasz": calinski_harabasz_score,
            "dunn": dunn_index
        }
        if "all" not in method:
            return [score_functions[m](x=x, labels=pred) for m in method if m in score_functions]
        return {
            "silhouette": silhouette_score(x=x, labels=pred),
            "davies_bouldin": davies_bouldin_index(x=x, labels=pred),
            "calinski_harabasz": calinski_harabasz_score(x=x, labels=pred),
            "dunn": dunn_index(x=x, labels=pred)
        }

    @property
    def cluster_center_(self) -> np.ndarray:
        """
        Get the cluster centers as a NumPy array.
        """
        return cp.asnumpy(self.neurons.reshape(-1, self.dim))

    def save(self, path: str) -> None:
        """
        Save the SOM model to a file.
        """
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str) -> 'SOM':
        """
        Load a SOM model from a file.
        """
        with open(path, 'rb') as file:
            return pickle.load(file)
