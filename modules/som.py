"""
    References :
    Chaudhary, V., Bhatia, R. S., & Ahlawat, A. K. (2014). A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons. Alexandria Engineering Journal, 53(4), 827-831. https://doi.org/10.1016/j.aej.2014.09.007
"""

import multiprocessing
import numpy as np
import math 
from .evals import silhouette_score, davies_bouldin_index, calinski_harabasz_score, dunn_index, compare_distribution
from .utils import random_initiate, find_most_edge_point, cos_distance
from .kde_kernel import initiate_kde
from .kmeans import KMeans
from .variables import INITIATION_METHOD_LIST, DISTANCE_METHOD_LIST, EVAL_METHOD_LIST
from tqdm import tqdm

# Self Organizing Matrix Class
class SOM(): 
    """
    SOM class is consist of:
        SOM.m (int): height of the matrix.
        SOM.n (int): width of the matrix.
        SOM.dim (int): input dimension of matrix.
        SOM.max_iter (int): maximum number of iteration for each training iteration.
        SOM.shape (tuple): shape of the matrix
        SOM.cur_learning_rate (float): current learning rate of matrix (a(t))
        SOM.initial_learning_rate (float): defined learning rate of SOM (a(0))
        SOM._trained (bool): status of the model
        SOM.init_method (str): neurons initiation method 
        self.cur_neighbor_rad (float): current neighbor radius of SOM (g(t))
        self.initial_neighbor_rad (float): initial neighbourhood radius of SOM (g(o))
        SOM.neurons (np.ndarray): value of neurons in the matrix, none if SOM.fit() have not called yet
        SOM.initial_neurons (np.ndarray): initial value of the neurons, none if SOM.fit() have not called yet
    """
    def __init__(self, m: int, n: int, dim: int, initiate_method:str, learning_rate:float, neighbour_rad: int, distance_function:str, max_iter=None) -> None:
        """
        Initiate the main parameter of Self Organizing Matrix Clustering

        Args:
            m (int): _description_
            n (int): _description_
            dim (int): _description_
            initiate_method (str): _description_
            max_iter (int): _description_
            learning_rate (float): _description_
            neighbour_rad (int): _description_
            distance_function (str): _description_

        Raises:
            ValueError: _description_
        
        Overall Time Complexity: O(1)
        """
        if learning_rate > 1.76:
            raise ValueError("Learning rate should be less than 1.76")
        if initiate_method not in INITIATION_METHOD_LIST:
            raise ValueError("There is no method called {}".format(initiate_method))
        if distance_function not in DISTANCE_METHOD_LIST:
            raise ValueError("There is no method called {}".format(initiate_method))
        
        if not max_iter:
            max_iter = np.inf
        # initiate all the attributes
        self.m = m
        self.n = n
        self.dim = dim
        self.max_iter = max_iter
        self.shape = (m,n,dim)
        self.cur_learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self._trained = False
        self.init_method = initiate_method
        self.dist_func = distance_function
        self.cur_neighbour_rad = neighbour_rad
        self.initial_neighbour_rad = neighbour_rad
        self.neurons = None 
        self.initial_neurons = None
        
    def initiate_plus_plus(self, X: np.ndarray):
        """
        Initiate the centroid value using kmeans++ algorithm
        Args:
            X (np.ndarray): Matrix of input data.
        Returns:
            np.ndarray: list of centroid for kmeans clustering.

        Overall Time Complexity: O(N*k*dim), k is number of cluster
        """

        # initiate empty list of centroids
        centroids = []

        # choose a random data point as the first centroid
        centroids.append(find_most_edge_point(X))

        # initiate the number of choices
        k = self.m * self.n

        # calculate the squared distances from each data point to the first centroid
        dist_sq = np.sum((X - centroids[0])**2, axis=1)

        # iterate k-1 times to fill the list of centroids
        for _ in range(k - 1):
            # choose the data point with the maximum squared distance as the next centroid
            furthest_data_idx = np.argmax(dist_sq)
            centroids.append(X[furthest_data_idx])

            # update the squared distances to the nearest centroid for each data point
            dist_sq = np.minimum(dist_sq, np.sum((X - centroids[-1])**2, axis=1))

        return np.array(centroids)
    
    def initiate_neuron(self, data: np.ndarray, min_val:float, max_val:float):
        """Initiate initial value of the neurons

        Args:
            min_val (float): the minimum value of the data input
            max_val (float): maximum value of the data input

        Raises:
            ValueError: There are no method named self.init_method or the method is not available yet

        Returns:
            list(): list of neurons to be initiate in self.neurons and self.initial_neurons
            
        Overall Time Complexity:
            self.init_method == "random": O(C), C is m*n*dim
            self.init_method == "kde": O(N*C), C is max(1/treshold, m*n*dim)
            self.init_method == "kmeans": O(N*C), C is max(1/treshold, m*n*dim)
            self.init_method == "kde_kmeans": O(N*C), C is max(1/treshold, m*n*dim)
            self.init_method == "kmeans++": O(N*C), C is k*k*dim
            self.init_method == "SOM++": O(N*C), C is k*k*dim
        """
        if self.init_method == "random" :
            # number of step = self.dim * self.m * self.n --> O(m * n * dim)
            return [[random_initiate(self.dim ,min_val=min_val, max_val=max_val) for j in range(self.m)] for i in range(self.n)]
        elif self.init_method == "kde" :
            neurons = initiate_kde(X=data, n_neurons=self.m*self.n)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        elif self.init_method == "kmeans":
            model = KMeans(n_clusters = (self.m * self.n), method="random")
            model.fit(X = data)
            neurons = model.centroids
            neurons = np.sort(neurons, axis=0)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        elif self.init_method == "kde_kmeans":
            model = KMeans(n_clusters = (self.m * self.n), method="kde")
            model.fit(X = data)
            neurons = model.centroids
            neurons = np.sort(neurons, axis=0)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        elif self.init_method == "kmeans++" :
            model = KMeans(n_clusters = (self.m * self.n), method="kmeans++")
            model.fit(X = data)
            neurons = model.centroids
            neurons = np.sort(neurons, axis=0)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        elif self.init_method == "SOM++" :
            neurons = self.initiate_plus_plus(data)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        else:
            raise ValueError("There is no method named {}".format(self.init_method))
    
    def index_bmu(self, x: np.array):
        """Find the index of best matching unit among all of neurons inside the matrix based on its euclidean distance

        Args:
            x (np.array): input array as comparison parameter

        Returns:
            tuple(): set of coordinates the best matching unit in (x,y) 
        
        Overall Time Complexity: O(m * n * dim)
        """
        neurons = np.reshape(self.neurons, (-1, self.dim)) # O(1)
        if self.dist_func == "euclidean":
            distances = np.linalg.norm(neurons - x, axis=1)
        elif self.dist_func == "cosine":
            distances = 1 - np.dot(neurons, x) / (np.linalg.norm(neurons, axis=1) * np.linalg.norm(x))
        min_index = np.argmin(distances)
        return np.unravel_index(min_index, (self.m, self.n)) # O(1)
    
    def gaussian_neighbourhood(self, x1, y1, x2, y2):
        """Represents gaussian function as the hyper parameter of updating weight of neurons
        Args:
            x1 (float): x coordinates of best matching unit
            y1 (float): y coordinates of best matching unit
            x2 (float): x coordinates of the neuron
            y2 (float): y coordinates of the neuron
        Returns:
            float: return the evaluation of h(t) = a(t) * exp(-||r_c - r_i||^2/(2 * o(t)^2))
    
        Overall Time Complexity: O(1)
        """
        lr = self.cur_learning_rate
        nr = self.cur_neighbour_rad
        nr = max(nr, 1e-9)  # Avoid division by zero
        dist_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
        exp = math.exp(-0.5 * dist_squared / (nr * nr))
        return lr * exp
    
    def update_neuron(self, x: np.array):
        """Update neurons based on the input data in each iteration
        Args:
            x (np.array): the input value from data
        
        Overall Complexity: O(m * n * dim)
        """
        # find index for the best matching unit index --> O(m * n * dim)
        col_bmu, row_bmu = self.index_bmu(x)
    
        # iterates through the matrix --> O(m * n * dim)
        for cur_col, col in enumerate(self.neurons):
            for cur_row, cur_weight in enumerate(col):
                # calculate the new weight, update only if the weight is > 0
                h = self.gaussian_neighbourhood(col_bmu, row_bmu, cur_col, cur_row)
                if h > 0:
                    if self.dist_func == "euclidean":
                        # new weight = cur weight + moving weight * distance
                        self.neurons[cur_col][cur_row] = cur_weight + h * (x - cur_weight)
                    elif self.dist_func == "cosine":
                        angle = cos_distance(x, cur_weight)
                        self.neurons[cur_col][cur_row] = [i + j * h for i, j in zip([math.cos(angle) * i for i in cur_weight], cur_weight)]
    
    def worker(self, X, epoch, shuffle, verbose, shared_neurons, shared_learning_rate, shared_neighbour_rad):
        neurons = np.frombuffer(shared_neurons.get_obj()).reshape((self.m, self.n, self.dim))
        cur_learning_rate = shared_learning_rate.value
        cur_neighbour_rad = shared_neighbour_rad.value
        
        n_sample = X.shape[0]
        total_iteration = min(epoch * n_sample, self.max_iter)
        
        global_iter_counter = 1
        for i in range(epoch):
            if global_iter_counter > self.max_iter:
                break
            
            if shuffle:
                np.random.shuffle(X)
            
            for idx in X:
                if global_iter_counter > self.max_iter:
                    break
                
                # Update neuron using the shared neurons array
                self.update_neuron(idx)
                
                global_iter_counter += 1
                power = global_iter_counter / total_iteration
                cur_learning_rate = self.initial_learning_rate ** (1 - power) * math.exp(-1 * global_iter_counter / self.initial_learning_rate)
                cur_neighbour_rad = self.initial_neighbour_rad ** (1 - power) * math.exp(-1 * global_iter_counter / self.initial_neighbour_rad)
        
        # Update the shared neurons array with the final state
        shared_neurons[:] = neurons.flatten()
        
    def fit_multi_process(self, X: np.ndarray, epoch: int, shuffle=True, verbose=True, num_processes=int(multiprocessing.cpu_count()*0.9)):
        if not self._trained:
            self.neurons = self.initiate_neuron(data=X, min_val=X.min(), max_val=X.max())
            self.initial_neurons = self.neurons
    
        if X.shape[1] != self.dim:
            raise ValueError("X.shape[1] should be the same as self.dim, but found {}".format(X.shape[1]))
        
        # Convert self.neurons to a NumPy array
        neurons_array = np.array(self.neurons)
        
        # Create shared arrays for neurons, learning rate, and neighbour radius
        shared_neurons = multiprocessing.Array('d', neurons_array.flatten())
        shared_learning_rate = multiprocessing.Value('d', self.initial_learning_rate)
        shared_neighbour_rad = multiprocessing.Value('d', self.initial_neighbour_rad)
        
        # Split the data into chunks for each process
        chunk_size = X.shape[0] // num_processes
        chunks = [X[i:i+chunk_size] for i in range(0, X.shape[0], chunk_size)]
        
        # Create a list to store the processes
        processes = []
        
        # Create and start the processes
        for i in range(num_processes):
            process = multiprocessing.Process(target=self.worker, args=(chunks[i], epoch, shuffle, verbose, shared_neurons, shared_learning_rate, shared_neighbour_rad))
            processes.append(process)
            process.start()
        
        # Wait for all processes to finish
        for process in processes:
            process.join()
        
        # Update the neurons with the final state from the shared array
        self.neurons = np.frombuffer(shared_neurons.get_obj()).reshape((self.m, self.n, self.dim)).tolist()
        
        self._trained = True
    
    def fit(self, X: np.ndarray, epoch: int, shuffle=True, verbose=True, use_multiprocessing=False):
        """Tune the neurons to learn the data
        Args:
            X (np.ndarray): Input data
            epoch (int): number of training iteration
            shuffle (bool, optional): the initate data to be evaluate in the matrix.
                Defaults to True.
        Raises:
            SyntaxError: SOM._trained() already true, which the model have been trained
            ValueError: The length of data columns is different with the length of the dimension
    
        Return:
            None: fit the neurons to the data
        
        Overall Time Complexity: O(epoch * N * N * C) -> worst case
        """
        if use_multiprocessing:
            self.fit_multi_process(X=X, epoch=epoch, shuffle=shuffle, verbose=verbose)
        if not self._trained:
            self.neurons = self.initiate_neuron(data=X, min_val=X.min(), max_val=X.max())  # O(m * n * dim)
            self.initial_neurons = self.neurons
    
        if X.shape[1] != self.dim:
            raise ValueError("X.shape[1] should be the same as self.dim, but found {}".format(X.shape[1]))
    
        n_sample = X.shape[0]
        total_iteration = min(epoch * n_sample, self.max_iter)
    
        global_iter_counter = 1
        for i in tqdm(range(epoch), disable=not verbose, desc=f'evaluation score: {compare_distribution(X, self.cluster_center_)}'):
            if global_iter_counter > self.max_iter:
                break
        
            if shuffle:
                np.random.shuffle(X)
        
            for idx in X:
                if global_iter_counter > self.max_iter:
                    break
            
                self.update_neuron(idx)  # O(m * n * dim)
                global_iter_counter += 1
                power = global_iter_counter / total_iteration
                self.cur_learning_rate = self.initial_learning_rate ** (1 - power) * math.exp(-1 * global_iter_counter / self.initial_learning_rate)
                self.cur_neighbour_rad = self.initial_neighbour_rad ** (1 - power) * math.exp(-1 * global_iter_counter / self.initial_neighbour_rad)
        
        self._trained = True
    
    def predict(self, X: np.ndarray):
        """Label the data based on the neurons using best matching unit
        Args:
            X (np.ndarray): input data to be predicted
        Raises:
            NotImplementedError: The model has not been trained yet, call SOM.fit() first
        Returns:
            np.array(): array of the labels of each data
            
        Overall Time Complexity: O(N * m * n * dim)
        """
        if not self._trained:
            raise NotImplementedError("SOM object should call fit() before predict()")
        
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimension {self.dim}. Received input with dimension {X.shape[1]}'
        
        labels = np.array([self.index_bmu(x) for x in X])
        labels = self.m * labels[:, 0] + labels[:, 1]
        return labels
    
    def fit_predict(self, X : np.ndarray, epoch : int, shuffle=True, verbose=True, use_multiprocessing=False):
        """Fit the model based on the data and return the prediciton of  the data

        Args:
            X (np.ndarray): Input data
            epoch (int): number of training iteration 
            shuffle (bool, optional): the initate data to be evaluate in the matrix. 
                Defaults to True.
        
        Returns:
            np.array(): the prediciton of each data
            
        Overall Time Complexity: O(epoch * N * m * n * dim)
        """
        self.fit(X = X, epoch = epoch, shuffle=shuffle, verbose=verbose, use_multiprocessing=use_multiprocessing) # O(epoch * N * m * n * dim)
        return self.predict(X=X) # O(N * m * n * dim)
    
    def evaluate(self, X:np.array, method:str="silhouette"):
        pred = self.predict(X)
        if method not in EVAL_METHOD_LIST:
            raise ValueError(f'{method} is not found in method list')
        
        if method == "silhouette":
            return silhouette_score(X=X, labels=pred)
        if method == "davies_bouldin":
            return davies_bouldin_index(X=X, labels=pred)
        if method == "calinski_harabasz":
            return calinski_harabasz_score(X=X, labels=pred)
        if method == "dunn":
            return dunn_index(X=X, labels=pred)
        if method == "all":
            return {"silhouette": silhouette_score(X=X, labels=pred),
                    "davies_bouldin": davies_bouldin_index(X=X, labels=pred), 
                    "calinski_harabasz": calinski_harabasz_score(X=X, labels=pred),
                    "dunn": dunn_index(X=X, labels=pred)}
    
    @property
    def cluster_center_(self):
        """Generate the list of all neurons

        Returns:
            np.ndarray(): list of all neurons with shape (m*n, dim)
        """
        return np.reshape(self.neurons, (-1, self.dim))