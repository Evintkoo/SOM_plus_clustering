"""
    References :
    Chaudhary, V., Bhatia, R. S., & Ahlawat, A. K. (2014). A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons. Alexandria Engineering Journal, 53(4), 827-831. https://doi.org/10.1016/j.aej.2014.09.007
"""

import numpy as np
import math 
import random
from .utils import random_initiate, euc_distance, gauss, std_dev, kernel_gauss, deriv, render_bar, cos_distance
from .kmeans import KMeans
from .variables import INITIATION_METHOD_LIST, DISTANCE_METHOD_LIST

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
        
    def initiate_plus_plus(self, X : np.ndarray):
        """
        Initiate the centroid value using kmeans++ algorithm

        Args:
            X (np.ndarray): Matrix of input data.

        Returns:
            np.ndarray: list of centroid for kmeans clustering.
        
        Overall Time Complexity: O(N*k*k*dim), k is number of cluster
        """
        
        # initiate empty list of centroids
        centroids = list()
        
        # choose a random  data
        centroids.append(random.choice(X))
        
        # initiate the number of choices
        k = self.m * self.n
        
        # iterate k-1 number to fill the list of neurons
        for c in range(k-1):
            # find the minimum euclidean distance square from the all centroids in each data point
            dist_arr = [min([euc_distance(j, i)*euc_distance(j, i) for j in centroids]) for i in X]
            
            # find the furthest vector of data 
            furthest_data = X[np.argmax(dist_arr)]
            
            # append the data to the list of centroid
            centroids.append(furthest_data)
        return centroids
        
    def find_initial_centroid(self, X : np.ndarray, treshold:float):
        """
        Find the initial centroid using kernel density function peaks.

        Args:
            X (np.ndarray): Matrix of input data.
            treshold (float): h value for deriv().

        Returns:
            np.ndarray: list of centroids based on the peak of each variable kernel density function.
        
        Overall Time Complexity: O(N*C), C is 1/treshold
        """
        
        # transpose the matrix of the data
        X = np.transpose(X)
        
        # create a list of neurons
        points = list()
        
        # iterates through data
        for items in X:
            # initiate a data of a variable
            xi = items
            
            # create an array of points for x axis 
            x = np.arange(min(xi),max(xi),treshold)
            
            # create a list of value of its derivative in range of data
            y = [deriv(i, treshold, xi) for i in x]
            
            # build a list of local maximum from the derivative value
            local_max = list()
            for i in range(len(y)-1):
                if (y[i] > 0 and y[i+1] < 0) or (y[i] < 0 and y[i+1] > 0):
                    local_max.append(i*0.001+min(xi))
            points.append(local_max)
        return points
    
    def create_initial_centroid_kde(self, X: np.ndarray, treshold = 0.001):
        """
        Initiate the centroid of kmeans using kernel density function peak.

        Args:
            X (np.ndarray): Matrix of input data.
            treshold (float, optional): h value for deriv(). Defaults to 0.001.

        Returns:
            np.ndarray: list of centroid for kmeans clustering.
        
        Overall Time Complexity: O(N*C), C is max(1/treshold, m*n*dim)
        """
        
        # create a list kde peak for all centroids
        c = self.find_initial_centroid(X, treshold)
        
        # fill the empty value with none and reshape the neuron size
        new_c = np.full(shape=(self.m * self.n,X.shape[1]), fill_value = None)
        
        # change the None value to a randomized float value
        for i in range(self.m * self.n):
            for j in range(X.shape[1]):
                try: 
                    new_c[i][j] = c[j][i]
                except:
                    new_c[i][j] = random.uniform(np.min(X),np.max(X))
        return new_c
    
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
            neurons = self.create_initial_centroid_kde(data) 
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
            min_index = np.argmin([euc_distance(neuron, x) for neuron in neurons]) # O(m * n * dim) 
        elif self.dist_func == "cosine":
            min_index = np.argmin([cos_distance(neuron, x) for neuron in neurons]) # O(m * n * dim) 

        return np.unravel_index(min_index, (self.m, self.n)) # O(1)
    
    def gaussian_neighbourhood(self, x1, y1, x2, y2):
        """Represents gaussian function as the hyper parameter of updating weight of neurons

        Args:
            x1 (_type_): x coordinates of best matching unit
            y1 (_type_): y coordinates of best matching unit
            x2 (_type_): x coordinates of the neuron
            y2 (_type_): y coordinates of the neuron

        Returns:
            float(): return the evaluation of h(t) = a(t) * exp(-||r_c - r_i||^2/(2 * o(t)^2))
        
        Overall Time Complexity: O(1)
        """
        lr = self.cur_learning_rate
        nr = self.cur_neighbour_rad
        if nr == 0:
            nr = 1e-9
        dist = float(euc_distance([x1, y1], [x2,y2]))
        exp = math.exp(-0.5 * ((dist/nr*dist/nr)))
        val = np.float64(lr * exp)
        return val
    
    def update_neuron(self, x:np.array):
        """Update neurons based on the input data in each iteration

        Args:
            x (np.array): the input value from data
            
        Overall Complexity: O(m * n * dim)
        """
        # find index for the best matching unit index --> O(m * n * dim)
        bmu_index = self.index_bmu(x)
        col_bmu = bmu_index[0]
        row_bmu = bmu_index[1]
        
        # iterates through the matrix --> O(m * n * dim)
        for cur_col in range(len(self.neurons)):
            for cur_row in range(len(self.neurons[0])):
                # initiate the current weight of the neurons
                cur_weight = self.neurons[cur_col][cur_row]
                
                # calculate the new weight, update only if the weight is > 0
                h = self.gaussian_neighbourhood(col_bmu, row_bmu, cur_col, cur_row)
                # !!! CHANGE USE DEGREE IF USE COSINE
                if h > 0:
                    if self.dist_func == "euclidean":
                        # new weight = cur weight + moving weight * distance
                        new_weight = cur_weight +  h * (x - cur_weight)
                    elif self.dist_func == "cosine":
                        angle = cos_distance(x,cur_weight)
                        new_weight = [i+j*h for i,j in zip([math.cos(angle)*i for i in cur_weight],cur_weight)]
                    # update the weight
                    self.neurons[cur_col][cur_row] = new_weight
    
    def fit(self, X : np.ndarray, epoch : int, shuffle=True, verbose=True):
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
        if not self._trained:
            self.neurons = self.initiate_neuron(data=X, min_val= X.min(), max_val= X.max()) # O(m * n * dim)
            # initiate new neurons
            self.initial_neurons = self.neurons
        
        if X.shape[1] != self.dim :
            raise ValueError("X.shape[1] should be the same as self.dim, but found {}".format(X.shape[1]))
        
        # initiate parameters
        global_iter_counter = 1
        n_sample = X.shape[0]
        total_iteration = min(epoch * n_sample, self.max_iter)
        
        # iterates through epoch --> O(epoch * N * m * n * dim)
        for i in range(epoch):
            if global_iter_counter > self.max_iter :
                break
            
            # shuffle the data
            if shuffle:
                np.random.shuffle(X)
            
            # iterates through data --> O(N * m * n * dim)
            for idx in X:
                if global_iter_counter > self.max_iter :
                    break
                input = idx
                
                # update the neurons --> O(m * n * dim)
                self.update_neuron(input)
                if verbose:
                    render_bar(global_iter_counter, total_iteration, "Training")
                
                # update parameter and hyperparameters --> O(1)
                global_iter_counter += 1
                power = global_iter_counter/total_iteration
                self.cur_learning_rate = self.initial_learning_rate**(1-power) * math.exp(-1 * global_iter_counter/self.initial_learning_rate)
                self.cur_neighbour_rad = self.initial_neighbour_rad**(1-power) * math.exp(-1 * global_iter_counter/self.initial_neighbour_rad)
        
        self._trained = True
        
        return 
    
    def predict(self, X: np.ndarray) :
        """Label the data based on the neurons using best matching unit

        Args:
            X (np.ndarray): input data to be predicted

        Raises:
            NotImplementedError: The model have not been trained yet, call SOM.fit() first

        Returns:
            np.array(): array of the label of each data
            
        Overall Time Complexity: O(N * m * n * dim)
        """
        if not self._trained:
            raise  NotImplementedError("SOM object should call fit() before predict()")
        
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimension {self.dim}. Rechieved input with dimension {X.shape[1]}'
        
        labels = [self.index_bmu(x) for x in X]
        labels = [(self.m*i + j) for i, j in labels]
        return labels
    
    def fit_predict(self, X : np.ndarray, epoch : int, shuffle=True, verbose=True):
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
        self.fit(X = X, epoch = epoch, shuffle=shuffle, verbose=verbose) # O(epoch * N * m * n * dim)
        return self.predict(X=X) # O(N * m * n * dim)
    
    def evaluate(self, method:str):
        if method not in DISTANCE_METHOD_LIST:
            raise ValueError("There is no method called {}".format(initiate_method))
        
        if method == "euclidean":
            dist_matrix = np.array([euc_distance(i,j) for j in self.cluster_center_ for i in self.cluster_center_])
        elif method == "cosine":
            dist_matrix = np.array([cos_distance(i,j) for j in self.cluster_center_ for i in self.cluster_center_])
        return np.sum(dist_matrix)
    
    @property
    def cluster_center_(self):
        """Generate the list of all neurons

        Returns:
            np.ndarray(): list of all neurons with shape (m*n, dim)
        """
        return np.reshape(self.neurons, (-1, self.dim))
