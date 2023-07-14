"""
    References :
    Chaudhary, V., Bhatia, R. S., & Ahlawat, A. K. (2014). A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons. Alexandria Engineering Journal, 53(4), 827-831. https://doi.org/10.1016/j.aej.2014.09.007
"""

import numpy as np
import math 
import random
from SOM_plus_clustering.utils import random_initiate, euc_distance, gauss, std_dev, kernel_gauss, deriv

class kmeans():
    """
    Kmeans class is consist of:
        kmeans.n_clusters(int): Number of centroids.
        kmeans.centroids(np.ndarray): Vector value of centroids with size of kmeans.n_clusters.
        kmeans._trained(bool): If the kmeans.fit() have called, returns true, false otherwise
        kmeans.method(str): Kmeans centroid initiation method.
    """
    def __init__(self, n_clusters: int, method:str):
        """
        Intiate the main parameter of kmeans clustering.

        Args:
            n_clusters (int): Number of centroids for kmeans.
            method (str): Initiation method for kmeans.
            
            
        Overall Time Complexity: O(1)
        """
        self.n_clusters = n_clusters
        self.centroids = None
        self._trained = False
        self.method = method
    
    def find_initial_centroid(self, X : np.ndarray, treshold:float) -> np.ndarray:
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
        X = np.transpose(X) # O(1)
        
        # create a list of centroids
        points = list() # O(1)
        
        # iterates through the data -> O(N*C)
        for items in X:
            # initiate a data of a variable
            xi = items
            
            # create an array of points for x axis 
            x = np.arange(min(xi),max(xi),treshold) # O(C)
            
            # create a list of value of its derivative in range of data
            y = [deriv(i, treshold, xi) for i in x] # O(C)
            
            # build a list of local maximum from the derivative value
            local_max = list()
            for i in range(len(y)-1): # O(C)
                if (y[i] > 0 and y[i+1] < 0) or (y[i] < 0 and y[i+1] > 0):
                    local_max.append(i*0.001+min(xi))
                    
            # append the list of local max of the variable
            points.append(local_max)
        return points

    def create_initial_centroid_kde(self, X: np.ndarray, treshold = 0.001) -> np.ndarray:
        """
        Initiate the centroid of kmeans using kernel density function peak.

        Args:
            X (np.ndarray): Matrix of input data.
            treshold (float, optional): h value for deriv(). Defaults to 0.001.

        Returns:
            np.ndarray: list of centroid for kmeans clustering.
            
        Overall Time Complexity: O(N*C), C is max(1/treshold, dim)
        """
        # create a list kde peak for all centroids
        c = self.find_initial_centroid(X, treshold) # O(N*C)
        
        # fill the empty value with none
        new_c = np.full(shape=(self.n_clusters,X.shape[1]), fill_value = None) # O(N*dim)
        
        # change the None value to a randomized float value 
        for i in range(self.n_clusters): # O(k*dim)
            for j in range(X.shape[1]): # O(dim)
                try: 
                    new_c[i][j] = c[j][i]
                except:
                    new_c[i][j] = random.uniform(np.min(X),np.max(X))
        return new_c
    
    def initiate_plus_plus(self, X : np.ndarray) -> np.ndarray:
        """
        Initiate the centroid value using kmeans++ algorithm

        Args:
            X (np.ndarray): Matrix of input data.

        Returns:
            np.ndarray: list of centroid for kmeans clustering.
            
        Overall Time Complexity: O(N*k*k*dim), k is number of cluster
        """
        # initiate empty list of centroids
        centroids = list() # O(1)
        
        # choose a random list of data
        centroids.append(random.choice(X)) # O(1)
        
        # initiate the number of k in kmeans
        k = self.n_clusters # O(1)
        
        # iterate k-1 number to fill the list of centroids
        for c in range(k-1): # O(k)
            # find the minimum euclidean distance square from the all centroids in each data point
            dist_arr = [min([euc_distance(j, i)*euc_distance(j, i) for j in centroids]) for i in X] # O(N*K*dim)
            
            # find the furthest vector of data 
            furthest_data = X[np.argmax(dist_arr)] # O(N)
            
            # append the data to the list of centroid
            centroids.append(furthest_data)
        return centroids
    
    def init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Initiate the centroids of kmeans clustering.

        Args:
            X (np.ndarray): Matrix of input data.

        Raises:
            ValueError: If there is no method name self.method.

        Returns:
            np.ndarray: list of centroids for kmeans clustering (ready to train).
        
        Overall Time Complexity:
            random: O(k*dim)
            kde: O(N*C*dim)
            kmeans++: O(N*k*dim)
        """
        
        if self.method == "random":
            # create a random value of data with length of self.n_clusters
            centroids = [random_initiate(dim=X.shape[1], min_val=X.min(), max_val=X.max()) for i in range(self.n_clusters)]
            self.centroids = centroids
        elif self.method == "kde": 
            # initiate centroids based on kernel density function peak 
            centroids = self.create_initial_centroid_kde(X)
            self.centroids = centroids
        elif self.method == "kmeans++" :
            # initiate the centroids using a kmean++ algorithm
            self.centroids = self.initiate_plus_plus(X)
        else:
            # raise an error if there is no such a method.
            raise ValueError("There is no method named {}".format())
        return 
    
    def update_centroids(self, x:np.array):
        """
        update the centroid value

        Args:
            x (np.array): input data
        
        Overall Time Complexity: O(k)
        """
        # new_centroids = np.array([X[self.cluster_labels == i].mean(axis=0) for i in range(self.k)])
        new_centroids = list() # O(1)
        
        # find the distance of centroids for each data
        centroids_distance = [euc_distance(x, i) for i in self.centroids] # O(k)
        
        # find the closest centroid in self.centroids
        closest_centroids_index = np.argmin(centroids_distance) # O(k)
        closest_centroids = self.centroids[closest_centroids_index] # O(1)
        
        # update the closest centroids to the data
        closest_centroids = [(i+j)/2 for i,j in zip(x,closest_centroids)] # O(k)
        
        # update the centroid in model
        self.centroids[closest_centroids_index] = closest_centroids # O(1)
        
    
    def fit(self, X: np.ndarray, epochs=3000, shuffle=True):
        """
        Train the kmeans model to find the best value of the centroid.

        Args:
            X (np.ndarray): Matrix of input data.
            epochs (int, optional): Number of training iteration. Defaults to 3000.
            shuffle (bool, optional): Shuffle the data. Defaults to True.

        Raises:
            SyntaxError: Only could train the model if kmeans._trained is false (have not been trained)
        
        Overall Time Complexity: O(N*C), C max (1/treshold, k*epoch)
        """
        if self._trained:
            raise SyntaxError("Cannot fit the model that have been trained")
        
        # initiate the centroid of kmeans model
        self.init_centroids(X) # O(N*C) assuming k << C
        
        # iterates several times for trains the data
        for epoch in range(epochs): # O(epoch)
            
            # shuffle the data
            if shuffle:
                np.random.shuffle(X) # O(N)
            
            # iterates through data to update centroids
            for x in X: # O(N*k)
                self.update_centroids(x) # O(k)
    
    def predict(self, X : np.ndarray) -> np.array:
        """
        Predict the cluster number from the matrix of data.

        Args:
            X (np.ndarray): Matrix of input data.

        Returns:
            np.array: list of cluster label.
            
        Overall Time Complplexity: O(N*k)
        """
        return [np.argmin([euc_distance(x, centers) for centers in self.centroids]) for x in X]

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
        SOM.method (str): neurons initiation method 
        self.cur_neighbor_rad (float): current neighbor radius of SOM (g(t))
        self.initial_neighbor_rad (float): initial neighbourhood radius of SOM (g(o))
        SOM.neurons (np.ndarray): value of neurons in the matrix, none if SOM.fit() have not called yet
        SOM.initial_neurons (np.ndarray): initial value of the neurons, none if SOM.fit() have not called yet
    """
    def __init__(self, m: int, n: int, dim: int, initiate_method:str, max_iter: int, learning_rate:float, neighbour_rad: int) -> None:
        """
        Initiate the main parameter of Self Organizing Matrix Clustering

        Args:
            m (int): _description_
            n (int): _description_
            dim (int): _description_
            method (str): _description_
            max_iter (int): _description_
            learning_rate (float): _description_
            neighbour_rad (int): _description_

        Raises:
            ValueError: _description_
        
        Overall Time Complexity: O(1)
        """
        if learning_rate > 1.76:
            raise ValueError("Learning rate should be less than 1.76")
        method_type = ["random", "kde", "kmeans", "kde_kmeans", "kmeans++", "SOM++"]
        if initiate_method not in method_type:
            raise ValueError("There is no method called {}".format(initiate_method))
        
        # initiate all the attributes
        self.m = m
        self.n = n
        self.dim = dim
        self.max_iter = max_iter
        self.shape = (m,n,dim)
        self.cur_learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self._trained = False
        self.method = initiate_method
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
            ValueError: There are no method named self.method or the method is not available yet

        Returns:
            list(): list of neurons to be initiate in self.neurons and self.initial_neurons
            
        Overall Time Complexity:
            self.method == "random": O(C), C is m*n*dim
            self.method == "kde": O(N*C), C is max(1/treshold, m*n*dim)
            self.method == "kmeans": O(N*C), C is max(1/treshold, m*n*dim)
            self.method == "kde_kmeans": O(N*C), C is max(1/treshold, m*n*dim)
            self.method == "kmeans++": O(N*C), C is k*k*dim
            self.method == "SOM++": O(N*C), C is k*k*dim
        """
        if self.method == "random" :
            # number of step = self.dim * self.m * self.n --> O(m * n * dim)
            return [[random_initiate(self.dim ,min_val=min_val, max_val=max_val) for j in range(self.m)] for i in range(self.n)]
        elif self.method == "kde" :
            neurons = self.create_initial_centroid_kde(data) 
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        elif self.method == "kmeans":
            model = kmeans(n_clusters = (self.m * self.n), method="random")
            model.fit(X = data)
            neurons = model.centroids
            neurons = np.sort(neurons, axis=0)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        elif self.method == "kde_kmeans":
            model = kmeans(n_clusters = (self.m * self.n), method="kde")
            model.fit(X = data)
            neurons = model.centroids
            neurons = np.sort(neurons, axis=0)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        elif self.method == "kmeans++" :
            model = kmeans(n_clusters = (self.m * self.n), method="kmeans++")
            model.fit(X = data)
            neurons = model.centroids
            neurons = np.sort(neurons, axis=0)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        elif self.method == "SOM++" :
            neurons = self.initiate_plus_plus(data)
            neurons = np.reshape(neurons, (self.m, self.n, self.dim))
            return neurons
        else:
            raise ValueError("There is no method named {}".format(self.method))
    
    def index_bmu(self, x: np.array):
        """Find the index of best matching unit among all of neurons inside the matrix based on its euclidean distance

        Args:
            x (np.array): input array as comparison parameter

        Returns:
            tuple(): set of coordinates the best matching unit in (x,y) 
        
        Overall Time Complexity: O(m * n * dim)
        """
        neurons = np.reshape(self.neurons, (-1, self.dim)) # O(1)
        min_index = np.argmin([euc_distance(neuron, x) for neuron in neurons]) # O(m * n * dim) 
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
                if h > 0:
                    new_weight = cur_weight +  h * (x - cur_weight)
                    # update the weight
                    self.neurons[cur_col][cur_row] = new_weight
    
    def fit(self, X : np.ndarray, epoch : int, shuffle=True):
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
        if self._trained:
            raise SyntaxError("Cannot fit the model that have been trained")
        
        if X.shape[1] != self.dim :
            raise ValueError("X.shape[1] should be the same as self.dim, but found {}".format(X.shape[1]))
        
        # initiate new neurons
        self.neurons = self.initiate_neuron(data=X, min_val= X.min(), max_val= X.max()) # O(m * n * dim)
        self.initial_neurons = self.neurons
        
        # initiate parameters
        global_iter_counter = 0
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
    
    def fit_predict(self, X : np.ndarray, epoch : int, shuffle=True):
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
        self.fit(X = X, epoch = epoch, shuffle=shuffle) # O(epoch * N * m * n * dim)
        return self.predict() # O(N * m * n * dim)
    
    @property
    def cluster_center_(self):
        """Generate the list of all neurons

        Returns:
            np.ndarray(): list of all neurons with shape (m*n, dim)
        """
        return np.reshape(self.neurons, (-1, self.dim))
