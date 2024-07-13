import numpy as np
import math 
import random
from .utils import random_initiate, euc_distance

class KMeans():
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
        elif self.method == "kmeans++":
            self.centroids = self.initiate_plus_plus(X=X)
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