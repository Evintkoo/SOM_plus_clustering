"""_summary_
    References :
    Chaudhary, V., Bhatia, R. S., & Ahlawat, A. K. (2014). A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons. Alexandria Engineering Journal, 53(4), 827-831. https://doi.org/10.1016/j.aej.2014.09.007
"""

import numpy as np
import pandas as pd
import math 
import random

# Initiate random number for grid in matrix with dimension of x
def random_initiate(dim: int, min_val:float, max_val:float):
    x = [random.uniform(0,100) for i in range(dim)]
    return x

# Euclidean distance function
def euc_distance(x: np.array, y: np.array):
    if len(x) != len(y):
        raise ValueError("input value has different length")
    else :
        dist = sum([(i2-i1)**2 for i1, i2 in zip(x, y)])**0.5
        return dist

# Self Organizing Matrix Class
class SOM(): 
    def __init__(self, m: int, n: int, dim: int, method:str, max_iter: int, learning_rate:float, neighbour_rad: int) -> None:
        if learning_rate > 1:
            raise ValueError("Learning rate should be less than 1")
        self.m = m
        self.n = n
        self.dim = dim
        self.max_iter = max_iter
        self.shape = (m,n)
        self.cur_learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self._trained = False
        self.method = method
        self.cur_neighbour_rad = neighbour_rad
        self.initial_neighbour_rad = neighbour_rad
        self.neurons = None # self.initiate_neuron()
    
    def initiate_neuron(self, min_val:float, max_val:float):
        if self.method == "random" :
            return [[random_initiate(self.dim ,min_val=min_val, max_val=max_val) for j in range(self.m)] for i in range(self.n)]
        elif self.method == "kmeans":
            raise ValueError("Sorry, {} have not available yet".format(self.method))
            return #!!!
        else:
            raise ValueError("There is no method named {}".format(self.method))
        
    def find_index(self, x:int):
        return (math.floor(x/self.m), x%self.n)
    
    
    def index_bmu(self, x: np.array):
        neurons = np.reshape(self.neurons, (-1, self.dim))
        min_index = np.argmin([euc_distance(neuron, x) for neuron in neurons])
        return self.find_index(min_index)
    
    def gaussian_neighbourhood(self, x1, y1, x2, y2):
        return self.cur_learning_rate * math.exp(-1 * (euc_distance([x1, y1], [x2, y2])**2 / (2 * self.cur_neighbour_rad**2)))
    
    def update_neuron(self, x:np.array):
        """
        Update the neurons in the Self Organizing Matrix
        """
        # find index for the best matching unit
        bmu_index = self.index_bmu(x)
        col_bmu = bmu_index[0]
        row_bmu = bmu_index[1]
        
        # # initialize current weight of neurons
        # cur_weight = self.neurons[bmu_index[0]][bmu_index[1]]
        # 
        # # update the neurons
        # new_weight = cur_weight + self.gaussian_neighbourhood() * (x - cur_weight)
        # self.neurons[bmu_index[0]][bmu_index[1]] = new_weight
        new_neurons = list()
        for cur_col in range(0, self.n):
            for cur_row in range(0, self.m):
                # initiate the current weight of the neurons
                cur_weight = self.neurons[cur_col][cur_row]
                
                # calculate the new weight
                new_weight = cur_weight + self.gaussian_neighbourhood(col_bmu, row_bmu, cur_col, cur_row) * (x - cur_weight)
                
                # update the weight
                self.neurons[cur_col][cur_row] = new_weight
    
    def fit(self, X : np.ndarray, epoch : int, shuffle=True):
        if self._trained:
            raise SyntaxError("Cannot fit the model that have been trained")
        self.neurons = self.initiate_neuron(min_val= X.min(), max_val= X.max())
        
        global_iter_counter = 0
        n_sample = X.shape[0]
        total_iteration = min(epoch * n_sample, self.max_iter)
        for i in range(epoch):
            if global_iter_counter > self.max_iter :
                break
            if shuffle:
                x = random.choice(X)
            else :
                x = X[i]
            
            for idx in X:
                if global_iter_counter > self.max_iter :
                    break
                input = idx
                self.update_neuron(input)
                global_iter_counter += 1
                power = global_iter_counter/total_iteration
                self.cur_learning_rate = self.initial_learning_rate**(1-power) * math.exp(-1 * global_iter_counter/self.initial_learning_rate)
                self.cur_neighbour_rad = self.initial_neighbour_rad**(1-power) * math.exp(-1 * global_iter_counter/self.initial_neighbour_rad)
        
        self._trained = True
        
        return 
    
    def predict(self, X) :
        if not self._trained:
            raise  NotImplementedError("SOM object should call fit() before predict()")
        
        assert len(X.shape) == 2, f'X should have two dimensions, not {len(X.shape)}'
        assert X.shape[1] == self.dim, f'This SOM has dimension {self.dim}. Rechieved input with dimension {X.shape[1]}'
        
        labels = [self.index_bmu(x) for x in X]
        labels = [(self.m*i + j) for i, j in labels]
        return labels
    
    def fit_predict(self):
        self.fit()
        return self.predict()
    
    @property
    def cluster_center_(self):
        return np.reshape(self.neurons, (-1, self.dim))
# if __name__ == "__main__":
#     x = [0, 0]
#     y = [3, 4, 5]
#     print(euc_distance(x,y))