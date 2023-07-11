import numpy as np
import math 
import random

def random_initiate(dim: int, min_val:float, max_val:float):
    """Initiate random number of value in range (min_val, max_val)

    Args:
        dim (int): dimension of the data
        min_val (float): minimum value of data
        max_val (float): maximum value of data

    Returns:
        np.array: array of randomly generated number
        
    Overall Complexity: O(dim)
    """
    x = [random.uniform(min_val,max_val) for i in range(dim)]
    return x

def euc_distance(x: np.array, y: np.array):
    """Calculate the euclidean distance of array x and y

    Args:
        x (np.array): array 1 input
        y (np.array): array 2 input

    Raises:
        ValueError: length of x and y is different

    Returns:
        float(): euclidean distance of x and y
    
    Overall Time Complexity: O(dim)
    """
    if len(x) != len(y):
        raise ValueError("input value has different length")
    else :
        dist = sum([(i2-i1)**2 for i1, i2 in zip(x, y)])**0.5
        return dist
    
def gauss(x):
    return math.exp(-0.5 * x * x)/math.sqrt(2*math.pi)

def std_dev(x):
    mean = np.mean(x)
    sums = sum( [(i - mean)**2 for i in x])**0.5
    return sums/len(x)

def kernel_gauss(x, xi):
    # silvermans bandwidth estimator
    iqr = (np.percentile(xi, 75) - np.percentile(xi, 25))/1.34
    h = iqr * (len(xi)**(-.2))
    return sum([gauss((x-i)/h) for i in xi]) / (len(xi)*h)

def deriv(x, h, xi):
    f_x = kernel_gauss(x, xi)
    f_xh = kernel_gauss(x+h, xi)
    return (f_xh-f_x)/h
