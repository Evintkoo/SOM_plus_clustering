import numpy as np
import math 
import random
import sys

def cos_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("input value has different length")
    else:
        mag_a = np.linalg.norm(vector1)
        mag_b = np.linalg.norm(vector2)
        d_cos = 1-mag_a*mag_b/(mag_a**2+mag_b**2)
        return math.acos(d_cos)

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

def euc_distance(x: np.array, y: np.array) -> float:
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
    
def gauss(x) -> float:
    """
    Return the function of gaussian distribution 

    Args:
        x (float): input value of gaussian function.

    Returns:
        float: the result of the gaussian function value
        
    Overall Complexity: O(1)
    """
    return math.exp(-0.5 * x * x)/math.sqrt(2*math.pi)

def std_dev(x: np.array) -> float:
    """
    Finding the standar deviation for a list of value 

    Args:
        x (np.array): list of value

    Returns:
        float: standard deviation value of the input list
    
    Overall Complexity: O(N)
    """
    
    # find the average of the data
    mean = np.mean(x) # O(N)
    
    # sums the distance of all data to the average
    sums = sum( [(i - mean)**2 for i in x])**0.5 # O(N)
    
    # return the average of the data
    return sums/len(x)

def kernel_gauss(x: float, xi : np.array):
    """_summary_

    Args:
        x (float): the x value of the kernel function
        xi (list): list of values

    Returns:
        float: the value of kernel density function
        
    Overall Time Complexity: O(N)
    """
    # silvermans bandwidth estimator
    iqr = (np.percentile(xi, 75) - np.percentile(xi, 25))/1.34 # O(1)
    h = iqr * (len(xi)**(-.2)) # O(1)
    
    # returns the value of the kernel density function
    return sum([gauss((x-i)/h) for i in xi]) / (len(xi)*h) # O(N)

def deriv(x: float, h: float, xi: np.array) -> float:
    """
    Estimate the derivative of kernel density function at x

    Args:
        x (float): input value of the the derivative
        h (float): h value of the derivative
        xi (np.array): list of value in an variable

    Returns:
        float: the estimation of the derivative of the kernel function
    
    Overall Time Complexity: O(N)
    """
    f_x = kernel_gauss(x, xi) # O(N)
    f_xh = kernel_gauss(x+h, xi) # O(N)
    return (f_xh-f_x)/h 

def render_bar(value, maxs, label):
    n_bar = 40 #size of progress bar
    j= value/maxs
    sys.stdout.write('\r')
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1-j))

    sys.stdout.write(f"{label.ljust(10)} | [{bar:{n_bar}s}] {int(100 * j)}% ")
    sys.stdout.flush()