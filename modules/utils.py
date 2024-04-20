import numpy as np
import math
import sys

def find_most_edge_point(points):
    # Calculate the center of the dataset
    center = np.mean(points, axis=0)
    
    # Calculate the Euclidean distance between each point and the center
    distances = np.sqrt(np.sum((points - center)**2, axis=1))
    
    # Find the index of the point with the maximum distance
    most_edge_index = np.argmax(distances)
    
    return points[most_edge_index]

def cos_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("input value has different length,", len(vector1), "!=", len(vector2))
    else:
        mag_a = np.linalg.norm(vector1)
        mag_b = np.linalg.norm(vector2)
        d_cos = 1-mag_a*mag_b/(mag_a**2+mag_b**2)
        return math.acos(d_cos)

def random_initiate(dim: int, min_val: float, max_val: float) -> np.ndarray:
    """
    Initiate an array of random numbers in the range (min_val, max_val).

    Args:
        dim (int): Dimension of the array.
        min_val (float): Minimum value of the random numbers.
        max_val (float): Maximum value of the random numbers.

    Returns:
        np.ndarray: Array of randomly generated numbers.

    Time Complexity: O(1)
    Space Complexity: O(dim)
    """
    return np.random.uniform(min_val, max_val, dim)

def euc_distance(x: np.array, y: np.array) -> float:
    """Calculate the Euclidean distance between arrays x and y.

    Args:
        x (np.array): Input array 1.
        y (np.array): Input array 2.

    Returns:
        float: Euclidean distance between x and y.

    Raises:
        ValueError: If the lengths of x and y are different.

    Time Complexity: O(n), where n is the length of the input arrays.
    """
    if len(x) != len(y):
        raise ValueError("Input arrays have different lengths.")
    return np.sqrt(np.sum((x - y) ** 2))
    
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