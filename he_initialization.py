import numpy as np

# Function for He initialization
def he_initialization(input_dim, output_dim):
    """
    Initializes weights using He initialization.

    Parameters:
    - input_dim (int): Number of input units (number of neurons in the previous layer).
    - output_dim (int): Number of output units (number of neurons in the current layer).

    Returns:
    - weights (numpy.ndarray): Initialized weights of shape (output_dim, input_dim).
    """
    stddev = np.sqrt(2. / input_dim)  # He initialization standard deviation
    weights = np.random.randn(output_dim, input_dim) * stddev  # Random weights
    return weights