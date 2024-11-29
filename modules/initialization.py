import numpy as np

def initiate_zero(P: int, Q:int):
    """
    ZerO Initialization for a single layer.

    Args:
        P: Input dimension of the layer
        Q: Output dimension of the layer

    Returns:
        Initialized weight matrix W
    """
    if P == Q:  # Identity mapping
        W = np.eye(P)
    elif P < Q:  # Propagate first P dimensions
        W = np.zeros((P, Q))
        np.fill_diagonal(W[:P, :P], 1)  # Partial identity matrix
    else:  # Apply Hadamard matrix
        m = int(np.ceil(np.log2(P)))
        c = 2 ** (-(m - 1) / 2)
        H_m = hadamard_matrix(m)
        H_m = H_m[:P, :P]  # Truncate Hadamard matrix to P x P
        I_star = np.zeros((P, Q))  # Partial identity
        np.fill_diagonal(I_star[:P, :P], 1)
        W = c * np.dot(H_m, I_star)

    return W

def hadamard_matrix(m: int):
    """
    Generates a Hadamard matrix of size 2^m.

    Args:
        m: Power of 2 for the size of the Hadamard matrix

    Returns:
        Hadamard matrix H of size 2^m x 2^m
    """
    if m == 0:
        return np.array([[1]])
    H_prev = hadamard_matrix(m - 1)
    return np.block([
        [H_prev, H_prev],
        [H_prev, -H_prev]
    ])

def initiate_naive_sharding(X: np.array, k: int):
    """
    Initialize centroids using the naive sharding method.

    Parameters:
    X (numpy.ndarray): The dataset, where each row is a data point.
    k (int): The number of clusters.

    Returns:
    numpy.ndarray: Initialized centroids.
    """
    # Step 1: Compute the sum of each data point's features
    composite_values = np.sum(X, axis=1)

    # Step 2: Sort data points by their composite values
    sorted_indices = np.argsort(composite_values)
    sorted_X = X[sorted_indices]

    # Step 3: Split the sorted data into k equal-sized shards
    shards = np.array_split(sorted_X, k)

    # Step 4: Compute the mean of each shard to determine centroids
    centroids = np.array([np.mean(shard, axis=0) for shard in shards])

    return centroids

# Function for He initialization
def initiate_he(input_dim, output_dim):
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

def initiate_lecun(input_shape, output_shape):
    fan_in = input_shape  # Number of input units
    stddev = np.sqrt(1.0 / fan_in)
    return np.random.normal(loc=0.0, scale=stddev, size=(input_shape, output_shape))