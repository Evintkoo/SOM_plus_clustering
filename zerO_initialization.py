import numpy as np

def zero_initialization_single_layer(P, Q):
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


def hadamard_matrix(m):
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


if __name__ == "__main__":
    # Example Usage:
    P = 3  # Input size
    Q = 4  # Output size
    W = zero_initialization_single_layer(P, Q)
    print(W)
