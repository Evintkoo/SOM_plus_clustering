import numpy as np

def naive_sharding_initialization(X, k):
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