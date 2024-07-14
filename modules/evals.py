import numpy as np

def silhouette_score(X, labels):
    """
    Calculate the Silhouette Coefficient for a clustering result.

    Parameters:
    X (array-like): The input data points, shape (n_samples, n_features).
    labels (array-like): The assigned cluster labels for each data point, shape (n_samples,).

    Returns:
    float: The Silhouette Coefficient.
    """
    n_samples = X.shape[0]
    n_clusters = len(np.unique(labels))

    # Compute the pairwise distances between data points
    distances = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1))

    # Compute the average intra-cluster distance (a) for each data point
    a = np.zeros(n_samples)
    for i in range(n_samples):
        cluster_points = X[labels == labels[i]]
        if len(cluster_points) > 1:
            a[i] = np.mean(distances[i][labels == labels[i]])
        else:
            a[i] = 0  # if the cluster has only one point, set a[i] to 0

    # Compute the average nearest-cluster distance (b) for each data point
    b = np.zeros(n_samples)
    for i in range(n_samples):
        other_cluster_distances = []
        for j in range(n_clusters):
            if j != labels[i] and np.any(labels == j):
                other_cluster_distances.append(np.mean(distances[i][labels == j]))
        if other_cluster_distances:
            b[i] = min(other_cluster_distances)
        else:
            b[i] = 0  # if no other cluster exists, set b[i] to 0

    # Compute the Silhouette Coefficient for each data point
    silhouette_coefficients = np.zeros(n_samples)
    for i in range(n_samples):
        if a[i] == 0 and b[i] == 0:
            silhouette_coefficients[i] = 0
        else:
            silhouette_coefficients[i] = (b[i] - a[i]) / max(a[i], b[i])

    # Compute the overall Silhouette Coefficient
    silhouette_score = np.mean(silhouette_coefficients)

    return silhouette_score

def davies_bouldin_index(X, labels):
    """
    Calculate the Davies-Bouldin Index for a clustering result.

    Args:
        X (numpy.ndarray): The input data points as a 2D array.
        labels (numpy.ndarray): The cluster labels for each data point as a 1D array.

    Returns:
        float: The Davies-Bouldin Index.
    """
    n_clusters = len(np.unique(labels))
    n_samples, n_features = X.shape

    centroids = np.zeros((n_clusters, n_features))
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        centroids[i] = np.mean(cluster_points, axis=0)

    distances = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))

    cluster_dispersions = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        cluster_dispersions[i] = np.mean(distances[labels == i, i])

    db_index = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                numerator = cluster_dispersions[i] + cluster_dispersions[j]
                denominator = np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2))
                ratio = numerator / denominator
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio

    db_index /= n_clusters
    return db_index

def calinski_harabasz_score(X, labels):
    """
    Calculates the Calinski-Harabasz Index for a clustering result.

    Args:
        X (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
        labels (numpy.ndarray): The cluster labels for each data point of shape (n_samples,).

    Returns:
        float: The Calinski-Harabasz Index.
    """
    n_samples, n_features = X.shape
    n_clusters = len(np.unique(labels))

    # Compute the overall centroid
    overall_centroid = np.mean(X, axis=0)

    # Compute the between-cluster dispersion
    between_dispersion = 0
    for cluster_label in np.unique(labels):
        cluster_mask = labels == cluster_label
        cluster_samples = X[cluster_mask]
        cluster_centroid = np.mean(cluster_samples, axis=0)
        cluster_size = cluster_samples.shape[0]
        between_dispersion += cluster_size * np.sum((cluster_centroid - overall_centroid) ** 2)

    # Compute the within-cluster dispersion
    within_dispersion = 0
    for cluster_label in np.unique(labels):
        cluster_mask = labels == cluster_label
        cluster_samples = X[cluster_mask]
        cluster_centroid = np.mean(cluster_samples, axis=0)
        within_dispersion += np.sum((cluster_samples - cluster_centroid) ** 2)

    # Compute the Calinski-Harabasz Index
    ch_index = (between_dispersion / (n_clusters - 1)) / (within_dispersion / (n_samples - n_clusters))

    return ch_index

def dunn_index(X, labels):
    """
    Calculate the Dunn Index for a clustering result.

    Parameters:
    X (array-like): The input data points as a 2D array.
    labels (array-like): The cluster labels for each data point.

    Returns:
    float: The Dunn Index value.
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    # Calculate the pairwise distances between data points
    distances = np.sqrt(np.sum(np.square(X[:, np.newaxis] - X), axis=2))

    # Initialize variables
    min_inter_cluster_distance = np.inf
    max_intra_cluster_distance = 0

    # Iterate over all pairs of clusters
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            # Find the minimum distance between points in cluster i and cluster j
            cluster_i_points = X[labels == unique_labels[i]]
            cluster_j_points = X[labels == unique_labels[j]]
            inter_cluster_distances = distances[labels == unique_labels[i]][:, labels == unique_labels[j]]
            min_distance = np.min(inter_cluster_distances)
            min_inter_cluster_distance = min(min_inter_cluster_distance, min_distance)

    # Iterate over all clusters
    for i in range(num_clusters):
        # Find the maximum distance between points within cluster i
        cluster_points = X[labels == unique_labels[i]]
        intra_cluster_distances = distances[labels == unique_labels[i]][:, labels == unique_labels[i]]
        max_distance = np.max(intra_cluster_distances)
        max_intra_cluster_distance = max(max_intra_cluster_distance, max_distance)

    # Calculate the Dunn Index
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn_index

def compare_distribution(data1: np.array, data2:np.array, num_bins: int= 100):
    data1 = np.transpose(data1)
    data2 = np.transpose(data2)
    mean_acc = []
    for i, j in zip(data1, data2):
        hist1, bin_edges = np.histogram(i, bins=num_bins)
        hist2, bin_edges = np.histogram(j, bins=num_bins)
        squared_diff = ((hist1/len(i) - hist2/len(j)) ** 2)**0.5
        mean_acc.append(np.mean(squared_diff))
    return np.mean(mean_acc)

def bcubed_precision(clusters, labels):
    """
    Menghitung BCubed Precision berdasarkan kluster dan label.
    
    Args:
    clusters (list): Daftar kluster yang dihasilkan oleh algoritma clustering.
    labels (list): Daftar label asli (ground truth) dari data.
    
    Returns:
    float: Nilai BCubed Precision.
    """
    n = len(labels)
    assert len(clusters) == n, "Panjang kluster dan label harus sama"
    
    precision_sum = 0.0
    
    for i in range(n):
        cluster_i = clusters[i]
        label_i = labels[i]
        
        # Item-item dalam kluster yang sama dengan item i
        same_cluster = [j for j in range(n) if clusters[j] == cluster_i]
        
        # Item-item dalam kluster yang memiliki label yang sama dengan item i
        same_label_in_cluster = [j for j in same_cluster if labels[j] == label_i]
        
        # Precision untuk item i
        precision_i = len(same_label_in_cluster) / len(same_cluster)
        precision_sum += precision_i
    
    # BCubed Precision
    bcubed_precision = precision_sum / n
    return bcubed_precision

def bcubed_recall(clusters, labels):
    """
    Menghitung BCubed Recall berdasarkan kluster dan label.
    
    Args:
    clusters (list): Daftar kluster yang dihasilkan oleh algoritma clustering.
    labels (list): Daftar label asli (ground truth) dari data.
    
    Returns:
    float: Nilai BCubed Recall.
    """
    n = len(labels)
    assert len(clusters) == n, "Panjang kluster dan label harus sama"
    
    recall_sum = 0.0
    
    for i in range(n):
        cluster_i = clusters[i]
        label_i = labels[i]
        
        # Item-item yang memiliki label yang sama dengan item i
        same_label = [j for j in range(n) if labels[j] == label_i]
        
        # Item-item dalam kluster yang memiliki label yang sama dengan item i
        same_label_in_cluster = [j for j in same_label if clusters[j] == cluster_i]
        
        # Recall untuk item i
        recall_i = len(same_label_in_cluster) / len(same_label)
        recall_sum += recall_i
    
    # BCubed Recall
    bcubed_recall = recall_sum / n
    return bcubed_recall
