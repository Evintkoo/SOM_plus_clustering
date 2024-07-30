import numpy as np
from scipy.spatial.distance import pdist, squareform

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Silhouette Coefficient for a clustering result.

    Args:
        X (np.ndarray): The input data points, shape (n_samples, n_features).
        labels (np.ndarray): The assigned cluster labels for each data point, shape (n_samples,).

    Returns:
        float: The Silhouette Coefficient. Returns 0.0 if there's only one cluster or if all clusters have only one sample.
    """
    distances = squareform(pdist(X))
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        return 0.0  # Return 0 if there's only one cluster

    a = np.zeros(n_samples)
    b = np.full(n_samples, np.inf)

    for label in unique_labels:
        mask = labels == label
        cluster_size = np.sum(mask)
        
        if cluster_size > 1:
            a[mask] = np.sum(distances[mask][:, mask], axis=1) / (cluster_size - 1)
        else:
            a[mask] = 0  # Set a to 0 for clusters with only one sample
        
        other_distances = [np.mean(distances[mask][:, labels == other_label]) 
                           for other_label in unique_labels if other_label != label]
        if other_distances:
            b[mask] = np.min(other_distances)

    s = np.zeros(n_samples)
    valid_samples = (a != 0) | (b != np.inf)
    s[valid_samples] = (b[valid_samples] - a[valid_samples]) / np.maximum(a[valid_samples], b[valid_samples])

    if np.sum(valid_samples) == 0:
        return 0.0  # Return 0 if all clusters have only one sample

    return float(np.mean(s[valid_samples]))

def davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Davies-Bouldin Index for a clustering result.

    Args:
        X (np.ndarray): The input data points, shape (n_samples, n_features).
        labels (np.ndarray): The cluster labels for each data point, shape (n_samples,).

    Returns:
        float: The Davies-Bouldin Index.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    centroids = np.array([np.mean(X[labels == i], axis=0) for i in unique_labels])
    
    dispersions = np.zeros(n_clusters)
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        dispersions[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))
    
    centroid_distances = pdist(centroids)
    
    db_index = 0
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(i + 1, n_clusters):
            ratio = (dispersions[i] + dispersions[j]) / centroid_distances[n_clusters * i + j - ((i + 2) * (i + 1)) // 2]
            max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    
    return db_index / n_clusters

def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Calinski-Harabasz Index for a clustering result.

    Args:
        X (np.ndarray): The input data matrix, shape (n_samples, n_features).
        labels (np.ndarray): The cluster labels for each data point, shape (n_samples,).

    Returns:
        float: The Calinski-Harabasz Index.
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    overall_centroid = np.mean(X, axis=0)
    
    between_dispersion = 0
    within_dispersion = 0
    
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_size = cluster_points.shape[0]
        cluster_centroid = np.mean(cluster_points, axis=0)
        
        between_dispersion += cluster_size * np.sum((cluster_centroid - overall_centroid) ** 2)
        within_dispersion += np.sum((cluster_points - cluster_centroid) ** 2)

    return ((n_samples - n_clusters) / (n_clusters - 1)) * (between_dispersion / within_dispersion)

def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Dunn Index for a clustering result.

    Args:
        X (np.ndarray): The input data points, shape (n_samples, n_features).
        labels (np.ndarray): The cluster labels for each data point, shape (n_samples,).

    Returns:
        float: The Dunn Index value.
    """
    distances = squareform(pdist(X))
    unique_labels = np.unique(labels)

    min_inter_cluster_distance = np.inf
    max_intra_cluster_distance = 0

    for i, label_i in enumerate(unique_labels):
        mask_i = labels == label_i
        intra_distances = distances[mask_i][:, mask_i]
        max_intra_cluster_distance = max(max_intra_cluster_distance, np.max(intra_distances))

        for label_j in unique_labels[i+1:]:
            mask_j = labels == label_j
            inter_distances = distances[mask_i][:, mask_j]
            min_inter_cluster_distance = min(min_inter_cluster_distance, np.min(inter_distances))

    return min_inter_cluster_distance / max_intra_cluster_distance

def compare_distribution(data1: np.ndarray, data2: np.ndarray, num_bins: int = 100) -> float:
    """
    Compare the distribution of two datasets.

    Args:
        data1 (np.ndarray): First dataset, shape (n_features, n_samples1).
        data2 (np.ndarray): Second dataset, shape (n_features, n_samples2).
        num_bins (int): Number of bins for histogram calculation.

    Returns:
        float: Mean of the average squared differences between normalized histograms.
    """
    mean_acc = []
    for i, j in zip(data1, data2):
        hist1, _ = np.histogram(i, bins=num_bins, density=True)
        hist2, _ = np.histogram(j, bins=num_bins, density=True)
        squared_diff = np.mean(np.abs(hist1 - hist2))
        mean_acc.append(squared_diff)
    return np.mean(mean_acc)

def bcubed_precision_recall(clusters: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Calculate BCubed Precision and Recall for clustering results.

    Args:
        clusters (np.ndarray): Cluster assignments, shape (n_samples,).
        labels (np.ndarray): Ground truth labels, shape (n_samples,).

    Returns:
        tuple: (BCubed Precision, BCubed Recall)
    """
    n = len(labels)
    assert len(clusters) == n, "Length of clusters and labels must be the same"

    precision_sum = 0.0
    recall_sum = 0.0

    for i in range(n):
        cluster_i = clusters[i]
        label_i = labels[i]

        same_cluster = clusters == cluster_i
        same_label = labels == label_i

        precision_i = np.mean(same_label[same_cluster])
        recall_i = np.mean(same_cluster[same_label])

        precision_sum += precision_i
        recall_sum += recall_i

    bcubed_precision = precision_sum / n
    bcubed_recall = recall_sum / n

    return bcubed_precision, bcubed_recall