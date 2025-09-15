"""
evals.py

This module provides a set of functions to evaluate clustering performance using various metrics. 
These metrics help in assessing the quality and effectiveness of different clustering algorithms 
by comparing the distance between points within clusters and across different clusters.

Functions:
    silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
        Calculates the Silhouette Coefficient for a given clustering result.
    
    davies_bouldin_index(x: np.ndarray, labels: np.ndarray) -> float:
        Computes the Davies-Bouldin Index for evaluating the clustering result.
    
    calinski_harabasz_score(x: np.ndarray, labels: np.ndarray) -> float:
        Calculates the Calinski-Harabasz Index for a clustering result.
    
    dunn_index(x: np.ndarray, labels: np.ndarray) -> float:
        Computes the Dunn Index, measuring the ratio of minimum inter-cluster distance to 
        maximum intra-cluster distance.

    compare_distribution(data1: np.ndarray, data2: np.ndarray, num_bins: int = 100) -> float:
        Compares the distribution of two datasets by calculating the mean of the 
        average squared differences between their normalized histograms.

    bcubed_precision_recall(clusters: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        Computes the BCubed Precision and Recall for clustering results, 
        comparing cluster assignments to ground truth labels.

Usage Example:
    ```python
    import numpy as np
    from evals import (
        silhouette_score, davies_bouldin_index, calinski_harabasz_score, 
        dunn_index, compare_distribution, bcubed_precision_recall
    )

    # Example input data
    x = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 0.5], [3.5, 4.5]])
    labels = np.array([0, 1, 0, 1])

    # Calculate various clustering metrics
    silhouette = silhouette_score(x, labels)
    db_index = davies_bouldin_index(x, labels)
    ch_index = calinski_harabasz_score(x, labels)
    dunn = dunn_index(x, labels)

    # Compare distributions between two datasets
    distribution_comparison = compare_distribution(x, x, num_bins=10)

    # Compute BCubed precision and recall
    precision, recall = bcubed_precision_recall(labels, labels)

    print(f"Silhouette Score: {silhouette}")
    print(f"Davies-Bouldin Index: {db_index}")
    print(f"Calinski-Harabasz Index: {ch_index}")
    print(f"Dunn Index: {dunn}")
    print(f"Distribution Comparison: {distribution_comparison}")
    print(f"BCubed Precision: {precision}, BCubed Recall: {recall}")
    ```

Dependencies:
    - numpy: Used for numerical computations and data manipulation.
    - scipy.spatial.distance: Provides functions for calculating pairwise distances
    - typing: Used for type hinting in function definitions.

Description:
    This module is useful for evaluating the effectiveness of clustering algorithms. Each function 
    calculates a specific clustering metric, helping to determine the best clustering approach
    These metrics provide insights into the compactness, separation, and distribution of clusters.

References:
    - Silhouette Score: https://en.wikipedia.org/wiki/Silhouette_(clustering)
    - Davies-Bouldin Index: https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
    - Calinski-Harabasz Index: https://en.wikipedia.org/wiki/Calinski%E2%80%93Harabasz_index
    - Dunn Index: https://en.wikipedia.org/wiki/Dunn_index
    - BCubed Precision and Recall: https://en.wikipedia.org/wiki/Cluster_analysis
"""

from typing import Tuple
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Input validation helper function
def _validate_clustering_inputs(x: np.ndarray, labels: np.ndarray) -> None:
    """
    Validate inputs for clustering evaluation metrics.
    
    Args:
        x (np.ndarray): The input data points
        labels (np.ndarray): The cluster labels
        
    Raises:
        ValueError: If inputs are invalid
        IndexError: If dimensions don't match
    """
    if x.size == 0:
        raise ValueError("Input data cannot be empty")
    
    if labels.size == 0:
        raise ValueError("Labels cannot be empty")
    
    if x.shape[0] != labels.shape[0]:
        raise IndexError("Number of data points must match number of labels")
    
    if x.shape[0] < 1:
        raise ValueError("Must have at least one data point")

# Unsupervised learning evaluation function

def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Silhouette Coefficient for a clustering result.
    Optimized for better performance and numerical stability.

    Args:
        x (np.ndarray): The input data points, shape (n_samples, n_features).
        labels (np.ndarray): The assigned cluster labels for each data point, shape (n_samples,).

    Returns:
        float: The Silhouette Coefficient. 
                Returns 0.0 if there's only one cluster or if all clusters have only one sample.
    """
    # Validate inputs
    _validate_clustering_inputs(x, labels)
    
    n_samples = x.shape[0]
    unique_labels = np.unique(labels)

    if len(unique_labels) == 1:
        return 0.0  # Return 0 if there's only one cluster

    # Pre-compute pairwise distances efficiently
    distances = squareform(pdist(x))
    
    # Pre-allocate arrays
    a = np.zeros(n_samples)
    b = np.full(n_samples, np.inf)

    # Process each cluster
    for label in unique_labels:
        mask = labels == label
        cluster_indices = np.where(mask)[0]
        cluster_size = len(cluster_indices)

        if cluster_size > 1:
            # Vectorized computation of intra-cluster distances
            cluster_distances = distances[np.ix_(cluster_indices, cluster_indices)]
            # Sum over rows and divide by (cluster_size - 1)
            a[cluster_indices] = (cluster_distances.sum(axis=1) - np.diag(cluster_distances)) / (cluster_size - 1)
        else:
            a[cluster_indices] = 0  # Set a to 0 for clusters with only one sample

        # Compute inter-cluster distances efficiently
        for other_label in unique_labels:
            if other_label != label:
                other_mask = labels == other_label
                other_indices = np.where(other_mask)[0]
                if len(other_indices) > 0:
                    # Vectorized computation of inter-cluster distances
                    inter_distances = distances[np.ix_(cluster_indices, other_indices)]
                    mean_inter_distances = inter_distances.mean(axis=1)
                    b[cluster_indices] = np.minimum(b[cluster_indices], mean_inter_distances)

    # Compute silhouette scores with numerical stability
    valid_samples = (a != 0) | (b != np.inf)
    s = np.zeros(n_samples)
    
    # Vectorized silhouette computation
    denominator = np.maximum(a[valid_samples], b[valid_samples])
    s[valid_samples] = (b[valid_samples] - a[valid_samples]) / denominator

    if np.sum(valid_samples) == 0:
        return 0.0  # Return 0 if all clusters have only one sample

    return float(np.mean(s[valid_samples]))

def davies_bouldin_index(x: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Davies-Bouldin Index for a clustering result.
    Optimized for better performance and numerical stability.

    Args:
        x (np.ndarray): The input data points, shape (n_samples, n_features).
        labels (np.ndarray): The cluster labels for each data point, shape (n_samples,).

    Returns:
        float: The Davies-Bouldin Index.
    """
    # Validate inputs
    _validate_clustering_inputs(x, labels)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1:
        return 0.0  # Return 0 for single cluster or empty clusters

    # Pre-compute centroids efficiently
    centroids = np.array([np.mean(x[labels == label], axis=0) for label in unique_labels])

    # Pre-compute intra-cluster dispersions
    dispersions = np.zeros(n_clusters)
    for i, label in enumerate(unique_labels):
        cluster_points = x[labels == label]
        if len(cluster_points) > 0:
            # Vectorized dispersion computation
            dispersions[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))

    # Compute pairwise centroid distances efficiently
    centroid_distances = pdist(centroids)

    # Compute Davies-Bouldin index
    db_index = 0.0
    for i in range(n_clusters):
        max_ratio = 0.0
        for j in range(i + 1, n_clusters):
            # Get distance between centroids i and j
            dist_idx = n_clusters * i + j - ((i + 2) * (i + 1)) // 2
            centroid_dist = centroid_distances[dist_idx]
            
            if centroid_dist > 1e-12:  # Avoid division by zero
                ratio = (dispersions[i] + dispersions[j]) / centroid_dist
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio

    return db_index / n_clusters

def calinski_harabasz_score(x: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Calinski-Harabasz Index for a clustering result.

    Args:
        x (np.ndarray): The input data matrix, shape (n_samples, n_features).
        labels (np.ndarray): The cluster labels for each data point, shape (n_samples,).

    Returns:
        float: The Calinski-Harabasz Index.
    """
    # Validate inputs
    _validate_clustering_inputs(x, labels)
    
    n_samples, _ = x.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    overall_centroid = np.mean(x, axis=0)

    between_dispersion = 0.0
    within_dispersion = 0.0

    for label in unique_labels:
        cluster_points = x[labels == label]
        cluster_size = cluster_points.shape[0]
        cluster_centroid = np.mean(cluster_points, axis=0)

        between_dispersion += cluster_size * np.sum((cluster_centroid - overall_centroid) ** 2)
        within_dispersion += np.sum((cluster_points - cluster_centroid) ** 2)

    return ((n_samples - n_clusters) / (n_clusters - 1)) * (between_dispersion / within_dispersion)

def dunn_index(x: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the Dunn Index for a clustering result.

    Args:
        x (np.ndarray): The input data points, shape (n_samples, n_features).
        labels (np.ndarray): The cluster labels for each data point, shape (n_samples,).

    Returns:
        float: The Dunn Index value.
    """
    # Validate inputs
    _validate_clustering_inputs(x, labels)
    
    distances = squareform(pdist(x))
    unique_labels = np.unique(labels)

    min_inter_cluster_distance = np.inf
    max_intra_cluster_distance = 0.0

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
    for feature1, feature2 in zip(data1, data2):
        hist1, _ = np.histogram(feature1, bins=num_bins, density=True)
        hist2, _ = np.histogram(feature2, bins=num_bins, density=True)
        squared_diff = np.mean(np.abs(hist1 - hist2))
        mean_acc.append(squared_diff)
    return np.mean(mean_acc)

def bcubed_precision_recall(clusters: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
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


# supervised learning function
def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predictions.

    Parameters:
    y_true (list or array): True labels
    y_pred (list or array): Predicted labels

    Returns:
    float: Accuracy as a percentage
    """
    correct = 0
    total = len(y_true)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1

    accuracy = correct / total * 100
    return accuracy

def f1_score(y_true, y_pred):
    """
    Calculate the F1 score of predictions.

    Parameters:
    y_true (list or array): True labels
    y_pred (list or array): Predicted labels

    Returns:
    float: F1 score
    """
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def recall(y_true, y_pred):
    """
    Calculate the recall of predictions.

    Parameters:
    y_true (list or array): True labels
    y_pred (list or array): Predicted labels

    Returns:
    float: Recall
    """
    tp = 0  # True positives
    fn = 0  # False negatives

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 1 and pred == 0:
            fn += 1

    if tp + fn == 0:
        return 0.0

    recall = tp / (tp + fn)
    return recall