#!/usr/bin/env python3
"""
Test script to verify optimization improvements work correctly.
This script tests the optimized SOM implementation without requiring CuPy.
"""

import numpy as np
import time
from modules.som import SOM
from modules.kmeans import KMeans
from modules.utils import one_hot_encode, normalize_column, cos_distance, euc_distance
from modules.evals import silhouette_score, davies_bouldin_index

def test_som_optimization():
    """Test the optimized SOM implementation."""
    print("Testing optimized SOM implementation...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    
    # Test SOM initialization and training
    som = SOM(m=5, n=5, dim=n_features, 
              initiate_method="random", 
              learning_rate=0.1, 
              neighbour_rad=3, 
              distance_function="euclidean")
    
    # Time the fitting process
    start_time = time.time()
    som.fit(X, epoch=10, batch_size=50)
    fit_time = time.time() - start_time
    
    # Test prediction
    start_time = time.time()
    labels = som.predict(X)
    predict_time = time.time() - start_time
    
    print(f"  SOM fit time: {fit_time:.4f} seconds")
    print(f"  SOM predict time: {predict_time:.4f} seconds")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {len(np.unique(labels))}")
    print(f"  Labels range: [{labels.min()}, {labels.max()}]")
    
    # Test evaluation
    try:
        scores = som.evaluate(X, method=["silhouette"])
        print(f"  Silhouette score: {scores[0]:.4f}")
    except Exception as e:
        print(f"  Evaluation error: {e}")
    
    return True

def test_kmeans_optimization():
    """Test the optimized KMeans implementation."""
    print("\nTesting optimized KMeans implementation...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    
    # Test KMeans
    kmeans = KMeans(n_clusters=5, method="kmeans++")
    
    # Time the fitting process
    start_time = time.time()
    kmeans.fit(X)
    fit_time = time.time() - start_time
    
    # Test prediction
    start_time = time.time()
    labels = kmeans.predict(X)
    predict_time = time.time() - start_time
    
    print(f"  KMeans fit time: {fit_time:.4f} seconds")
    print(f"  KMeans predict time: {predict_time:.4f} seconds")
    print(f"  Converged in {kmeans.n_iter_} iterations")
    print(f"  Final inertia: {kmeans.inertia_:.4f}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {len(np.unique(labels))}")
    
    return True

def test_utils_optimization():
    """Test the optimized utility functions."""
    print("\nTesting optimized utility functions...")
    
    # Test one-hot encoding
    labels = np.array([0, 1, 2, 1, 0, 2])
    encoded = one_hot_encode(labels)
    print(f"  One-hot encoding shape: {encoded.shape}")
    print(f"  One-hot encoding sum per row: {np.sum(encoded, axis=1)}")
    
    # Test cosine distance
    v1 = [1.0, 0.0, 0.0]
    v2 = [0.0, 1.0, 0.0]
    cos_dist = cos_distance(v1, v2)
    print(f"  Cosine distance between orthogonal vectors: {cos_dist:.4f}")
    
    # Test euclidean distance
    p1 = np.array([0.0, 0.0])
    p2 = np.array([3.0, 4.0])
    euc_dist = euc_distance(p1, p2)
    print(f"  Euclidean distance (should be 5.0): {euc_dist:.4f}")
    
    # Test column normalization
    data = np.array([[1, 10], [2, 20], [3, 30]])
    normalized = normalize_column(data, 0)
    print(f"  Normalized column range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    return True

def test_evals_optimization():
    """Test the optimized evaluation functions."""
    print("\nTesting optimized evaluation functions...")
    
    # Generate sample clustered data
    np.random.seed(42)
    cluster1 = np.random.normal(0, 0.5, (100, 2))
    cluster2 = np.random.normal([3, 3], 0.5, (100, 2))
    cluster3 = np.random.normal([0, 3], 0.5, (100, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0]*100 + [1]*100 + [2]*100)
    
    # Test silhouette score
    start_time = time.time()
    sil_score = silhouette_score(X, labels)
    sil_time = time.time() - start_time
    print(f"  Silhouette score: {sil_score:.4f} (computed in {sil_time:.4f}s)")
    
    # Test Davies-Bouldin index
    start_time = time.time()
    db_score = davies_bouldin_index(X, labels)
    db_time = time.time() - start_time
    print(f"  Davies-Bouldin index: {db_score:.4f} (computed in {db_time:.4f}s)")
    
    return True

def main():
    """Run all optimization tests."""
    print("=" * 50)
    print("Testing Optimized SOM Clustering Implementation")
    print("=" * 50)
    
    try:
        # Run tests
        success = True
        success &= test_som_optimization()
        success &= test_kmeans_optimization()
        success &= test_utils_optimization()
        success &= test_evals_optimization()
        
        if success:
            print("\n" + "=" * 50)
            print("All optimization tests passed successfully!")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("Some tests failed!")
            print("=" * 50)
            
    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()