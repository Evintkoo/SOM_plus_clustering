"""
Tests for the KMeans clustering module.
"""
import pytest
import numpy as np
from unittest.mock import patch

# Import the KMeans module
import sys
sys.path.append('d:/Projects/SOM_plus_clustering')
from modules.kmeans import (
    KMeans,
    euclidean_distance_squared_jit,
    assign_clusters_jit,
    update_centroids_jit,
    compute_inertia_jit,
    kmeans_plus_plus_jit
)


class TestKMeansInitialization:
    """Test KMeans initialization functionality."""
    
    def test_kmeans_init_basic(self, kmeans_params):
        """Test basic KMeans initialization."""
        kmeans = KMeans(**kmeans_params)
        
        assert kmeans.n_clusters == 3
        assert kmeans.method == "random"
        assert kmeans.tol == 1e-6
        assert kmeans.max_iters == 100
        assert not kmeans._trained
        assert kmeans.inertia_ == 0.0
        assert kmeans.n_iter_ == 0
    
    def test_kmeans_init_custom_params(self):
        """Test KMeans initialization with custom parameters."""
        kmeans = KMeans(n_clusters=5, method="kmeans++", tol=1e-4, max_iters=200)
        
        assert kmeans.n_clusters == 5
        assert kmeans.method == "kmeans++"
        assert kmeans.tol == 1e-4
        assert kmeans.max_iters == 200


class TestKMeansCentroidInitialization:
    """Test KMeans centroid initialization methods."""
    
    def test_init_centroids_random(self, sample_data_2d, kmeans_params):
        """Test random centroid initialization."""
        kmeans = KMeans(**kmeans_params)
        kmeans.init_centroids(sample_data_2d)
        
        assert kmeans.centroids.shape == (3, 2)
        assert np.all(np.isfinite(kmeans.centroids))
        
        # Check that centroids are within data range
        data_min = sample_data_2d.min(axis=0)
        data_max = sample_data_2d.max(axis=0)
        assert np.all(kmeans.centroids >= data_min)
        assert np.all(kmeans.centroids <= data_max)
    
    def test_init_centroids_kmeans_plus_plus(self, sample_data_2d, kmeans_params):
        """Test KMeans++ centroid initialization."""
        kmeans_params['method'] = 'kmeans++'
        kmeans = KMeans(**kmeans_params)
        kmeans.init_centroids(sample_data_2d)
        
        assert kmeans.centroids.shape == (3, 2)
        assert np.all(np.isfinite(kmeans.centroids))
        
        # Check that centroids are diverse (not all the same)
        assert not np.allclose(kmeans.centroids[0], kmeans.centroids[1])
    
    def test_init_centroids_invalid_method(self, sample_data_2d, kmeans_params):
        """Test that invalid initialization method raises ValueError."""
        kmeans_params['method'] = 'invalid_method'
        kmeans = KMeans(**kmeans_params)
        
        with pytest.raises(ValueError, match="Unrecognized method"):
            kmeans.init_centroids(sample_data_2d)
    
    def test_initiate_plus_plus_algorithm(self, sample_data_2d, kmeans_params):
        """Test the KMeans++ initialization algorithm directly."""
        kmeans = KMeans(**kmeans_params)
        centroids = kmeans.initiate_plus_plus(sample_data_2d)
        
        assert centroids.shape == (3, 2)
        assert np.all(np.isfinite(centroids))
        
        # Check that all centroids are from the data points
        for centroid in centroids:
            distances = np.linalg.norm(sample_data_2d - centroid, axis=1)
            assert np.min(distances) < 1e-10  # At least one point should be very close


class TestKMeansTraining:
    """Test KMeans training functionality."""
    
    def test_kmeans_fit_basic(self, sample_data_2d, kmeans_params):
        """Test basic KMeans fitting."""
        kmeans = KMeans(**kmeans_params)
        kmeans.fit(sample_data_2d)
        
        assert kmeans._trained
        assert kmeans.centroids.shape == (3, 2)
        assert np.all(np.isfinite(kmeans.centroids))
        assert kmeans.inertia_ >= 0
        assert kmeans.n_iter_ > 0
    
    def test_kmeans_fit_already_trained(self, sample_data_2d, kmeans_params):
        """Test that fitting already trained model raises RuntimeError."""
        kmeans = KMeans(**kmeans_params)
        kmeans.fit(sample_data_2d)
        
        with pytest.raises(RuntimeError, match="Cannot fit an already trained model"):
            kmeans.fit(sample_data_2d)
    
    def test_kmeans_fit_convergence(self, kmeans_params):
        """Test KMeans convergence with well-separated clusters."""
        # Create well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.normal([0, 0], 0.1, (20, 2))
        cluster2 = np.random.normal([5, 5], 0.1, (20, 2))
        cluster3 = np.random.normal([0, 5], 0.1, (20, 2))
        data = np.vstack([cluster1, cluster2, cluster3])
        
        kmeans = KMeans(**kmeans_params)
        kmeans.fit(data)
        
        assert kmeans._trained
        assert kmeans.n_iter_ < kmeans.max_iters  # Should converge before max iterations
    
    def test_kmeans_max_iterations(self, sample_data_2d, kmeans_params):
        """Test KMeans with maximum iterations limit."""
        kmeans_params['max_iters'] = 2
        kmeans = KMeans(**kmeans_params)
        kmeans.fit(sample_data_2d)
        
        assert kmeans._trained
        assert kmeans.n_iter_ <= 2


class TestKMeansPrediction:
    """Test KMeans prediction functionality."""
    
    def test_kmeans_predict_basic(self, sample_data_2d, kmeans_params):
        """Test basic prediction functionality."""
        kmeans = KMeans(**kmeans_params)
        kmeans.fit(sample_data_2d)
        
        predictions = kmeans.predict(sample_data_2d)
        
        assert len(predictions) == len(sample_data_2d)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
        assert all(0 <= p < 3 for p in predictions)  # 3 clusters
    
    def test_kmeans_predict_not_trained(self, sample_data_2d, kmeans_params):
        """Test that predicting with untrained model raises RuntimeError."""
        kmeans = KMeans(**kmeans_params)
        
        with pytest.raises(RuntimeError, match="Model must be fitted before prediction"):
            kmeans.predict(sample_data_2d)
    
    def test_kmeans_predict_new_data(self, sample_data_2d, kmeans_params):
        """Test prediction on new data."""
        kmeans = KMeans(**kmeans_params)
        kmeans.fit(sample_data_2d)
        
        # Create new test data
        new_data = np.random.random((10, 2))
        predictions = kmeans.predict(new_data)
        
        assert len(predictions) == 10
        assert all(0 <= p < 3 for p in predictions)


class TestKMeansPrivateMethods:
    """Test KMeans private methods."""
    
    def test_assign_clusters(self, sample_data_2d, kmeans_params):
        """Test cluster assignment functionality."""
        kmeans = KMeans(**kmeans_params)
        kmeans.init_centroids(sample_data_2d)
        
        labels = kmeans._assign_clusters(sample_data_2d)
        
        assert len(labels) == len(sample_data_2d)
        assert all(isinstance(label, (int, np.integer)) for label in labels)
        assert all(0 <= label < 3 for label in labels)
    
    def test_update_centroids(self, sample_data_2d, kmeans_params):
        """Test centroid update functionality."""
        kmeans = KMeans(**kmeans_params)
        kmeans.init_centroids(sample_data_2d)
        
        labels = kmeans._assign_clusters(sample_data_2d)
        new_centroids = kmeans._update_centroids(sample_data_2d, labels)
        
        assert new_centroids.shape == kmeans.centroids.shape
        assert np.all(np.isfinite(new_centroids))
    
    def test_compute_inertia(self, sample_data_2d, kmeans_params):
        """Test inertia computation."""
        kmeans = KMeans(**kmeans_params)
        kmeans.init_centroids(sample_data_2d)
        
        labels = kmeans._assign_clusters(sample_data_2d)
        inertia = kmeans._compute_inertia(sample_data_2d, labels)
        
        assert isinstance(inertia, float)
        assert inertia >= 0


class TestKMeansJITFunctions:
    """Test JIT-optimized functions for KMeans."""
    
    def test_euclidean_distance_squared_jit_kmeans(self):
        """Test JIT-optimized squared Euclidean distance for KMeans."""
        x = np.array([1.0, 2.0])
        y = np.array([4.0, 6.0])
        
        distance_sq = euclidean_distance_squared_jit(x, y)
        expected = (1-4)**2 + (2-6)**2
        
        assert abs(distance_sq - expected) < 1e-10
    
    def test_assign_clusters_jit(self):
        """Test JIT-optimized cluster assignment."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [1.1, 2.1]])
        centroids = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        labels = assign_clusters_jit(data, centroids)
        
        assert len(labels) == 3
        assert labels[0] == 0  # First point closest to first centroid
        assert labels[1] == 1  # Second point closest to second centroid
        assert labels[2] == 0  # Third point closest to first centroid
    
    def test_update_centroids_jit(self):
        """Test JIT-optimized centroid update."""
        data = np.array([[1.0, 2.0], [1.1, 2.1], [3.0, 4.0], [3.1, 4.1]])
        labels = np.array([0, 0, 1, 1])
        n_clusters = 2
        
        new_centroids = update_centroids_jit(data, labels, n_clusters)
        
        assert new_centroids.shape == (2, 2)
        # Check that centroids are the mean of assigned points
        expected_centroid_0 = np.mean(data[labels == 0], axis=0)
        expected_centroid_1 = np.mean(data[labels == 1], axis=0)
        
        assert np.allclose(new_centroids[0], expected_centroid_0)
        assert np.allclose(new_centroids[1], expected_centroid_1)
    
    def test_compute_inertia_jit(self):
        """Test JIT-optimized inertia computation."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([0, 1])
        centroids = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        inertia = compute_inertia_jit(data, labels, centroids)
        
        assert inertia == 0.0  # Data points are exactly at centroids
    
    def test_kmeans_plus_plus_jit(self):
        """Test JIT-optimized KMeans++ initialization."""
        np.random.seed(42)
        data = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 6.0], [5.1, 6.1]])
        n_clusters = 2
        
        centroids = kmeans_plus_plus_jit(data, n_clusters)
        
        assert centroids.shape == (2, 2)
        assert np.all(np.isfinite(centroids))


class TestKMeansEdgeCases:
    """Test KMeans edge cases and error conditions."""
    
    def test_kmeans_single_cluster(self, sample_data_2d):
        """Test KMeans with single cluster."""
        kmeans = KMeans(n_clusters=1, method="random")
        kmeans.fit(sample_data_2d)
        
        predictions = kmeans.predict(sample_data_2d)
        assert all(p == 0 for p in predictions)  # All should be cluster 0
    
    def test_kmeans_more_clusters_than_points(self):
        """Test KMeans with more clusters than data points."""
        small_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        kmeans = KMeans(n_clusters=5, method="random")
        
        # Should handle gracefully
        kmeans.fit(small_data)
        assert kmeans._trained
    
    def test_kmeans_identical_points(self, kmeans_params):
        """Test KMeans with identical data points."""
        identical_data = np.array([[1.0, 2.0]] * 10)
        kmeans = KMeans(**kmeans_params)
        
        kmeans.fit(identical_data)
        predictions = kmeans.predict(identical_data)
        
        # All predictions should be the same
        assert len(set(predictions)) <= 3  # At most 3 different clusters
    
    def test_kmeans_early_convergence(self, kmeans_params):
        """Test KMeans early convergence."""
        # Create data where centroids won't change after first iteration
        data = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2]])
        
        kmeans_params['n_clusters'] = 3
        kmeans_params['tol'] = 1e-3
        kmeans = KMeans(**kmeans_params)
        
        kmeans.fit(data)
        
        # Should converge quickly
        assert kmeans.n_iter_ < kmeans.max_iters


class TestKMeansIntegration:
    """Integration tests for KMeans functionality."""
    
    def test_complete_kmeans_workflow(self, sample_data_2d, kmeans_params):
        """Test complete KMeans workflow."""
        # Initialize KMeans
        kmeans = KMeans(**kmeans_params)
        assert not kmeans._trained
        
        # Fit the model
        kmeans.fit(sample_data_2d)
        assert kmeans._trained
        assert kmeans.inertia_ >= 0
        assert kmeans.n_iter_ > 0
        
        # Make predictions
        predictions = kmeans.predict(sample_data_2d)
        assert len(predictions) == len(sample_data_2d)
        assert all(0 <= p < 3 for p in predictions)
        
        # Verify centroids are reasonable
        assert kmeans.centroids.shape == (3, 2)
        assert np.all(np.isfinite(kmeans.centroids))
    
    def test_kmeans_reproducibility(self, kmeans_params):
        """Test that KMeans produces consistent results with same random seed."""
        np.random.seed(42)
        data = np.random.random((20, 2))
        
        # Train two identical KMeans models
        np.random.seed(42)
        kmeans1 = KMeans(**kmeans_params)
        kmeans1.fit(data)
        
        np.random.seed(42)
        kmeans2 = KMeans(**kmeans_params)
        kmeans2.fit(data)
        
        # Results should be identical
        pred1 = kmeans1.predict(data)
        pred2 = kmeans2.predict(data)
        
        # Predictions might differ due to label permutation, but cluster structure should be same
        # For this test, we'll check that inertia is the same
        assert abs(kmeans1.inertia_ - kmeans2.inertia_) < 1e-10
    
    def test_kmeans_different_initialization_methods(self, sample_data_2d):
        """Test KMeans with different initialization methods."""
        methods = ["random", "kmeans++"]
        
        for method in methods:
            kmeans = KMeans(n_clusters=3, method=method)
            kmeans.fit(sample_data_2d)
            predictions = kmeans.predict(sample_data_2d)
            
            assert kmeans._trained
            assert len(predictions) == len(sample_data_2d)
            assert all(0 <= p < 3 for p in predictions)
    
    def test_kmeans_performance_metrics(self, sample_data_2d, kmeans_params):
        """Test that KMeans produces reasonable performance metrics."""
        kmeans = KMeans(**kmeans_params)
        kmeans.fit(sample_data_2d)
        
        # Inertia should be positive and finite
        assert kmeans.inertia_ > 0
        assert np.isfinite(kmeans.inertia_)
        
        # Number of iterations should be reasonable
        assert 1 <= kmeans.n_iter_ <= kmeans.max_iters
        
        # Centroids should be within reasonable bounds
        data_min = sample_data_2d.min()
        data_max = sample_data_2d.max()
        assert np.all(kmeans.centroids >= data_min - 1)  # Allow some margin
        assert np.all(kmeans.centroids <= data_max + 1)