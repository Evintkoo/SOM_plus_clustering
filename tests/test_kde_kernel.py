"""
Tests for the KDE kernel module.
"""
import pytest
import numpy as np
from unittest.mock import patch

# Import the KDE kernel module
import sys
sys.path.append('d:/Projects/SOM_plus_clustering')
from modules.kde_kernel import (
    gaussian_kernel,
    kde_multidimensional,
    find_local_maxima,
    bandwidth_estimator,
    initiate_kde
)


class TestGaussianKernel:
    """Test Gaussian kernel function."""
    
    def test_gaussian_kernel_basic(self):
        """Test basic Gaussian kernel computation."""
        x = np.array([0.0, 0.0])
        xi = np.array([0.0, 0.0])
        bandwidth = 1.0
        
        # At identical points, kernel should be maximum
        kernel_value = gaussian_kernel(x, xi, bandwidth)
        
        # Should be the normalization constant
        d = len(x)
        expected = 1 / (np.sqrt(2 * np.pi) ** d * bandwidth ** d)
        
        assert abs(kernel_value - expected) < 1e-10
    
    def test_gaussian_kernel_distance_decay(self):
        """Test that Gaussian kernel decays with distance."""
        x = np.array([0.0, 0.0])
        xi1 = np.array([1.0, 0.0])
        xi2 = np.array([2.0, 0.0])
        bandwidth = 1.0
        
        kernel1 = gaussian_kernel(x, xi1, bandwidth)
        kernel2 = gaussian_kernel(x, xi2, bandwidth)
        
        # Closer point should have higher kernel value
        assert kernel1 > kernel2
        assert kernel1 > 0
        assert kernel2 > 0
    
    def test_gaussian_kernel_bandwidth_effect(self):
        """Test effect of bandwidth on kernel values."""
        x = np.array([0.0, 0.0])
        xi = np.array([1.0, 1.0])
        
        kernel_small = gaussian_kernel(x, xi, bandwidth=0.5)
        kernel_large = gaussian_kernel(x, xi, bandwidth=2.0)
        
        # Smaller bandwidth should give lower values for distant points
        assert kernel_small < kernel_large
    
    def test_gaussian_kernel_multidimensional(self):
        """Test Gaussian kernel in higher dimensions."""
        x = np.array([1.0, 2.0, 3.0])
        xi = np.array([1.1, 2.1, 3.1])
        bandwidth = 1.0
        
        kernel_value = gaussian_kernel(x, xi, bandwidth)
        
        assert kernel_value > 0
        assert np.isfinite(kernel_value)
    
    def test_gaussian_kernel_symmetry(self):
        """Test that Gaussian kernel is symmetric."""
        x = np.array([1.0, 2.0])
        xi = np.array([3.0, 4.0])
        bandwidth = 1.5
        
        kernel1 = gaussian_kernel(x, xi, bandwidth)
        kernel2 = gaussian_kernel(xi, x, bandwidth)
        
        assert abs(kernel1 - kernel2) < 1e-15
    
    def test_gaussian_kernel_zero_bandwidth(self):
        """Test Gaussian kernel with very small bandwidth."""
        x = np.array([0.0, 0.0])
        xi = np.array([0.0, 0.0])
        
        # Very small bandwidth
        bandwidth = 1e-10
        kernel_value = gaussian_kernel(x, xi, bandwidth)
        
        # Should be very large but finite
        assert np.isfinite(kernel_value)
        assert kernel_value > 0


class TestKDEMultidimensional:
    """Test multidimensional KDE computation."""
    
    def test_kde_multidimensional_basic(self):
        """Test basic KDE computation."""
        # Simple 2D data
        data = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        
        # Evaluation points
        points = np.array([
            [0.5, 0.5],
            [1.5, 1.5]
        ])
        
        bandwidth = 1.0
        kde_values = kde_multidimensional(data, points, bandwidth)
        
        assert len(kde_values) == len(points)
        assert np.all(kde_values > 0)
        assert np.all(np.isfinite(kde_values))
    
    def test_kde_multidimensional_single_point(self):
        """Test KDE with single data point."""
        data = np.array([[1.0, 2.0]])
        points = np.array([[1.0, 2.0], [2.0, 3.0]])
        bandwidth = 1.0
        
        kde_values = kde_multidimensional(data, points, bandwidth)
        
        # First point (at data location) should have higher density
        assert kde_values[0] > kde_values[1]
        assert np.all(kde_values > 0)
    
    def test_kde_multidimensional_identical_points(self):
        """Test KDE with identical data points."""
        data = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        
        points = np.array([[1.0, 1.0], [2.0, 2.0]])
        bandwidth = 1.0
        
        kde_values = kde_multidimensional(data, points, bandwidth)
        
        # Point at data location should have much higher density
        assert kde_values[0] > kde_values[1]
    
    def test_kde_multidimensional_different_bandwidths(self):
        """Test KDE with different bandwidths."""
        data = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        points = np.array([[0.5, 0.5]])
        
        kde_small = kde_multidimensional(data, points, bandwidth=0.1)
        kde_large = kde_multidimensional(data, points, bandwidth=2.0)
        
        # Different bandwidths should give different values
        assert kde_small[0] != kde_large[0]
        assert np.all(kde_small > 0)
        assert np.all(kde_large > 0)
    
    def test_kde_multidimensional_higher_dimensions(self):
        """Test KDE in higher dimensions."""
        # 3D data
        np.random.seed(42)
        data = np.random.randn(10, 3)
        points = np.random.randn(5, 3)
        bandwidth = 1.0
        
        kde_values = kde_multidimensional(data, points, bandwidth)
        
        assert len(kde_values) == 5
        assert np.all(kde_values > 0)
        assert np.all(np.isfinite(kde_values))


class TestFindLocalMaxima:
    """Test local maxima finding functionality."""
    
    def test_find_local_maxima_basic(self):
        """Test basic local maxima finding."""
        # Create values with clear local maxima
        kde_values = np.array([1, 3, 2, 5, 1, 4, 2])
        points = np.arange(len(kde_values)).reshape(-1, 1)
        
        maxima = find_local_maxima(kde_values, points)
        
        # Should find maxima at indices 1, 3, 5
        expected_indices = [1, 3, 5]
        
        assert len(maxima) == 3
        for i, idx in enumerate(expected_indices):
            assert np.allclose(maxima[i], [idx])
    
    def test_find_local_maxima_no_maxima(self):
        """Test when there are no local maxima."""
        # Monotonically increasing
        kde_values = np.array([1, 2, 3, 4, 5])
        points = np.arange(len(kde_values)).reshape(-1, 1)
        
        maxima = find_local_maxima(kde_values, points)
        
        # Should find no maxima
        assert len(maxima) == 0
    
    def test_find_local_maxima_single_maximum(self):
        """Test with single local maximum."""
        kde_values = np.array([1, 2, 5, 3, 1])
        points = np.arange(len(kde_values)).reshape(-1, 1)
        
        maxima = find_local_maxima(kde_values, points)
        
        assert len(maxima) == 1
        assert np.allclose(maxima[0], [2])  # Index 2 is the maximum
    
    def test_find_local_maxima_edge_cases(self):
        """Test edge cases for local maxima finding."""
        # Very short array
        kde_values = np.array([1, 2, 1])
        points = np.arange(len(kde_values)).reshape(-1, 1)
        
        maxima = find_local_maxima(kde_values, points)
        
        assert len(maxima) == 1
        assert np.allclose(maxima[0], [1])
        
        # Two-element array (no interior points)
        kde_values = np.array([1, 2])
        points = np.arange(len(kde_values)).reshape(-1, 1)
        
        maxima = find_local_maxima(kde_values, points)
        
        assert len(maxima) == 0
    
    def test_find_local_maxima_multidimensional_points(self):
        """Test local maxima finding with multidimensional points."""
        kde_values = np.array([1, 4, 2, 5, 1])
        points = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4]
        ])
        
        maxima = find_local_maxima(kde_values, points)
        
        # Should find maxima at positions (1,1) and (3,3)
        assert len(maxima) == 2
        assert np.allclose(maxima[0], [1, 1])
        assert np.allclose(maxima[1], [3, 3])


class TestBandwidthEstimator:
    """Test bandwidth estimation functionality."""
    
    def test_bandwidth_estimator_basic(self):
        """Test basic bandwidth estimation."""
        data = np.array([1, 2, 3, 4, 5])
        
        bandwidth = bandwidth_estimator(data)
        
        # Should be positive and finite
        assert bandwidth > 0
        assert np.isfinite(bandwidth)
        
        # Check formula: (max - min) / (1 + log2(n))
        expected = (5 - 1) / (1 + np.log2(5))
        assert abs(bandwidth - expected) < 1e-10
    
    def test_bandwidth_estimator_larger_dataset(self):
        """Test bandwidth estimation with larger dataset."""
        np.random.seed(42)
        data = np.random.randn(100)
        
        bandwidth = bandwidth_estimator(data)
        
        assert bandwidth > 0
        assert np.isfinite(bandwidth)
        
        # Verify formula
        n = len(data)
        expected = (np.max(data) - np.min(data)) / (1 + np.log2(n))
        assert abs(bandwidth - expected) < 1e-10
    
    def test_bandwidth_estimator_constant_data(self):
        """Test bandwidth estimation with constant data."""
        data = np.array([5, 5, 5, 5])
        
        bandwidth = bandwidth_estimator(data)
        
        # Should be zero when all values are the same
        assert bandwidth == 0.0
    
    def test_bandwidth_estimator_two_points(self):
        """Test bandwidth estimation with minimum valid size."""
        data = np.array([1, 3])
        
        bandwidth = bandwidth_estimator(data)
        
        # Formula: (3-1) / (1 + log2(2)) = 2 / 2 = 1
        expected = 2.0 / (1 + np.log2(2))
        assert abs(bandwidth - expected) < 1e-10
    
    def test_bandwidth_estimator_insufficient_data(self):
        """Test bandwidth estimation with insufficient data."""
        data = np.array([5])
        
        with pytest.raises(ValueError, match="Data must contain at least two points"):
            bandwidth_estimator(data)
        
        # Empty array
        data = np.array([])
        
        with pytest.raises(ValueError, match="Data must contain at least two points"):
            bandwidth_estimator(data)
    
    def test_bandwidth_estimator_scaling(self):
        """Test that bandwidth scales with data range."""
        # Small range data
        data_small = np.array([1.0, 1.1, 1.2])
        bandwidth_small = bandwidth_estimator(data_small)
        
        # Large range data
        data_large = np.array([1.0, 10.0, 20.0])
        bandwidth_large = bandwidth_estimator(data_large)
        
        # Larger range should give larger bandwidth
        assert bandwidth_large > bandwidth_small


class TestInitiateKDE:
    """Test KDE initialization functionality."""
    
    def test_initiate_kde_basic(self):
        """Test basic KDE initialization."""
        np.random.seed(42)
        x = np.random.randn(50, 2)
        n_neurons = 10
        
        neurons = initiate_kde(x, n_neurons)
        
        assert neurons.shape == (n_neurons, 2)
        assert np.all(np.isfinite(neurons))
    
    def test_initiate_kde_custom_bandwidth(self):
        """Test KDE initialization with custom bandwidth."""
        np.random.seed(42)
        x = np.random.randn(30, 2)
        n_neurons = 5
        bandwidth = 0.5
        
        neurons = initiate_kde(x, n_neurons, bandwidth)
        
        assert neurons.shape == (n_neurons, 2)
        assert np.all(np.isfinite(neurons))
    
    def test_initiate_kde_insufficient_maxima(self):
        """Test KDE initialization when there are insufficient local maxima."""
        # Create data with very few local maxima
        x = np.array([
            [0, 0],
            [0.1, 0.1],
            [0.2, 0.2]
        ])
        n_neurons = 10  # More neurons than likely maxima
        
        # Should raise ValueError due to insufficient maxima
        with pytest.raises(ValueError, match="Maximum number of neurons is"):
            initiate_kde(x, n_neurons)
    
    def test_initiate_kde_higher_dimensions(self):
        """Test KDE initialization in higher dimensions."""
        np.random.seed(42)
        x = np.random.randn(100, 5)  # 5D data
        n_neurons = 15
        
        neurons = initiate_kde(x, n_neurons)
        
        assert neurons.shape == (n_neurons, 5)
        assert np.all(np.isfinite(neurons))
    
    def test_initiate_kde_neuron_selection(self):
        """Test that KDE selects diverse neurons."""
        # Create data with well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.normal([0, 0], 0.1, (20, 2))
        cluster2 = np.random.normal([5, 5], 0.1, (20, 2))
        cluster3 = np.random.normal([0, 5], 0.1, (20, 2))
        x = np.vstack([cluster1, cluster2, cluster3])
        
        n_neurons = 5
        neurons = initiate_kde(x, n_neurons, bandwidth=0.5)
        
        assert neurons.shape == (n_neurons, 2)
        
        # Neurons should be diverse (not all clustered together)
        pairwise_distances = []
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                dist = np.linalg.norm(neurons[i] - neurons[j])
                pairwise_distances.append(dist)
        
        # At least some pairs should be well-separated
        assert max(pairwise_distances) > 1.0
    
    def test_initiate_kde_single_neuron(self):
        """Test KDE initialization with single neuron."""
        np.random.seed(42)
        x = np.random.randn(20, 2)
        n_neurons = 1
        
        neurons = initiate_kde(x, n_neurons)
        
        assert neurons.shape == (1, 2)
        assert np.all(np.isfinite(neurons))
    
    def test_initiate_kde_reproducibility(self):
        """Test that KDE initialization is reproducible."""
        np.random.seed(42)
        x = np.random.randn(50, 2)
        n_neurons = 8
        
        np.random.seed(123)
        neurons1 = initiate_kde(x, n_neurons, bandwidth=1.0)
        
        np.random.seed(123)
        neurons2 = initiate_kde(x, n_neurons, bandwidth=1.0)
        
        # Should be identical with same random seed
        assert np.allclose(neurons1, neurons2)


class TestKDEEdgeCases:
    """Test edge cases and error conditions for KDE functions."""
    
    def test_gaussian_kernel_extreme_values(self):
        """Test Gaussian kernel with extreme values."""
        # Very large coordinates
        x = np.array([1e6, 1e6])
        xi = np.array([1e6 + 1, 1e6 + 1])
        bandwidth = 1.0
        
        kernel_value = gaussian_kernel(x, xi, bandwidth)
        
        assert np.isfinite(kernel_value)
        assert kernel_value >= 0
        
        # Very small coordinates
        x = np.array([1e-10, 1e-10])
        xi = np.array([2e-10, 2e-10])
        
        kernel_value = gaussian_kernel(x, xi, bandwidth)
        
        assert np.isfinite(kernel_value)
        assert kernel_value >= 0
    
    def test_kde_with_zero_bandwidth(self):
        """Test KDE behavior with very small bandwidth."""
        data = np.array([[0, 0], [1, 1]])
        points = np.array([[0, 0]])
        
        # Very small bandwidth should still work
        kde_values = kde_multidimensional(data, points, bandwidth=1e-10)
        
        assert np.all(np.isfinite(kde_values))
        assert np.all(kde_values >= 0)
    
    def test_kde_with_large_bandwidth(self):
        """Test KDE behavior with very large bandwidth."""
        data = np.array([[0, 0], [1, 1], [2, 2]])
        points = np.array([[0, 0], [1, 1], [2, 2]])
        
        # Very large bandwidth should smooth everything
        kde_values = kde_multidimensional(data, points, bandwidth=1e6)
        
        assert np.all(np.isfinite(kde_values))
        assert np.all(kde_values > 0)
        
        # Values should be very similar due to large bandwidth
        assert max(kde_values) / min(kde_values) < 2.0
    
    def test_bandwidth_estimator_extreme_cases(self):
        """Test bandwidth estimator with extreme data."""
        # Very large values
        data = np.array([1e6, 2e6, 3e6])
        bandwidth = bandwidth_estimator(data)
        
        assert np.isfinite(bandwidth)
        assert bandwidth > 0
        
        # Very small values
        data = np.array([1e-6, 2e-6, 3e-6])
        bandwidth = bandwidth_estimator(data)
        
        assert np.isfinite(bandwidth)
        assert bandwidth > 0
        
        # Mixed positive/negative
        data = np.array([-100, 0, 100])
        bandwidth = bandwidth_estimator(data)
        
        assert np.isfinite(bandwidth)
        assert bandwidth > 0
    
    def test_initiate_kde_edge_cases(self):
        """Test KDE initialization edge cases."""
        # Very small dataset
        x = np.array([[1, 2], [1.1, 2.1]])
        
        try:
            neurons = initiate_kde(x, 1)
            assert neurons.shape == (1, 2)
        except ValueError:
            pass  # Acceptable if insufficient local maxima
        
        # Dataset with identical points
        x = np.array([[1, 1], [1, 1], [1, 1]])
        
        try:
            neurons = initiate_kde(x, 1)
            assert neurons.shape == (1, 2)
        except ValueError:
            pass  # Acceptable behavior


class TestKDEIntegration:
    """Integration tests for KDE functionality."""
    
    def test_complete_kde_workflow(self):
        """Test complete KDE workflow from data to neuron selection."""
        np.random.seed(42)
        
        # Create multi-modal data
        cluster1 = np.random.normal([0, 0], 0.5, (30, 2))
        cluster2 = np.random.normal([4, 4], 0.5, (30, 2))
        cluster3 = np.random.normal([0, 4], 0.5, (30, 2))
        x = np.vstack([cluster1, cluster2, cluster3])
        
        # Estimate bandwidth
        bandwidth = bandwidth_estimator(x.flatten())
        assert bandwidth > 0
        
        # Compute KDE on the data itself
        kde_values = kde_multidimensional(x, x, bandwidth)
        assert len(kde_values) == len(x)
        assert np.all(kde_values > 0)
        
        # Find local maxima
        maxima = find_local_maxima(kde_values, x)
        
        # Should find some maxima
        assert len(maxima) > 0
        
        # Select neurons
        if len(maxima) >= 5:
            neurons = initiate_kde(x, 5, bandwidth)
            assert neurons.shape == (5, 2)
            assert np.all(np.isfinite(neurons))
    
    def test_kde_with_different_data_distributions(self):
        """Test KDE with different data distributions."""
        np.random.seed(42)
        
        # Uniform distribution
        x_uniform = np.random.uniform(-2, 2, (50, 2))
        neurons_uniform = initiate_kde(x_uniform, 5)
        assert neurons_uniform.shape == (5, 2)
        
        # Normal distribution
        x_normal = np.random.normal(0, 1, (50, 2))
        neurons_normal = initiate_kde(x_normal, 5)
        assert neurons_normal.shape == (5, 2)
        
        # Different distributions should give different results
        assert not np.allclose(neurons_uniform, neurons_normal)
    
    def test_kde_scaling_behavior(self):
        """Test KDE behavior with different dataset sizes."""
        np.random.seed(42)
        
        # Small dataset
        x_small = np.random.randn(20, 2)
        
        # Large dataset
        x_large = np.random.randn(200, 2)
        
        # Both should work
        try:
            neurons_small = initiate_kde(x_small, 3)
            assert neurons_small.shape == (3, 2)
        except ValueError:
            pass  # May not have enough local maxima
        
        neurons_large = initiate_kde(x_large, 10)
        assert neurons_large.shape == (10, 2)
    
    def test_kde_parameter_sensitivity(self):
        """Test sensitivity to KDE parameters."""
        np.random.seed(42)
        x = np.random.randn(100, 2)
        
        # Different bandwidths should potentially give different results
        try:
            neurons_small_bw = initiate_kde(x, 5, bandwidth=0.1)
            neurons_large_bw = initiate_kde(x, 5, bandwidth=2.0)
            
            # Results may be different (but not required to be)
            assert neurons_small_bw.shape == neurons_large_bw.shape
            assert np.all(np.isfinite(neurons_small_bw))
            assert np.all(np.isfinite(neurons_large_bw))
        except ValueError:
            pass  # May not have enough local maxima with some bandwidths