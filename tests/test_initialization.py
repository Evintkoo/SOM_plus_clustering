"""
Tests for the initialization module.
"""
import pytest
import numpy as np
import math

# Import the initialization module
import sys
sys.path.append('d:/Projects/SOM_plus_clustering')
from modules.initialization import (
    initiate_zero,
    hadamard_matrix,
    initiate_naive_sharding,
    initiate_he,
    initiate_lecun,
    svd_orthonormal,
    initiate_lsuv
)


class TestHadamardMatrix:
    """Test Hadamard matrix generation."""
    
    def test_hadamard_matrix_m0(self):
        """Test Hadamard matrix for m=0."""
        H = hadamard_matrix(0)
        expected = np.array([[1]])
        
        assert np.array_equal(H, expected)
    
    def test_hadamard_matrix_m1(self):
        """Test Hadamard matrix for m=1."""
        H = hadamard_matrix(1)
        expected = np.array([[1, 1], [1, -1]])
        
        assert np.array_equal(H, expected)
    
    def test_hadamard_matrix_m2(self):
        """Test Hadamard matrix for m=2."""
        H = hadamard_matrix(2)
        expected = np.array([
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1]
        ])
        
        assert np.array_equal(H, expected)
    
    def test_hadamard_matrix_properties(self):
        """Test mathematical properties of Hadamard matrices."""
        for m in range(4):
            H = hadamard_matrix(m)
            size = 2**m
            
            # Check size
            assert H.shape == (size, size)
            
            # Check that all elements are ±1
            assert np.all(np.abs(H) == 1)
            
            # Check orthogonality: H^T * H = size * I
            if size > 1:
                product = H.T @ H
                expected = size * np.eye(size)
                assert np.allclose(product, expected)
    
    def test_hadamard_matrix_larger(self):
        """Test Hadamard matrix for larger values."""
        H = hadamard_matrix(3)
        
        assert H.shape == (8, 8)
        assert np.all(np.abs(H) == 1)
        
        # Check orthogonality
        product = H.T @ H
        expected = 8 * np.eye(8)
        assert np.allclose(product, expected)


class TestZeroInitialization:
    """Test zero initialization method."""
    
    def test_initiate_zero_identity_case(self):
        """Test zero initialization when P == Q (identity mapping)."""
        P, Q = 3, 3
        W = initiate_zero(P, Q)
        
        assert W.shape == (P, Q)
        assert np.array_equal(W, np.eye(3))
    
    def test_initiate_zero_expand_case(self):
        """Test zero initialization when P < Q (propagate first P dimensions)."""
        P, Q = 2, 4
        W = initiate_zero(P, Q)
        
        assert W.shape == (P, Q)
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        assert np.array_equal(W, expected)
    
    def test_initiate_zero_hadamard_case(self):
        """Test zero initialization when P > Q (apply Hadamard matrix)."""
        P, Q = 4, 2
        W = initiate_zero(P, Q)
        
        assert W.shape == (P, Q)
        # Check that it uses Hadamard transformation
        assert np.all(np.isfinite(W))
        
        # The result should be a scaled Hadamard transformation
        # Check that the scaling factor is correct
        m = int(np.ceil(np.log2(P)))
        c = 2 ** (-(m - 1) / 2)
        
        # Values should be scaled by c
        unique_vals = np.unique(np.abs(W[W != 0]))
        if len(unique_vals) > 0:
            assert np.allclose(unique_vals[0], c, atol=1e-10)
    
    def test_initiate_zero_edge_cases(self):
        """Test zero initialization edge cases."""
        # Single dimension
        W = initiate_zero(1, 1)
        assert np.array_equal(W, np.array([[1]]))
        
        # P=1, Q>1
        W = initiate_zero(1, 3)
        expected = np.array([[1, 0, 0]])
        assert np.array_equal(W, expected)
        
        # P>1, Q=1
        W = initiate_zero(3, 1)
        assert W.shape == (3, 1)
        assert np.all(np.isfinite(W))
    
    def test_initiate_zero_larger_dimensions(self):
        """Test zero initialization with larger dimensions."""
        P, Q = 8, 3
        W = initiate_zero(P, Q)
        
        assert W.shape == (P, Q)
        assert np.all(np.isfinite(W))
        
        # Should use Hadamard matrix approach
        # Check that it's not just zeros or ones
        assert not np.all(W == 0)
        assert not np.all(W == 1)


class TestNaiveSharding:
    """Test naive sharding initialization method."""
    
    def test_initiate_naive_sharding_basic(self):
        """Test basic naive sharding functionality."""
        # Create simple dataset
        X = np.array([
            [1, 1],  # Sum = 2
            [2, 2],  # Sum = 4
            [3, 3],  # Sum = 6
            [4, 4],  # Sum = 8
            [5, 5],  # Sum = 10
            [6, 6]   # Sum = 12
        ])
        k = 3
        
        centroids = initiate_naive_sharding(X, k)
        
        assert centroids.shape == (k, 2)
        assert np.all(np.isfinite(centroids))
        
        # Check that centroids are reasonable (should be between data bounds)
        data_min = X.min(axis=0)
        data_max = X.max(axis=0)
        assert np.all(centroids >= data_min)
        assert np.all(centroids <= data_max)
    
    def test_initiate_naive_sharding_sorted_behavior(self):
        """Test that naive sharding respects sorting by composite values."""
        # Create dataset where composite values determine ordering
        X = np.array([
            [0, 1],  # Sum = 1
            [1, 0],  # Sum = 1
            [2, 2],  # Sum = 4
            [3, 3],  # Sum = 6
            [1, 4],  # Sum = 5
            [4, 1]   # Sum = 5
        ])
        k = 2
        
        centroids = initiate_naive_sharding(X, k)
        
        assert centroids.shape == (k, 2)
        assert np.all(np.isfinite(centroids))
    
    def test_initiate_naive_sharding_single_cluster(self):
        """Test naive sharding with single cluster."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        k = 1
        
        centroids = initiate_naive_sharding(X, k)
        
        assert centroids.shape == (1, 2)
        # Should be the mean of all data
        expected = np.mean(X, axis=0)
        assert np.allclose(centroids[0], expected)
    
    def test_initiate_naive_sharding_more_clusters_than_points(self):
        """Test naive sharding when k > number of data points."""
        X = np.array([[1, 2], [3, 4]])
        k = 5
        
        centroids = initiate_naive_sharding(X, k)
        
        assert centroids.shape == (k, 2)
        # Some centroids might be duplicates, but should be finite
        assert np.all(np.isfinite(centroids))
    
    def test_initiate_naive_sharding_identical_composite_values(self):
        """Test naive sharding with identical composite values."""
        # All points have the same sum
        X = np.array([
            [1, 2],  # Sum = 3
            [0, 3],  # Sum = 3
            [2, 1],  # Sum = 3
            [3, 0]   # Sum = 3
        ])
        k = 2
        
        centroids = initiate_naive_sharding(X, k)
        
        assert centroids.shape == (k, 2)
        assert np.all(np.isfinite(centroids))


class TestHeInitialization:
    """Test He initialization method."""
    
    def test_initiate_he_basic(self):
        """Test basic He initialization."""
        input_dim = 10
        output_dim = 5
        
        weights = initiate_he(input_dim, output_dim)
        
        assert weights.shape == (output_dim, input_dim)
        assert np.all(np.isfinite(weights))
    
    def test_initiate_he_variance(self):
        """Test that He initialization has correct variance."""
        input_dim = 100
        output_dim = 50
        
        # Generate multiple initializations to test variance
        weights_list = [initiate_he(input_dim, output_dim) for _ in range(10)]
        all_weights = np.concatenate([w.flatten() for w in weights_list])
        
        # Theoretical variance for He initialization
        expected_variance = 2.0 / input_dim
        actual_variance = np.var(all_weights)
        
        # Allow some tolerance due to randomness
        assert abs(actual_variance - expected_variance) < 0.1
    
    def test_initiate_he_mean(self):
        """Test that He initialization has zero mean."""
        input_dim = 50
        output_dim = 25
        
        weights = initiate_he(input_dim, output_dim)
        
        # Mean should be close to zero
        assert abs(np.mean(weights)) < 0.1
    
    def test_initiate_he_different_dimensions(self):
        """Test He initialization with different dimensions."""
        test_cases = [(1, 1), (5, 10), (20, 5), (100, 100)]
        
        for input_dim, output_dim in test_cases:
            weights = initiate_he(input_dim, output_dim)
            
            assert weights.shape == (output_dim, input_dim)
            assert np.all(np.isfinite(weights))
            
            # For very small matrices, skip variance checking as it's not meaningful
            if weights.size >= 4:
                # Standard deviation should be approximately sqrt(2/input_dim)
                expected_std = np.sqrt(2.0 / input_dim)
                actual_std = np.std(weights)
                
                # Allow some tolerance
                assert abs(actual_std - expected_std) < 0.5


class TestLecunInitialization:
    """Test LeCun initialization method."""
    
    def test_initiate_lecun_basic(self):
        """Test basic LeCun initialization."""
        input_shape = 10
        output_shape = 5
        
        weights = initiate_lecun(input_shape, output_shape)
        
        assert weights.shape == (input_shape, output_shape)
        assert np.all(np.isfinite(weights))
    
    def test_initiate_lecun_variance(self):
        """Test that LeCun initialization has correct variance."""
        input_shape = 100
        output_shape = 50
        
        # Generate multiple initializations to test variance
        weights_list = [initiate_lecun(input_shape, output_shape) for _ in range(10)]
        all_weights = np.concatenate([w.flatten() for w in weights_list])
        
        # Theoretical variance for LeCun initialization
        expected_variance = 1.0 / input_shape
        actual_variance = np.var(all_weights)
        
        # Allow some tolerance due to randomness
        assert abs(actual_variance - expected_variance) < 0.1
    
    def test_initiate_lecun_mean(self):
        """Test that LeCun initialization has zero mean."""
        input_shape = 50
        output_shape = 25
        
        weights = initiate_lecun(input_shape, output_shape)
        
        # Mean should be close to zero
        assert abs(np.mean(weights)) < 0.1
    
    def test_initiate_lecun_different_dimensions(self):
        """Test LeCun initialization with different dimensions."""
        test_cases = [(1, 1), (5, 10), (20, 5), (100, 100)]
        
        for input_shape, output_shape in test_cases:
            weights = initiate_lecun(input_shape, output_shape)
            
            assert weights.shape == (input_shape, output_shape)
            assert np.all(np.isfinite(weights))
            
            # For very small matrices, skip variance checking as it's not meaningful
            if weights.size >= 4:
                # Standard deviation should be approximately sqrt(1/input_shape)
                expected_std = np.sqrt(1.0 / input_shape)
                actual_std = np.std(weights)
                
                # Allow some tolerance
                assert abs(actual_std - expected_std) < 0.5


class TestSVDOrthonormal:
    """Test SVD orthonormal matrix generation."""
    
    def test_svd_orthonormal_basic(self):
        """Test basic SVD orthonormal matrix generation."""
        shape = (5, 5)
        
        matrix = svd_orthonormal(shape)
        
        assert matrix.shape == shape
        assert np.all(np.isfinite(matrix))
    
    def test_svd_orthonormal_orthogonality(self):
        """Test that SVD orthonormal matrices are orthonormal."""
        shape = (4, 4)
        
        matrix = svd_orthonormal(shape)
        
        # Check orthonormality: Q^T * Q = I
        product = matrix.T @ matrix
        identity = np.eye(shape[0])
        
        assert np.allclose(product, identity, atol=1e-10)
    
    def test_svd_orthonormal_rectangular(self):
        """Test SVD orthonormal with rectangular matrices."""
        shapes = [(3, 5), (5, 3), (2, 4), (4, 2)]
        
        for shape in shapes:
            matrix = svd_orthonormal(shape)
            
            assert matrix.shape == shape
            assert np.all(np.isfinite(matrix))
            
            # Check partial orthonormality
            if shape[0] <= shape[1]:
                # Q^T * Q = I (for tall matrices)
                product = matrix @ matrix.T
                identity = np.eye(shape[0])
                assert np.allclose(product, identity, atol=1e-10)
            else:
                # Q * Q^T = I (for wide matrices)
                product = matrix.T @ matrix
                identity = np.eye(shape[1])
                assert np.allclose(product, identity, atol=1e-10)
    
    def test_svd_orthonormal_invalid_shape(self):
        """Test that invalid shapes raise ValueError."""
        with pytest.raises(ValueError, match="Shape must have exactly 2 dimensions"):
            svd_orthonormal((5,))
        
        with pytest.raises(ValueError, match="Shape must have exactly 2 dimensions"):
            svd_orthonormal((2, 3, 4))


class TestLSUVInitialization:
    """Test LSUV initialization method."""
    
    def test_initiate_lsuv_basic(self):
        """Test basic LSUV initialization."""
        input_dim = 5
        output_dim = 3
        X_batch = np.random.randn(10, input_dim)
        
        weights = initiate_lsuv(input_dim, output_dim, X_batch)
        
        assert weights.shape == (input_dim, output_dim)
        assert np.all(np.isfinite(weights))
    
    def test_initiate_lsuv_variance_adjustment(self):
        """Test that LSUV adjusts variance towards 1."""
        input_dim = 10
        output_dim = 5
        X_batch = np.random.randn(50, input_dim)
        
        weights = initiate_lsuv(input_dim, output_dim, X_batch, tol=0.01, max_iter=20)
        
        # Compute activations
        activations = X_batch @ weights
        variance = np.var(activations)
        
        # Variance should be close to 1
        assert abs(variance - 1.0) < 0.1
    
    def test_initiate_lsuv_orthonormal_start(self):
        """Test that LSUV starts with orthonormal initialization."""
        input_dim = 4
        output_dim = 4
        X_batch = np.random.randn(20, input_dim)
        
        # Test with max_iter=0 to see initial orthonormal matrix
        weights = initiate_lsuv(input_dim, output_dim, X_batch, max_iter=0)
        
        # Should be orthonormal (or close to it)
        product = weights.T @ weights
        identity = np.eye(output_dim)
        
        assert np.allclose(product, identity, atol=1e-10)
    
    def test_initiate_lsuv_convergence(self):
        """Test that LSUV converges within max iterations."""
        input_dim = 6
        output_dim = 3
        X_batch = np.random.randn(30, input_dim)
        
        # Use tight tolerance to test convergence
        weights = initiate_lsuv(input_dim, output_dim, X_batch, tol=0.001, max_iter=50)
        
        activations = X_batch @ weights
        variance = np.var(activations)
        
        # Should achieve the tolerance
        assert abs(variance - 1.0) < 0.01
    
    def test_initiate_lsuv_different_batch_sizes(self):
        """Test LSUV with different batch sizes."""
        input_dim = 5
        output_dim = 3
        
        batch_sizes = [5, 20, 100]
        
        for batch_size in batch_sizes:
            X_batch = np.random.randn(batch_size, input_dim)
            weights = initiate_lsuv(input_dim, output_dim, X_batch)
            
            assert weights.shape == (input_dim, output_dim)
            assert np.all(np.isfinite(weights))
    
    def test_initiate_lsuv_rectangular_matrices(self):
        """Test LSUV with rectangular weight matrices."""
        test_cases = [(8, 3), (3, 8), (10, 5)]
        
        for input_dim, output_dim in test_cases:
            X_batch = np.random.randn(25, input_dim)
            weights = initiate_lsuv(input_dim, output_dim, X_batch)
            
            assert weights.shape == (input_dim, output_dim)
            assert np.all(np.isfinite(weights))
            
            # Check that variance adjustment worked
            activations = X_batch @ weights
            variance = np.var(activations)
            assert abs(variance - 1.0) < 0.2  # Allow some tolerance


class TestInitializationEdgeCases:
    """Test edge cases and error conditions for initialization methods."""
    
    def test_initialization_with_zero_dimensions(self):
        """Test initialization methods with zero dimensions."""
        # Zero initialization should handle edge cases
        W = initiate_zero(0, 5)
        assert W.shape == (0, 5)
        
        W = initiate_zero(5, 0)
        assert W.shape == (5, 0)
    
    def test_initialization_with_large_dimensions(self):
        """Test initialization methods with large dimensions."""
        # Test that methods can handle reasonably large dimensions
        large_dim = 1000
        
        # He initialization
        weights = initiate_he(large_dim, 100)
        assert weights.shape == (100, large_dim)
        assert np.all(np.isfinite(weights))
        
        # LeCun initialization
        weights = initiate_lecun(large_dim, 100)
        assert weights.shape == (large_dim, 100)
        assert np.all(np.isfinite(weights))
    
    def test_naive_sharding_with_extreme_k(self):
        """Test naive sharding with extreme k values."""
        X = np.random.randn(10, 3)
        
        # k = 0 (should handle gracefully)
        try:
            centroids = initiate_naive_sharding(X, 0)
            assert centroids.shape == (0, 3)
        except (ValueError, IndexError):
            pass  # Acceptable to raise error for k=0
        
        # Very large k
        centroids = initiate_naive_sharding(X, 100)
        assert centroids.shape == (100, 3)
        assert np.all(np.isfinite(centroids))
    
    def test_lsuv_with_degenerate_data(self):
        """Test LSUV with degenerate input data."""
        input_dim = 5
        output_dim = 3
        
        # All zeros
        X_batch = np.zeros((10, input_dim))
        weights = initiate_lsuv(input_dim, output_dim, X_batch, max_iter=5)
        
        assert weights.shape == (input_dim, output_dim)
        assert np.all(np.isfinite(weights))
        
        # Constant values
        X_batch = np.ones((10, input_dim)) * 5
        weights = initiate_lsuv(input_dim, output_dim, X_batch, max_iter=5)
        
        assert weights.shape == (input_dim, output_dim)
        assert np.all(np.isfinite(weights))


class TestInitializationIntegration:
    """Integration tests for initialization methods."""
    
    def test_initialization_methods_comparison(self):
        """Compare different initialization methods."""
        input_dim = 20
        output_dim = 10
        X_batch = np.random.randn(50, input_dim)
        
        # Test all methods
        he_weights = initiate_he(input_dim, output_dim)
        lecun_weights = initiate_lecun(input_dim, output_dim)
        zero_weights = initiate_zero(input_dim, output_dim)
        lsuv_weights = initiate_lsuv(input_dim, output_dim, X_batch)
        
        # All should have correct shapes
        assert he_weights.shape == (output_dim, input_dim)
        assert lecun_weights.shape == (input_dim, output_dim)
        assert zero_weights.shape == (input_dim, output_dim)
        assert lsuv_weights.shape == (input_dim, output_dim)
        
        # All should be finite
        assert np.all(np.isfinite(he_weights))
        assert np.all(np.isfinite(lecun_weights))
        assert np.all(np.isfinite(zero_weights))
        assert np.all(np.isfinite(lsuv_weights))
    
    def test_initialization_reproducibility(self):
        """Test that initialization methods are reproducible with same seed."""
        input_dim = 10
        output_dim = 5
        X_batch = np.random.randn(20, input_dim)
        
        # Test He initialization
        np.random.seed(42)
        he_weights1 = initiate_he(input_dim, output_dim)
        np.random.seed(42)
        he_weights2 = initiate_he(input_dim, output_dim)
        assert np.allclose(he_weights1, he_weights2)
        
        # Test LeCun initialization
        np.random.seed(42)
        lecun_weights1 = initiate_lecun(input_dim, output_dim)
        np.random.seed(42)
        lecun_weights2 = initiate_lecun(input_dim, output_dim)
        assert np.allclose(lecun_weights1, lecun_weights2)
        
        # Test naive sharding
        np.random.seed(42)
        centroids1 = initiate_naive_sharding(X_batch, 3)
        np.random.seed(42)
        centroids2 = initiate_naive_sharding(X_batch, 3)
        assert np.allclose(centroids1, centroids2)
    
    def test_initialization_statistical_properties(self):
        """Test statistical properties of initialization methods."""
        input_dim = 100
        output_dim = 50
        
        # Generate multiple samples for statistical testing
        n_samples = 20
        
        # He initialization
        he_samples = [initiate_he(input_dim, output_dim) for _ in range(n_samples)]
        he_means = [np.mean(sample) for sample in he_samples]
        he_vars = [np.var(sample) for sample in he_samples]
        
        # Mean should be close to 0
        assert abs(np.mean(he_means)) < 0.05
        # Variance should be close to 2/input_dim
        expected_var = 2.0 / input_dim
        assert abs(np.mean(he_vars) - expected_var) < 0.01
        
        # LeCun initialization
        lecun_samples = [initiate_lecun(input_dim, output_dim) for _ in range(n_samples)]
        lecun_means = [np.mean(sample) for sample in lecun_samples]
        lecun_vars = [np.var(sample) for sample in lecun_samples]
        
        # Mean should be close to 0
        assert abs(np.mean(lecun_means)) < 0.05
        # Variance should be close to 1/input_dim
        expected_var = 1.0 / input_dim
        assert abs(np.mean(lecun_vars) - expected_var) < 0.01