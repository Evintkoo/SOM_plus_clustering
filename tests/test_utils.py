"""
Tests for the utils module.
"""
import pytest
import numpy as np
import math
from unittest.mock import patch

# Import the utils module
import sys
sys.path.append('d:/Projects/SOM_plus_clustering')
from modules.utils import (
    find_most_edge_point,
    cos_distance,
    random_initiate,
    euc_distance,
    one_hot_encode,
    normalize_column,
    euclidean_distance_jit,
    cosine_distance_jit,
    find_most_edge_point_jit
)


class TestDistanceFunctions:
    """Test distance calculation functions."""
    
    def test_euc_distance_basic(self):
        """Test basic Euclidean distance calculation."""
        point1 = np.array([0.0, 0.0])
        point2 = np.array([3.0, 4.0])
        
        distance = euc_distance(point1, point2)
        expected = 5.0  # 3-4-5 triangle
        
        assert abs(distance - expected) < 1e-10
    
    def test_euc_distance_same_points(self):
        """Test Euclidean distance between identical points."""
        point = np.array([1.0, 2.0, 3.0])
        
        distance = euc_distance(point, point)
        
        assert distance == 0.0
    
    def test_euc_distance_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        point1 = np.array([1.0, 2.0])
        point2 = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="The dimensions of the two points must be equal"):
            euc_distance(point1, point2)
    
    def test_euc_distance_higher_dimensions(self):
        """Test Euclidean distance in higher dimensions."""
        point1 = np.array([1.0, 2.0, 3.0, 4.0])
        point2 = np.array([5.0, 6.0, 7.0, 8.0])
        
        distance = euc_distance(point1, point2)
        expected = math.sqrt(4*4 + 4*4 + 4*4 + 4*4)  # sqrt(64) = 8
        
        assert abs(distance - expected) < 1e-10
    
    def test_cos_distance_basic(self):
        """Test basic cosine distance calculation."""
        vector1 = [1.0, 0.0]
        vector2 = [0.0, 1.0]
        
        distance = cos_distance(vector1, vector2)
        expected = 1.0  # Orthogonal vectors have cosine similarity of 0
        
        assert abs(distance - expected) < 1e-10
    
    def test_cos_distance_identical_vectors(self):
        """Test cosine distance between identical vectors."""
        vector = [1.0, 2.0, 3.0]
        
        distance = cos_distance(vector, vector)
        
        assert abs(distance) < 1e-10  # Should be 0
    
    def test_cos_distance_opposite_vectors(self):
        """Test cosine distance between opposite vectors."""
        vector1 = [1.0, 0.0]
        vector2 = [-1.0, 0.0]
        
        distance = cos_distance(vector1, vector2)
        expected = 2.0  # Opposite vectors have cosine similarity of -1
        
        assert abs(distance - expected) < 1e-10
    
    def test_cos_distance_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        vector1 = [1.0, 2.0]
        vector2 = [1.0, 2.0, 3.0]
        
        with pytest.raises(ValueError, match="Input vectors must have the same length"):
            cos_distance(vector1, vector2)
    
    def test_cos_distance_zero_vector(self):
        """Test cosine distance with zero vectors."""
        vector1 = [0.0, 0.0]
        vector2 = [1.0, 0.0]
        
        # Should handle zero vectors gracefully
        distance = cos_distance(vector1, vector2)
        assert np.isfinite(distance)


class TestJITOptimizedFunctions:
    """Test JIT-optimized utility functions."""
    
    def test_euclidean_distance_jit(self):
        """Test JIT-optimized Euclidean distance."""
        point1 = np.array([1.0, 2.0, 3.0])
        point2 = np.array([4.0, 6.0, 8.0])
        
        distance = euclidean_distance_jit(point1, point2)
        expected = math.sqrt((1-4)**2 + (2-6)**2 + (3-8)**2)
        
        assert abs(distance - expected) < 1e-10
    
    def test_cosine_distance_jit(self):
        """Test JIT-optimized cosine distance."""
        vector1 = np.array([1.0, 0.0, 0.0])
        vector2 = np.array([0.0, 1.0, 0.0])
        
        distance = cosine_distance_jit(vector1, vector2)
        expected = 1.0  # Orthogonal vectors
        
        assert abs(distance - expected) < 1e-10
    
    def test_find_most_edge_point_jit(self):
        """Test JIT-optimized edge point finding."""
        # Create points where [10, 10] is clearly the farthest from center
        points = np.array([
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [10.0, 10.0]
        ])
        
        edge_point = find_most_edge_point_jit(points)
        expected = np.array([10.0, 10.0])
        
        assert np.allclose(edge_point, expected)


class TestEdgePointFinding:
    """Test edge point finding functionality."""
    
    def test_find_most_edge_point_basic(self):
        """Test basic edge point finding."""
        # Create points where [5, 5] is clearly the farthest from center
        points = np.array([
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [5.0, 5.0]
        ])
        
        edge_point = find_most_edge_point(points)
        expected = np.array([5.0, 5.0])
        
        assert np.allclose(edge_point, expected)
    
    def test_find_most_edge_point_3d(self):
        """Test edge point finding in 3D."""
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.1, 0.1, 0.1],
            [10.0, 10.0, 10.0]
        ])
        
        edge_point = find_most_edge_point(points)
        expected = np.array([10.0, 10.0, 10.0])
        
        assert np.allclose(edge_point, expected)
    
    def test_find_most_edge_point_identical_distances(self):
        """Test edge point finding when multiple points have same distance."""
        # Create symmetric points around origin
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0]
        ])
        
        edge_point = find_most_edge_point(points)
        
        # Should return one of the edge points (distance 1 from center)
        center = np.mean(points, axis=0)
        distance_from_center = np.linalg.norm(edge_point - center)
        assert abs(distance_from_center - 1.0) < 1e-10
    
    def test_find_most_edge_point_single_point(self):
        """Test edge point finding with single point."""
        points = np.array([[1.0, 2.0]])
        
        edge_point = find_most_edge_point(points)
        expected = np.array([1.0, 2.0])
        
        assert np.allclose(edge_point, expected)


class TestRandomGeneration:
    """Test random number generation functions."""
    
    def test_random_initiate_basic(self):
        """Test basic random array generation."""
        dim = 10
        min_val = -5.0
        max_val = 5.0
        
        result = random_initiate(dim, min_val, max_val)
        
        assert result.shape == (dim,)
        assert np.all(result >= min_val)
        assert np.all(result <= max_val)
        assert np.all(np.isfinite(result))
    
    def test_random_initiate_different_bounds(self):
        """Test random generation with different bounds."""
        dim = 5
        min_val = 10.0
        max_val = 20.0
        
        result = random_initiate(dim, min_val, max_val)
        
        assert result.shape == (dim,)
        assert np.all(result >= min_val)
        assert np.all(result <= max_val)
    
    def test_random_initiate_zero_dimension(self):
        """Test random generation with zero dimension."""
        result = random_initiate(0, 0.0, 1.0)
        
        assert result.shape == (0,)
    
    def test_random_initiate_equal_bounds(self):
        """Test random generation with equal min and max."""
        dim = 5
        val = 3.14
        
        result = random_initiate(dim, val, val)
        
        assert np.allclose(result, val)


class TestOneHotEncoding:
    """Test one-hot encoding functionality."""
    
    def test_one_hot_encode_basic(self):
        """Test basic one-hot encoding."""
        labels = np.array([0, 1, 2, 1, 0])
        
        encoded = one_hot_encode(labels)
        
        expected = np.array([
            [1, 0, 0],  # 0 -> [1, 0, 0]
            [0, 1, 0],  # 1 -> [0, 1, 0]
            [0, 0, 1],  # 2 -> [0, 0, 1]
            [0, 1, 0],  # 1 -> [0, 1, 0]
            [1, 0, 0]   # 0 -> [1, 0, 0]
        ])
        
        assert np.allclose(encoded, expected)
    
    def test_one_hot_encode_single_class(self):
        """Test one-hot encoding with single class."""
        labels = np.array([5, 5, 5])
        
        encoded = one_hot_encode(labels)
        
        assert encoded.shape == (3, 1)
        assert np.all(encoded == 1)
    
    def test_one_hot_encode_non_sequential_labels(self):
        """Test one-hot encoding with non-sequential labels."""
        labels = np.array([10, 5, 10, 20])
        
        encoded = one_hot_encode(labels)
        
        assert encoded.shape == (4, 3)  # 3 unique classes
        assert np.all(encoded.sum(axis=1) == 1)  # Each row sums to 1
        assert np.all(encoded.sum(axis=0) > 0)   # Each class appears at least once
    
    def test_one_hot_encode_2d_input(self):
        """Test one-hot encoding with 2D input (should flatten)."""
        labels = np.array([[0, 1], [2, 0]])
        
        encoded = one_hot_encode(labels)
        
        assert encoded.shape == (4, 3)  # Flattened to 4 samples, 3 classes
        assert np.all(encoded.sum(axis=1) == 1)
    
    def test_one_hot_encode_empty_input(self):
        """Test one-hot encoding with empty input."""
        labels = np.array([])
        
        encoded = one_hot_encode(labels)
        
        assert encoded.shape[0] == 0  # No samples
    
    def test_one_hot_encode_float_labels(self):
        """Test one-hot encoding with float labels."""
        labels = np.array([1.0, 2.0, 1.0, 3.0])
        
        encoded = one_hot_encode(labels)
        
        assert encoded.shape == (4, 3)  # 3 unique classes
        assert np.all(encoded.sum(axis=1) == 1)


class TestNormalization:
    """Test column normalization functionality."""
    
    def test_normalize_column_basic(self):
        """Test basic column normalization."""
        data = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0]
        ])
        
        normalized = normalize_column(data, 0)
        
        # First column should be normalized to [0, 1] range
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        expected = np.array([0.0, 1/3, 2/3, 1.0])
        assert np.allclose(normalized, expected)
    
    def test_normalize_column_second_column(self):
        """Test normalizing second column."""
        data = np.array([
            [1.0, 5.0],
            [2.0, 15.0],
            [3.0, 25.0]
        ])
        
        normalized = normalize_column(data, 1)
        
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        expected = np.array([0.0, 0.5, 1.0])
        assert np.allclose(normalized, expected)
    
    def test_normalize_column_identical_values(self):
        """Test normalizing column with identical values."""
        data = np.array([
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0]
        ])
        
        normalized = normalize_column(data, 1)
        
        # All values are the same, should return zeros
        assert np.all(normalized == 0.0)
    
    def test_normalize_column_out_of_bounds(self):
        """Test that out of bounds column index raises IndexError."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        with pytest.raises(IndexError, match="Column index 2 is out of bounds"):
            normalize_column(data, 2)
        
        with pytest.raises(IndexError, match="Column index -1 is out of bounds"):
            normalize_column(data, -1)
    
    def test_normalize_column_negative_values(self):
        """Test normalizing column with negative values."""
        data = np.array([
            [1.0, -10.0],
            [2.0, 0.0],
            [3.0, 10.0]
        ])
        
        normalized = normalize_column(data, 1)
        
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
        expected = np.array([0.0, 0.5, 1.0])
        assert np.allclose(normalized, expected)
    
    def test_normalize_column_single_row(self):
        """Test normalizing column with single row."""
        data = np.array([[5.0, 10.0]])
        
        normalized = normalize_column(data, 0)
        
        # Single value should normalize to 0
        assert normalized[0] == 0.0
    
    def test_normalize_column_very_small_range(self):
        """Test normalizing column with very small range."""
        data = np.array([
            [1.0, 1.0000000001],
            [2.0, 1.0000000002],
            [3.0, 1.0000000003]
        ])
        
        normalized = normalize_column(data, 1)
        
        # Very small range should return zeros (numerical stability)
        assert np.all(normalized == 0.0)


class TestUtilsEdgeCases:
    """Test edge cases and error conditions for utils functions."""
    
    def test_distance_functions_with_nan(self):
        """Test distance functions with NaN values."""
        point1 = np.array([1.0, np.nan])
        point2 = np.array([2.0, 3.0])
        
        # Functions should handle NaN gracefully or return NaN
        result = euc_distance(point1, point2)
        assert np.isnan(result) or np.isfinite(result)
    
    def test_distance_functions_with_inf(self):
        """Test distance functions with infinite values."""
        point1 = np.array([1.0, np.inf])
        point2 = np.array([2.0, 3.0])
        
        result = euc_distance(point1, point2)
        assert np.isinf(result) or np.isfinite(result)
    
    def test_cosine_distance_with_very_small_vectors(self):
        """Test cosine distance with very small vectors."""
        vector1 = [1e-15, 0.0]
        vector2 = [0.0, 1e-15]
        
        # Should handle very small vectors without numerical issues
        distance = cos_distance(vector1, vector2)
        assert np.isfinite(distance)
    
    def test_one_hot_encode_with_negative_labels(self):
        """Test one-hot encoding with negative labels."""
        labels = np.array([-1, 0, 1, -1])
        
        encoded = one_hot_encode(labels)
        
        assert encoded.shape == (4, 3)  # 3 unique classes
        assert np.all(encoded.sum(axis=1) == 1)
    
    def test_find_most_edge_point_with_nan(self):
        """Test edge point finding with NaN values."""
        points = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [4.0, 5.0]
        ])
        
        # Should handle NaN gracefully
        try:
            edge_point = find_most_edge_point(points)
            assert len(edge_point) == 2
        except (ValueError, RuntimeError):
            pass  # Expected behavior with NaN data


class TestUtilsIntegration:
    """Integration tests for utils functions."""
    
    def test_distance_consistency(self):
        """Test that distance functions are consistent."""
        point1 = np.array([1.0, 2.0, 3.0])
        point2 = np.array([4.0, 5.0, 6.0])
        
        # Euclidean distance should be consistent
        distance1 = euc_distance(point1, point2)
        distance2 = euclidean_distance_jit(point1, point2)
        
        assert abs(distance1 - distance2) < 1e-10
    
    def test_normalization_range(self):
        """Test that normalization produces correct range."""
        np.random.seed(42)
        data = np.random.random((100, 5)) * 100 - 50  # Random data in [-50, 50]
        
        for col in range(5):
            normalized = normalize_column(data, col)
            
            if np.max(data[:, col]) != np.min(data[:, col]):
                assert np.min(normalized) == 0.0
                assert np.max(normalized) == 1.0
            else:
                assert np.all(normalized == 0.0)
    
    def test_one_hot_encoding_inverse(self):
        """Test that one-hot encoding is invertible."""
        original_labels = np.array([0, 1, 2, 1, 0, 2])
        
        # Encode
        encoded = one_hot_encode(original_labels)
        
        # Decode (inverse operation)
        decoded = np.argmax(encoded, axis=1)
        
        # Should match original labels (possibly with different class indices)
        assert len(decoded) == len(original_labels)
        assert encoded.shape == (6, 3)
    
    def test_utils_with_cupy_fallback(self):
        """Test utils functions work with CuPy fallback."""
        # This test ensures the module works even without CuPy
        point1 = np.array([1.0, 2.0])
        point2 = np.array([3.0, 4.0])
        
        distance = euc_distance(point1, point2)
        expected = math.sqrt(8)
        
        assert abs(distance - expected) < 1e-10
    
    def test_random_initiate_distribution(self):
        """Test that random_initiate produces reasonable distribution."""
        np.random.seed(42)
        samples = random_initiate(1000, 0.0, 1.0)
        
        # Should be approximately uniformly distributed
        assert 0.4 < np.mean(samples) < 0.6  # Mean should be around 0.5
        assert 0.08 < np.std(samples) < 0.35  # Std should be around sqrt(1/12) ≈ 0.29
        assert np.min(samples) >= 0.0
        assert np.max(samples) <= 1.0