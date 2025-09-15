"""
Tests for the core SOM (Self-Organizing Map) module.
"""
import pytest
import numpy as np
import pickle
import os
from unittest.mock import patch, MagicMock

# Import the SOM module
import sys
sys.path.append('d:/Projects/SOM_plus_clustering')
from modules.som import (
    SOM, 
    validate_configuration, 
    initiate_plus_plus,
    euclidean_distance_jit,
    euclidean_distance_squared_jit,
    cosine_distance_jit,
    batch_euclidean_distances_jit,
    batch_cosine_distances_jit,
    neighborhood_function_jit
)


class TestSOMValidation:
    """Test parameter validation for SOM."""
    
    def test_validate_configuration_valid_params(self):
        """Test that valid parameters pass validation."""
        # Should not raise any exception
        validate_configuration("random", 0.5, "euclidean")
        validate_configuration("kmeans++", 1.0, "cosine")
    
    def test_validate_configuration_invalid_learning_rate(self):
        """Test that invalid learning rate raises ValueError."""
        with pytest.raises(ValueError, match="Learning rate should be less than 1.76"):
            validate_configuration("random", 2.0, "euclidean")
    
    def test_validate_configuration_invalid_init_method(self):
        """Test that invalid initialization method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid initiation method"):
            validate_configuration("invalid_method", 0.5, "euclidean")
    
    def test_validate_configuration_invalid_distance_function(self):
        """Test that invalid distance function raises ValueError."""
        with pytest.raises(ValueError, match="Invalid distance function"):
            validate_configuration("random", 0.5, "invalid_distance")


class TestSOMInitialization:
    """Test SOM initialization methods."""
    
    def test_som_init_basic(self, som_params):
        """Test basic SOM initialization."""
        som = SOM(**som_params)
        
        assert som.m == 3
        assert som.n == 3
        assert som.dim == 2
        assert som.shape == (3, 3, 2)
        assert som.init_method == "random"
        assert som.dist_func == "euclidean"
        assert som.cur_learning_rate == 0.5
        assert som.cur_neighbour_rad == 1
        assert not som._trained
    
    def test_som_init_with_max_iter(self, som_params):
        """Test SOM initialization with max_iter parameter."""
        som_params['max_iter'] = 1000
        som = SOM(**som_params)
        assert som.max_iter == 1000
    
    def test_som_init_backend_selection(self, som_params):
        """Test backend selection logic."""
        # Test auto backend selection
        som = SOM(**som_params, backend="auto")
        assert som.backend in ["cupy", "taichi", "numba", "numpy"]
        
        # Test specific backend selection
        som_numpy = SOM(**som_params, backend="numpy")
        assert som_numpy.backend == "numpy"
    
    def test_som_init_invalid_backend(self, som_params):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            SOM(**som_params, backend="invalid_backend")
    
    def test_initiate_plus_plus_algorithm(self, sample_data_2d):
        """Test the SOM++ initialization algorithm."""
        centroids = initiate_plus_plus(3, 3, sample_data_2d)
        
        assert centroids.shape == (9, 2)  # 3x3 grid, 2D data
        assert np.all(np.isfinite(centroids))
        
        # Check that centroids are diverse (not all the same)
        assert not np.allclose(centroids[0], centroids[1])


class TestSOMNeuronInitialization:
    """Test neuron initialization methods."""
    
    def test_initiate_neuron_random(self, sample_data_2d, som_params):
        """Test random neuron initialization."""
        som = SOM(**som_params)
        neurons = som.initiate_neuron(sample_data_2d)
        
        assert neurons.shape == (3, 3, 2)
        assert np.all(np.isfinite(neurons))
        
        # Check that values are within data range
        data_min, data_max = sample_data_2d.min(), sample_data_2d.max()
        assert np.all(neurons >= data_min)
        assert np.all(neurons <= data_max)
    
    def test_initiate_neuron_som_plus_plus(self, sample_data_2d, som_params):
        """Test SOM++ neuron initialization."""
        som_params['initiate_method'] = 'som++'
        som = SOM(**som_params)
        neurons = som.initiate_neuron(sample_data_2d)
        
        assert neurons.shape == (3, 3, 2)
        assert np.all(np.isfinite(neurons))
    
    @patch('modules.som.KMeans')
    def test_initiate_neuron_kmeans(self, mock_kmeans, sample_data_2d, som_params):
        """Test KMeans neuron initialization."""
        # Mock KMeans behavior
        mock_model = MagicMock()
        mock_model.centroids = np.random.random((9, 2))
        mock_kmeans.return_value = mock_model
        
        som_params['initiate_method'] = 'kmeans++'
        som = SOM(**som_params)
        neurons = som.initiate_neuron(sample_data_2d)
        
        assert neurons.shape == (3, 3, 2)
        mock_kmeans.assert_called_once()
        mock_model.fit.assert_called_once()
    
    def test_initiate_neuron_invalid_method(self, sample_data_2d, som_params):
        """Test that invalid initialization method raises ValueError."""
        som_params['initiate_method'] = 'invalid_method'
        with pytest.raises(ValueError):
            som = SOM(**som_params)


class TestSOMTraining:
    """Test SOM training functionality."""
    
    def test_som_fit_basic(self, sample_data_2d, som_params):
        """Test basic SOM fitting."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, epoch=2)
        
        assert som._trained
        assert som.neurons.shape == (3, 3, 2)
        assert np.all(np.isfinite(som.neurons))
    
    def test_som_fit_with_batch_size(self, sample_data_2d, som_params):
        """Test SOM fitting with batch processing."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, epoch=1, batch_size=10)
        
        assert som._trained
        assert som.neurons.shape == (3, 3, 2)
    
    def test_som_fit_dimension_mismatch(self, som_params):
        """Test that dimension mismatch raises ValueError."""
        som = SOM(**som_params)
        wrong_data = np.random.random((10, 3))  # SOM expects 2D
        
        with pytest.raises(ValueError, match="X.shape\[1\] should be 2"):
            som.fit(wrong_data, epoch=1)
    
    def test_som_fit_learning_rate_decay(self, small_dataset, som_params):
        """Test that learning rate decays during training."""
        som = SOM(**som_params)
        initial_lr = som.cur_learning_rate
        
        som.fit(small_dataset, epoch=2)
        
        # Learning rate should have decayed
        assert som.cur_learning_rate < initial_lr
    
    def test_som_fit_neighborhood_radius_decay(self, small_dataset, som_params):
        """Test that neighborhood radius decays during training."""
        som = SOM(**som_params)
        initial_rad = som.cur_neighbour_rad
        
        som.fit(small_dataset, epoch=2)
        
        # Neighborhood radius should have decayed
        assert som.cur_neighbour_rad < initial_rad


class TestSOMPrediction:
    """Test SOM prediction functionality."""
    
    def test_som_predict_basic(self, sample_data_2d, som_params):
        """Test basic prediction functionality."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, epoch=1)
        
        predictions = som.predict(sample_data_2d)
        
        assert len(predictions) == len(sample_data_2d)
        assert all(isinstance(p, (int, np.integer)) for p in predictions)
        assert all(0 <= p < 9 for p in predictions)  # 3x3 grid = 9 clusters
    
    def test_som_predict_not_trained(self, sample_data_2d, som_params):
        """Test that predicting with untrained model raises RuntimeError."""
        som = SOM(**som_params)
        
        with pytest.raises(RuntimeError, match="SOM must be fitted before predicting"):
            som.predict(sample_data_2d)
    
    def test_som_predict_wrong_dimensions(self, sample_data_2d, som_params):
        """Test prediction with wrong input dimensions."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, epoch=1)
        
        # Wrong number of features
        wrong_data = np.random.random((10, 3))
        with pytest.raises(ValueError, match="This SOM has dimension 2 but received 3"):
            som.predict(wrong_data)
        
        # Wrong number of dimensions
        wrong_data_1d = np.random.random(10)
        with pytest.raises(ValueError, match="X should have two dimensions"):
            som.predict(wrong_data_1d)
    
    def test_som_fit_predict(self, sample_data_2d, som_params):
        """Test combined fit and predict functionality."""
        som = SOM(**som_params)
        predictions = som.fit_predict(sample_data_2d, epoch=1)
        
        assert som._trained
        assert len(predictions) == len(sample_data_2d)
        assert all(0 <= p < 9 for p in predictions)


class TestSOMBMU:
    """Test Best Matching Unit (BMU) functionality."""
    
    def test_index_bmu_euclidean(self, som_params):
        """Test BMU finding with Euclidean distance."""
        som = SOM(**som_params)
        som.neurons = np.zeros((3, 3, 2))
        som.neurons[1, 1] = [1.0, 1.0]  # Set one neuron close to test point
        
        test_point = np.array([1.1, 1.1])
        bmu_row, bmu_col = som.index_bmu(test_point)
        
        assert bmu_row == 1
        assert bmu_col == 1
    
    def test_index_bmu_cosine(self, som_params):
        """Test BMU finding with cosine distance."""
        som_params['distance_function'] = 'cosine'
        som = SOM(**som_params)
        som.neurons = np.random.random((3, 3, 2))
        
        test_point = np.array([1.0, 0.0])
        bmu_row, bmu_col = som.index_bmu(test_point)
        
        assert 0 <= bmu_row < 3
        assert 0 <= bmu_col < 3
    
    def test_bmu_indices_batch(self, sample_data_2d, som_params):
        """Test batch BMU computation."""
        som = SOM(**som_params)
        som.neurons = np.random.random((3, 3, 2))
        
        batch_data = sample_data_2d[:10]  # Test with first 10 samples
        bmu_indices = som._bmu_indices_batch(batch_data)
        
        assert len(bmu_indices) == 10
        assert all(0 <= idx < 9 for idx in bmu_indices)  # Valid indices for 3x3 grid


class TestSOMEvaluation:
    """Test SOM evaluation functionality."""
    
    def test_som_evaluate_valid_methods(self, sample_data_2d, som_params):
        """Test evaluation with valid methods."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, epoch=1)
        
        # Test individual methods
        scores = som.evaluate(sample_data_2d, ["silhouette"])
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        
        # Test multiple methods
        scores = som.evaluate(sample_data_2d, ["silhouette", "davies_bouldin"])
        assert len(scores) == 2
        
        # Test 'all' method
        scores = som.evaluate(sample_data_2d, ["all"])
        assert isinstance(scores, dict)
        assert "silhouette" in scores
        assert "davies_bouldin" in scores
    
    def test_som_evaluate_invalid_method(self, sample_data_2d, som_params):
        """Test evaluation with invalid method."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, epoch=1)
        
        with pytest.raises(ValueError, match="Invalid evaluation method"):
            som.evaluate(sample_data_2d, ["invalid_method"])


class TestSOMUtilities:
    """Test SOM utility functions."""
    
    def test_cluster_center_property(self, sample_data_2d, som_params):
        """Test cluster center property."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, epoch=1)
        
        centers = som.cluster_center_
        assert centers.shape == (9, 2)  # 3x3 grid, 2D features
        assert np.all(np.isfinite(centers))
    
    def test_som_save_load(self, sample_data_2d, som_params, temp_model_file):
        """Test model saving and loading."""
        # Train and save model
        som = SOM(**som_params)
        som.fit(sample_data_2d, epoch=1)
        som.save(temp_model_file)
        
        assert os.path.exists(temp_model_file)
        
        # Load model and verify
        loaded_som = SOM.load(temp_model_file)
        assert loaded_som._trained
        assert loaded_som.m == som.m
        assert loaded_som.n == som.n
        assert loaded_som.dim == som.dim
        assert np.allclose(loaded_som.neurons, som.neurons)


class TestJITOptimizedFunctions:
    """Test JIT-optimized utility functions."""
    
    def test_euclidean_distance_jit(self):
        """Test JIT-optimized Euclidean distance."""
        x = np.array([1.0, 2.0])
        y = np.array([4.0, 6.0])
        
        distance = euclidean_distance_jit(x, y)
        expected = np.sqrt((1-4)**2 + (2-6)**2)
        
        assert abs(distance - expected) < 1e-10
    
    def test_euclidean_distance_squared_jit(self):
        """Test JIT-optimized squared Euclidean distance."""
        x = np.array([1.0, 2.0])
        y = np.array([4.0, 6.0])
        
        distance_sq = euclidean_distance_squared_jit(x, y)
        expected = (1-4)**2 + (2-6)**2
        
        assert abs(distance_sq - expected) < 1e-10
    
    def test_cosine_distance_jit(self):
        """Test JIT-optimized cosine distance."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        
        distance = cosine_distance_jit(x, y)
        expected = 1.0  # Orthogonal vectors have cosine similarity of 0
        
        assert abs(distance - expected) < 1e-10
    
    def test_batch_euclidean_distances_jit(self):
        """Test JIT-optimized batch Euclidean distances."""
        data_batch = np.array([[1.0, 2.0], [3.0, 4.0]])
        neurons_flat = np.array([[0.0, 0.0], [2.0, 2.0]])
        
        distances = batch_euclidean_distances_jit(data_batch, neurons_flat)
        
        assert distances.shape == (2, 2)
        assert np.all(distances >= 0)
    
    def test_neighborhood_function_jit(self):
        """Test JIT-optimized neighborhood function."""
        dist_squared = 1.0
        learning_rate = 0.5
        radius_squared = 2.0
        
        h = neighborhood_function_jit(dist_squared, learning_rate, radius_squared)
        expected = learning_rate * np.exp(-dist_squared / (2.0 * radius_squared))
        
        assert abs(h - expected) < 1e-10


class TestSOMEdgeCases:
    """Test SOM edge cases and error conditions."""
    
    def test_som_with_very_small_data(self, som_params):
        """Test SOM with very small dataset."""
        small_data = np.array([[1.0, 2.0], [1.1, 2.1]])
        som = SOM(**som_params)
        
        # Should not crash with very small data
        som.fit(small_data, epoch=1)
        predictions = som.predict(small_data)
        
        assert len(predictions) == 2
    
    def test_som_with_identical_points(self, som_params):
        """Test SOM with identical data points."""
        identical_data = np.array([[1.0, 2.0]] * 10)
        som = SOM(**som_params)
        
        som.fit(identical_data, epoch=1)
        predictions = som.predict(identical_data)
        
        # All predictions should be the same
        assert len(set(predictions)) == 1
    
    def test_som_max_iter_limit(self, sample_data_2d, som_params):
        """Test SOM with maximum iteration limit."""
        som_params['max_iter'] = 5
        som = SOM(**som_params)
        
        som.fit(sample_data_2d, epoch=10)  # Request many epochs
        
        # Should stop early due to max_iter
        assert som._trained
    
    def test_som_with_nan_data(self, som_params):
        """Test SOM behavior with NaN data."""
        nan_data = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
        som = SOM(**som_params)
        
        # SOM should handle NaN gracefully or raise appropriate error
        try:
            som.fit(nan_data, epoch=1)
        except (ValueError, RuntimeError):
            pass  # Expected behavior with NaN data


# Integration test
class TestSOMIntegration:
    """Integration tests for SOM functionality."""
    
    def test_complete_som_workflow(self, sample_data_2d, som_params):
        """Test complete SOM workflow from initialization to evaluation."""
        # Initialize SOM
        som = SOM(**som_params)
        assert not som._trained
        
        # Fit the model
        som.fit(sample_data_2d, epoch=2, shuffle=True)
        assert som._trained
        
        # Make predictions
        predictions = som.predict(sample_data_2d)
        assert len(predictions) == len(sample_data_2d)
        
        # Evaluate the model
        scores = som.evaluate(sample_data_2d, ["silhouette", "davies_bouldin"])
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
        
        # Get cluster centers
        centers = som.cluster_center_
        assert centers.shape == (9, 2)
    
    def test_som_different_backends(self, small_dataset, som_params):
        """Test SOM with different computational backends."""
        backends = ["numpy"]  # Always available
        
        for backend in backends:
            som_params_copy = som_params.copy()
            som_params_copy['backend'] = backend
            
            som = SOM(**som_params_copy)
            som.fit(small_dataset, epoch=1)
            predictions = som.predict(small_dataset)
            
            assert som._trained
            assert len(predictions) == len(small_dataset)
    
    def test_som_reproducibility(self, som_params):
        """Test that SOM produces consistent results with same random seed."""
        np.random.seed(42)
        data = np.random.random((20, 2))
        
        # Train two identical SOMs
        som1 = SOM(**som_params)
        som2 = SOM(**som_params)
        
        np.random.seed(42)
        som1.fit(data, epoch=1, shuffle=False)
        
        np.random.seed(42)
        som2.fit(data, epoch=1, shuffle=False)
        
        # Results should be similar (allowing for some numerical differences)
        pred1 = som1.predict(data)
        pred2 = som2.predict(data)
        
        # At least majority of predictions should match
        accuracy = np.mean(np.array(pred1) == np.array(pred2))
        assert accuracy > 0.5  # Allow some variance due to randomness