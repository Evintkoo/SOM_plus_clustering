"""
Tests for the SOM classification module.
"""
import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock

# Import the SOM classification module
import sys
sys.path.append('d:/Projects/SOM_plus_clustering')
from modules.som_classification import (
    SOM,
    validate_configuration,
    initiate_plus_plus
)


class TestSOMClassificationValidation:
    """Test parameter validation for SOM classification."""
    
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


class TestSOMClassificationInitialization:
    """Test SOM classification initialization methods."""
    
    def test_som_init_basic(self, som_params):
        """Test basic SOM classification initialization."""
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
    
    def test_initiate_plus_plus_classification(self, sample_data_2d):
        """Test the SOM++ initialization algorithm for classification."""
        centroids = initiate_plus_plus(3, 3, sample_data_2d)
        
        assert centroids.shape == (9, 2)  # 3x3 grid, 2D data
        assert np.all(np.isfinite(centroids))
        
        # Check that centroids are diverse (not all the same)
        assert not np.allclose(centroids[0], centroids[1])


class TestSOMClassificationNeuronInitialization:
    """Test neuron initialization methods for classification."""
    
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
    
    @patch('modules.som_classification.KMeans')
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
        
        with pytest.raises(ValueError, match="Invalid initiation method"):
            som = SOM(**som_params)


class TestSOMClassificationTraining:
    """Test SOM classification training functionality."""
    
    def test_som_fit_basic(self, sample_data_2d, sample_labels, som_params):
        """Test basic SOM classification fitting."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, sample_labels, epoch=2)
        
        assert som._trained
        assert som.neurons.shape == (3, 3, 2)
        assert som.neuron_label.shape == (3, 3)
        assert np.all(np.isfinite(som.neurons))
    
    def test_som_fit_dimension_mismatch(self, sample_labels, som_params):
        """Test that dimension mismatch raises ValueError."""
        som = SOM(**som_params)
        wrong_data = np.random.random((10, 3))  # SOM expects 2D
        
        with pytest.raises(ValueError, match="X.shape\\[1\\] should be 2"):
            som.fit(wrong_data, sample_labels, epoch=1)
    
    def test_som_fit_learning_rate_decay(self, small_dataset, som_params):
        """Test that learning rate decays during training."""
        som = SOM(**som_params)
        initial_lr = som.cur_learning_rate
        
        # Create dummy labels for small dataset
        labels = np.random.randint(0, 3, len(small_dataset))
        som.fit(small_dataset, labels, epoch=2)
        
        # Learning rate should have decayed
        assert som.cur_learning_rate < initial_lr
    
    def test_som_fit_neighborhood_radius_decay(self, small_dataset, som_params):
        """Test that neighborhood radius decays during training."""
        som = SOM(**som_params)
        initial_rad = som.cur_neighbour_rad
        
        # Create dummy labels for small dataset
        labels = np.random.randint(0, 3, len(small_dataset))
        som.fit(small_dataset, labels, epoch=2)
        
        # Neighborhood radius should have decayed
        assert som.cur_neighbour_rad < initial_rad


class TestSOMClassificationBMU:
    """Test Best Matching Unit (BMU) functionality for classification."""
    
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


class TestSOMClassificationPrediction:
    """Test SOM classification prediction functionality."""
    
    def test_som_predict_basic(self, sample_data_2d, sample_labels, som_params):
        """Test basic classification prediction functionality."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, sample_labels, epoch=1, verbose=False)
        
        predictions = som.predict(sample_data_2d)
        
        assert len(predictions) == len(sample_data_2d)
        # Predictions should be from the original label set
        unique_labels = set(sample_labels)
        for pred in predictions:
            assert pred in unique_labels
    
    def test_som_predict_not_trained(self, sample_data_2d, som_params):
        """Test that predicting with untrained model raises RuntimeError."""
        som = SOM(**som_params)
        
        with pytest.raises(RuntimeError, match="SOM must be fitted before predicting"):
            som.predict(sample_data_2d)
    
    def test_som_predict_wrong_dimensions(self, sample_data_2d, sample_labels, som_params):
        """Test prediction with wrong input dimensions."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, sample_labels, epoch=1, verbose=False)
        
        # Wrong number of features
        wrong_data = np.random.random((10, 3))
        with pytest.raises(AssertionError):
            som.predict(wrong_data)
        
        # Wrong number of dimensions
        wrong_data_1d = np.random.random(10)
        with pytest.raises(AssertionError):
            som.predict(wrong_data_1d)
    
    def test_som_fit_predict(self, sample_data_2d, sample_labels, som_params):
        """Test combined fit and predict functionality."""
        som = SOM(**som_params)
        predictions = som.fit_predict(sample_data_2d, sample_labels, epoch=1, verbose=False)
        
        assert som._trained
        assert len(predictions) == len(sample_data_2d)
        
        # Predictions should be from the original label set
        unique_labels = set(sample_labels)
        for pred in predictions:
            assert pred in unique_labels


class TestSOMClassificationEvaluation:
    """Test SOM classification evaluation functionality."""
    
    def test_som_evaluate_valid_methods(self, sample_data_2d, sample_labels, som_params):
        """Test evaluation with valid methods."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, sample_labels, epoch=1, verbose=False)
        
        # Test individual methods
        scores = som.evaluate(sample_data_2d, sample_labels, ["accuracy"])
        assert len(scores) == 1
        assert isinstance(scores[0], float)
        assert 0 <= scores[0] <= 100  # Accuracy should be between 0 and 100
        
        # Test multiple methods
        scores = som.evaluate(sample_data_2d, sample_labels, ["accuracy", "f1_score"])
        assert len(scores) == 2
        
        # Test 'all' method
        scores = som.evaluate(sample_data_2d, sample_labels, ["all"])
        assert isinstance(scores, dict)
        assert "accuracy" in scores
        assert "f1_score" in scores
        assert "recall" in scores
    
    def test_som_evaluate_invalid_method(self, sample_data_2d, sample_labels, som_params):
        """Test evaluation with invalid method."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, sample_labels, epoch=1, verbose=False)
        
        with pytest.raises(ValueError, match="Invalid evaluation method"):
            som.evaluate(sample_data_2d, sample_labels, ["invalid_method"])


class TestSOMClassificationNeighborhood:
    """Test neighborhood function for classification."""
    
    def test_gaussian_neighbourhood(self, som_params):
        """Test Gaussian neighborhood function."""
        som = SOM(**som_params)
        
        # Test neighborhood at same position (should be maximum)
        h_same = som.gaussian_neighbourhood(1, 1, 1, 1)
        assert h_same == som.cur_learning_rate
        
        # Test neighborhood at distance (should be less)
        h_dist = som.gaussian_neighbourhood(1, 1, 2, 2)
        assert 0 < h_dist < som.cur_learning_rate
        
        # Test neighborhood at far distance (should be very small)
        h_far = som.gaussian_neighbourhood(0, 0, 2, 2)
        assert 0 <= h_far < h_dist


class TestSOMClassificationNeuronUpdate:
    """Test neuron update functionality for classification."""
    
    def test_update_neuron_euclidean(self, som_params):
        """Test neuron update with Euclidean distance."""
        som = SOM(**som_params)
        som.neurons = np.ones((3, 3, 2))  # Initialize all neurons to [1, 1]
        
        # Test point that should pull neurons towards [2, 2]
        test_point = np.array([2.0, 2.0])
        initial_neurons = som.neurons.copy()
        
        som.update_neuron(test_point)
        
        # Neurons should have moved towards the test point
        assert not np.allclose(som.neurons, initial_neurons)
    
    def test_update_neuron_cosine(self, som_params):
        """Test neuron update with cosine distance."""
        som_params['distance_function'] = 'cosine'
        som = SOM(**som_params)
        som.neurons = np.ones((3, 3, 2))
        
        test_point = np.array([2.0, 1.0])
        initial_neurons = som.neurons.copy()
        
        som.update_neuron(test_point)
        
        # Neurons should have been updated
        assert not np.allclose(som.neurons, initial_neurons)


class TestSOMClassificationUtilities:
    """Test SOM classification utility functions."""
    
    def test_cluster_center_property(self, sample_data_2d, sample_labels, som_params):
        """Test cluster center property."""
        som = SOM(**som_params)
        som.fit(sample_data_2d, sample_labels, epoch=1, verbose=False)
        
        centers = som.cluster_center_
        assert centers.shape == (9, 2)  # 3x3 grid, 2D features
        assert np.all(np.isfinite(centers))
    
    def test_som_save_load(self, sample_data_2d, sample_labels, som_params, temp_model_file):
        """Test model saving and loading."""
        # Train and save model
        som = SOM(**som_params)
        som.fit(sample_data_2d, sample_labels, epoch=1, verbose=False)
        som.save(temp_model_file)
        
        assert os.path.exists(temp_model_file)
        
        # Load model and verify
        loaded_som = SOM.load(temp_model_file)
        assert loaded_som._trained
        assert loaded_som.m == som.m
        assert loaded_som.n == som.n
        assert loaded_som.dim == som.dim
        assert np.allclose(loaded_som.neurons, som.neurons)
        assert np.array_equal(loaded_som.neuron_label, som.neuron_label)


class TestSOMClassificationWorkerFunction:
    """Test worker function for parallel processing."""
    
    def test_worker_function(self, small_dataset, som_params):
        """Test the worker function for parallel processing."""
        som = SOM(**som_params)
        som.neurons = np.random.random((3, 3, 2))
        
        # Create dummy labels
        labels = np.random.randint(0, 3, len(small_dataset))
        
        # Test worker function
        result_neurons = som._worker(small_dataset, epoch=1, shuffle=False)
        
        assert result_neurons.shape == (3, 3, 2)
        assert np.all(np.isfinite(result_neurons))


class TestSOMClassificationEdgeCases:
    """Test SOM classification edge cases and error conditions."""
    
    def test_som_with_binary_classification(self, test_data_helper):
        """Test SOM with binary classification."""
        # Create binary classification data
        X, y = test_data_helper.create_linearly_separable_data(
            n_samples=40, n_features=2, n_classes=2, random_state=42
        )
        
        som_params = {
            'm': 2, 'n': 2, 'dim': 2, 'initiate_method': 'random',
            'learning_rate': 0.5, 'neighbour_rad': 1, 'distance_function': 'euclidean'
        }
        
        som = SOM(**som_params)
        som.fit(X, y, epoch=1, verbose=False)
        predictions = som.predict(X)
        
        assert len(predictions) == len(X)
        # Should only predict classes 0 and 1
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_som_with_multiclass_classification(self, test_data_helper):
        """Test SOM with multiclass classification."""
        # Create multiclass classification data
        X, y = test_data_helper.create_linearly_separable_data(
            n_samples=60, n_features=2, n_classes=3, random_state=42
        )
        
        som_params = {
            'm': 3, 'n': 3, 'dim': 2, 'initiate_method': 'random',
            'learning_rate': 0.5, 'neighbour_rad': 1, 'distance_function': 'euclidean'
        }
        
        som = SOM(**som_params)
        som.fit(X, y, epoch=1, verbose=False)
        predictions = som.predict(X)
        
        assert len(predictions) == len(X)
        # Should only predict classes 0, 1, and 2
        assert all(pred in [0, 1, 2] for pred in predictions)
    
    def test_som_with_single_class(self, som_params):
        """Test SOM with single class data."""
        # All samples have the same label
        data = np.random.random((20, 2))
        labels = np.zeros(20, dtype=int)
        
        som = SOM(**som_params)
        som.fit(data, labels, epoch=1, verbose=False)
        predictions = som.predict(data)
        
        # All predictions should be 0
        assert all(pred == 0 for pred in predictions)


# Integration test
class TestSOMClassificationIntegration:
    """Integration tests for SOM classification functionality."""
    
    def test_complete_som_classification_workflow(self, sample_data_2d, sample_labels, som_params):
        """Test complete SOM classification workflow."""
        # Initialize SOM
        som = SOM(**som_params)
        assert not som._trained
        
        # Fit the model
        som.fit(sample_data_2d, sample_labels, epoch=2, verbose=False)
        assert som._trained
        
        # Make predictions
        predictions = som.predict(sample_data_2d)
        assert len(predictions) == len(sample_data_2d)
        
        # Evaluate the model
        scores = som.evaluate(sample_data_2d, sample_labels, ["accuracy", "f1_score"])
        assert len(scores) == 2
        assert all(isinstance(score, float) for score in scores)
        
        # Get cluster centers
        centers = som.cluster_center_
        assert centers.shape == (9, 2)
    
    def test_som_classification_performance(self, test_data_helper):
        """Test SOM classification performance on well-separated data."""
        # Create well-separated data for better classification performance
        X, y = test_data_helper.create_linearly_separable_data(
            n_samples=60, n_features=2, n_classes=3, random_state=42
        )
        
        som_params = {
            'm': 4, 'n': 4, 'dim': 2, 'initiate_method': 'random',
            'learning_rate': 0.3, 'neighbour_rad': 2, 'distance_function': 'euclidean'
        }
        
        som = SOM(**som_params)
        som.fit(X, y, epoch=5, verbose=False)
        predictions = som.predict(X)
        
        # Calculate accuracy manually
        correct = sum(1 for true, pred in zip(y, predictions) if true == pred)
        accuracy = (correct / len(y)) * 100
        
        # Should achieve reasonable accuracy on well-separated data
        assert accuracy > 30  # Allowing for some randomness in initialization
    
    def test_som_classification_reproducibility(self, som_params):
        """Test that SOM classification produces consistent results with same random seed."""
        np.random.seed(42)
        data = np.random.random((20, 2))
        labels = np.random.randint(0, 3, 20)
        
        # Train two identical SOMs
        som1 = SOM(**som_params)
        som2 = SOM(**som_params)
        
        np.random.seed(42)
        som1.fit(data, labels, epoch=1, shuffle=False, verbose=False, n_jobs=1)
        
        np.random.seed(42)
        som2.fit(data, labels, epoch=1, shuffle=False, verbose=False, n_jobs=1)
        
        # Results should be similar (allowing for some numerical differences)
        pred1 = som1.predict(data)
        pred2 = som2.predict(data)
        
        # At least majority of predictions should match
        accuracy = np.mean(np.array(pred1) == np.array(pred2))
        assert accuracy > 0.7  # Allow some variance due to randomness