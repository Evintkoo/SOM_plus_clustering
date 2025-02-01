import unittest
import numpy as np
import cupy as cp
import os
import tempfile

# Adjust import based on your project structure
from modules.som import SOM  # This is your GPU-enabled SOM

# Dummy external functions and constants for testing purposes.
# In your project these should be replaced by the actual implementations.
def validate_configuration(initiate_method, learning_rate, distance_function):
    # For example, if learning_rate is out of (0,1), throw an error.
    if not (0 < learning_rate <= 1):
        raise ValueError("Learning rate must be in (0,1]")
    valid_inits = ["random", "kde", "kmeans", "som++", "zero", "he", "naive_sharding", "lecun", "lsuv"]
    if initiate_method not in valid_inits:
        raise ValueError("Invalid initiation method")
    if distance_function not in ["euclidean", "cosine"]:
        raise ValueError("Invalid distance function")

# Dummy evaluation metric functions.
def silhouette_score(x, labels):
    return 0.5

def davies_bouldin_index(x, labels):
    return 1.0

def calinski_harabasz_score(x, labels):
    return 100.0

def dunn_index(x, labels):
    return 0.2

EVAL_METHOD_LIST = ["silhouette", "davies_bouldin", "calinski_harabasz", "dunn"]

# If your module relies on external initialization methods (like initiate_kde),
# provide dummy implementations here for testing.
def initiate_kde(x, n_neurons):
    return np.random.rand(n_neurons, x.shape[1])

def initiate_plus_plus(m, n, x):
    return np.random.rand(m * n, x.shape[1])

def initiate_zero(P, Q):
    return np.zeros((P, Q))

def initiate_he(input_dim, output_dim):
    return np.random.randn(output_dim, input_dim)

def initiate_naive_sharding(X, k):
    idx = np.linspace(0, len(X)-1, k, dtype=int)
    return X[idx]

def initiate_lecun(input_shape, output_shape):
    return np.random.randn(output_shape, input_shape)

def initiate_lsuv(input_dim, output_dim, X_batch):
    return np.random.randn(output_dim, input_dim)

# Dummy KMeans for initialization methods.
class KMeans:
    def __init__(self, n_clusters, method):
        self.n_clusters = n_clusters
        self.method = method
        self.centroids = None

    def fit(self, x):
        self.centroids = np.random.rand(self.n_clusters, x.shape[1])


class TestSelfOrganizingMap(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters before each test."""
        cp.random.seed(42)
        # Create data as a CuPy array
        data_cpu = np.random.rand(100, 3)  # 100 samples, 3-dimensional data on CPU
        self.data = cp.asarray(data_cpu)
        self.m = 10  # Height of SOM grid
        self.n = 10  # Width of SOM grid

    def test_initialization(self):
        """Test SOM initialization with various parameters."""
        # Test valid initialization methods
        valid_methods = ["random", "kde", "kmeans", "som++", "zero"]
        for method in valid_methods:
            som = SOM(m=self.m, n=self.n, dim=3, 
                      initiate_method=method, 
                      learning_rate=0.1, 
                      neighbour_rad=5, 
                      distance_function="euclidean")
            self.assertIsNotNone(som)

    def test_invalid_initialization(self):
        """Test invalid initialization parameters."""
        # Test invalid learning rate (should be in (0,1])
        with self.assertRaises(ValueError):
            SOM(m=self.m, n=self.n, dim=3, 
                initiate_method="random", 
                learning_rate=2.0, 
                neighbour_rad=5, 
                distance_function="euclidean")

        # Test invalid initiation method
        with self.assertRaises(ValueError):
            SOM(m=self.m, n=self.n, dim=3, 
                initiate_method="invalid_method", 
                learning_rate=0.1, 
                neighbour_rad=5, 
                distance_function="euclidean")

        # Test invalid distance function
        with self.assertRaises(ValueError):
            SOM(m=self.m, n=self.n, dim=3, 
                initiate_method="random", 
                learning_rate=0.1, 
                neighbour_rad=5, 
                distance_function="invalid_distance")

    def test_neuron_initiation(self):
        """Test neuron initialization for different methods."""
        som = SOM(m=self.m, n=self.n, dim=3, 
                  initiate_method="random", 
                  learning_rate=0.1, 
                  neighbour_rad=5, 
                  distance_function="euclidean")
        
        neurons = som.initiate_neuron(cp.asnumpy(self.data))
        # Check neuron shape
        self.assertEqual(neurons.shape, (self.m, self.n, 3))
        
        # Check value ranges (for random initialization)
        neurons_cpu = cp.asnumpy(neurons)
        self.assertTrue(np.all(neurons_cpu >= self.data.min().get()))
        self.assertTrue(np.all(neurons_cpu <= self.data.max().get()))

    def test_fit_predict(self):
        """Test SOM fit and predict methods."""
        som = SOM(m=self.m, n=self.n, dim=3, 
                  initiate_method="random", 
                  learning_rate=0.1, 
                  neighbour_rad=5, 
                  distance_function="euclidean")
        
        # Fit and predict. Note: fit_predict expects CPU data (NumPy array)
        labels = som.fit_predict(x=cp.asnumpy(self.data), epoch=10)
        
        # Check labels: they should be a 1D NumPy array with length equal to the number of samples.
        self.assertEqual(len(labels), cp.asnumpy(self.data).shape[0])
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < self.m * self.n))

    def test_predict_before_fit(self):
        """Test prediction before fitting raises an error."""
        som = SOM(m=self.m, n=self.n, dim=3, 
                  initiate_method="random", 
                  learning_rate=0.1, 
                  neighbour_rad=5, 
                  distance_function="euclidean")
        
        with self.assertRaises(RuntimeError):
            som.predict(cp.asnumpy(self.data))

    def test_evaluation_methods(self):
        """Test different evaluation methods."""
        som = SOM(m=self.m, n=self.n, dim=3, 
                  initiate_method="random", 
                  learning_rate=0.1, 
                  neighbour_rad=5, 
                  distance_function="euclidean")
        
        som.fit(x=cp.asnumpy(self.data), epoch=10)
        
        # Test individual method
        silhouette = som.evaluate(x=cp.asnumpy(self.data), method=["silhouette"])
        self.assertEqual(len(silhouette), 1)
        
        # Test all methods
        all_scores = som.evaluate(x=cp.asnumpy(self.data), method=["all"])
        expected_methods = ["silhouette", "davies_bouldin", 
                             "calinski_harabasz", "dunn"]
        self.assertTrue(all(method in all_scores for method in expected_methods))

    def test_save_and_load(self):
        """Test saving and loading SOM model."""
        # Create and train a SOM
        original_som = SOM(m=self.m, n=self.n, dim=3, 
                           initiate_method="random", 
                           learning_rate=0.1, 
                           neighbour_rad=5, 
                           distance_function="euclidean")
        
        original_som.fit(x=cp.asnumpy(self.data), epoch=10)
        
        # Use a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save the model
            original_som.save(temp_path)
            
            # Load the model
            loaded_som = SOM.load(temp_path)
            
            # Compare key attributes
            self.assertEqual(original_som.m, loaded_som.m)
            self.assertEqual(original_som.n, loaded_som.n)
            self.assertEqual(original_som.dim, loaded_som.dim)
            
            # Compare neurons (convert to CPU arrays for comparison)
            np.testing.assert_array_almost_equal(
                cp.asnumpy(original_som.neurons), 
                cp.asnumpy(loaded_som.neurons)
            )
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

    def test_cosine_distance_function(self):
        """Test SOM with cosine distance function."""
        som = SOM(m=self.m, n=self.n, dim=3, 
                  initiate_method="random", 
                  learning_rate=0.1, 
                  neighbour_rad=5, 
                  distance_function="cosine")
        
        labels = som.fit_predict(x=cp.asnumpy(self.data), epoch=10)
        
        # Check labels
        self.assertEqual(len(labels), cp.asnumpy(self.data).shape[0])
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < self.m * self.n))

if __name__ == '__main__':
    unittest.main()
