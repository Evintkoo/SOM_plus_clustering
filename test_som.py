import unittest
import numpy as np
import os
import tempfile

from modules.som import SOM  # Adjust import based on your project structure

class TestSelfOrganizingMap(unittest.TestCase):
    def setUp(self):
        """Set up test data and parameters before each test."""
        np.random.seed(42)
        self.data = np.random.rand(100, 3)  # 100 samples, 3-dimensional data
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
        # Test invalid learning rate
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
        
        neurons = som.initiate_neuron(self.data)
        
        # Check neuron shape
        self.assertEqual(neurons.shape, (self.m, self.n, 3))
        
        # Check value ranges (for random initialization)
        self.assertTrue(np.all(neurons >= self.data.min()))
        self.assertTrue(np.all(neurons <= self.data.max()))

    def test_fit_predict(self):
        """Test SOM fit and predict methods."""
        som = SOM(m=self.m, n=self.n, dim=3, 
                  initiate_method="random", 
                  learning_rate=0.1, 
                  neighbour_rad=5, 
                  distance_function="euclidean")
        
        # Fit and predict
        labels = som.fit_predict(x=self.data, epoch=10)
        
        # Check labels
        self.assertEqual(len(labels), len(self.data))
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
            som.predict(self.data)

    def test_evaluation_methods(self):
        """Test different evaluation methods."""
        som = SOM(m=self.m, n=self.n, dim=3, 
                  initiate_method="random", 
                  learning_rate=0.1, 
                  neighbour_rad=5, 
                  distance_function="euclidean")
        
        som.fit(x=self.data, epoch=10)
        
        # Test individual method
        silhouette = som.evaluate(x=self.data, method=["silhouette"])
        self.assertEqual(len(silhouette), 1)
        
        # Test all methods
        all_scores = som.evaluate(x=self.data, method=["all"])
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
        
        original_som.fit(x=self.data, epoch=10)
        
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
            
            # Compare neurons
            np.testing.assert_array_almost_equal(
                original_som.neurons, 
                loaded_som.neurons
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
        
        labels = som.fit_predict(x=self.data, epoch=10)
        
        # Check labels
        self.assertEqual(len(labels), len(self.data))
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels < self.m * self.n))

if __name__ == '__main__':
    unittest.main()