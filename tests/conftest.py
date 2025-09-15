"""
Common fixtures and utilities for testing.
"""
import pytest
import numpy as np
import tempfile
import os
from typing import Tuple, Any


@pytest.fixture
def sample_data_2d() -> np.ndarray:
    """Generate sample 2D data for testing."""
    np.random.seed(42)
    # Create three clusters
    cluster1 = np.random.normal([0, 0], 0.5, (30, 2))
    cluster2 = np.random.normal([3, 3], 0.5, (30, 2))
    cluster3 = np.random.normal([0, 3], 0.5, (30, 2))
    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def sample_data_3d() -> np.ndarray:
    """Generate sample 3D data for testing."""
    np.random.seed(42)
    # Create two clusters in 3D space
    cluster1 = np.random.normal([0, 0, 0], 0.8, (40, 3))
    cluster2 = np.random.normal([4, 4, 4], 0.8, (40, 3))
    return np.vstack([cluster1, cluster2])


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Generate sample labels for classification testing."""
    # Labels for 90 samples (matching sample_data_2d)
    return np.array([0] * 30 + [1] * 30 + [2] * 30)


@pytest.fixture
def binary_labels() -> np.ndarray:
    """Generate binary labels for classification testing."""
    # Binary labels for 80 samples (matching sample_data_3d)
    return np.array([0] * 40 + [1] * 40)


@pytest.fixture
def small_dataset() -> np.ndarray:
    """Generate a small dataset for quick testing."""
    np.random.seed(42)
    return np.random.random((10, 2))


@pytest.fixture
def temp_model_file():
    """Create a temporary file for model saving/loading tests."""
    fd, path = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def som_params() -> dict:
    """Default SOM parameters for testing."""
    return {
        'm': 3,
        'n': 3,
        'dim': 2,
        'initiate_method': 'random',
        'learning_rate': 0.5,
        'neighbour_rad': 1,
        'distance_function': 'euclidean'
    }


@pytest.fixture
def kmeans_params() -> dict:
    """Default KMeans parameters for testing."""
    return {
        'n_clusters': 3,
        'method': 'random',
        'tol': 1e-6,
        'max_iters': 100
    }


class TestDataHelper:
    """Helper class for generating test data."""
    
    @staticmethod
    def create_linearly_separable_data(n_samples: int = 100, n_features: int = 2, 
                                     n_classes: int = 2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Create linearly separable data for classification tests."""
        np.random.seed(random_state)
        
        # Generate random centroids for each class
        centroids = np.random.uniform(-5, 5, (n_classes, n_features))
        
        X = []
        y = []
        
        samples_per_class = n_samples // n_classes
        
        for i in range(n_classes):
            # Generate samples around each centroid
            class_samples = np.random.normal(
                centroids[i], 
                1.0, 
                (samples_per_class, n_features)
            )
            X.append(class_samples)
            y.extend([i] * samples_per_class)
        
        X = np.vstack(X)
        y = np.array(y)
        
        return X, y
    
    @staticmethod
    def create_noisy_data(base_data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add noise to existing data."""
        noise = np.random.normal(0, noise_level, base_data.shape)
        return base_data + noise


@pytest.fixture
def test_data_helper():
    """Provide access to test data helper methods."""
    return TestDataHelper