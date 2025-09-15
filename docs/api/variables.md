# Variables Module

## Overview

The `variables.py` module defines configuration constants and lists used throughout the SOM clustering package. These variables provide centralized configuration for initialization methods, distance functions, and evaluation metrics.

## Constants

### `INITIATION_METHOD_LIST`

List of available initialization methods for SOM and clustering algorithms.

```python
INITIATION_METHOD_LIST = [
    "random", "kde", "kmeans", "kmeans++", "som++", 
    "zero", "he", "naive_sharding", "lecun", "lsuv"
]
```

**Available Methods:**

#### Neural Network Initialization Methods

- **`"random"`**: Random uniform initialization within data bounds
  - Simple and fast
  - Good baseline method
  - May lead to suboptimal convergence

- **`"zero"`**: Zero initialization with identity/Hadamard matrices
  - Structured initialization for specific layer configurations
  - Uses identity matrices for equal dimensions
  - Applies Hadamard transformation for different dimensions

- **`"he"`**: He initialization for ReLU activations
  - Variance: `Var(W) = 2/n_in`
  - Optimal for ReLU and variants
  - Prevents vanishing/exploding gradients

- **`"lecun"`**: LeCun initialization for symmetric activations
  - Variance: `Var(W) = 1/n_in`
  - Suitable for tanh and sigmoid activations
  - Maintains signal propagation

- **`"lsuv"`**: Layer-wise Sequential Unit-Variance initialization
  - Iteratively normalizes activation variance to 1
  - Combines orthogonal initialization with variance control
  - More computationally expensive but precise

#### Clustering-Based Initialization Methods

- **`"kde"`**: Kernel Density Estimation initialization
  - Selects neurons at high-density regions
  - Uses local maxima detection
  - Good for data with clear modes

- **`"kmeans"`**: Standard K-means clustering initialization
  - Uses K-means centroids as initial neurons
  - Fast and effective for spherical clusters
  - May converge to local optima

- **`"kmeans++"`**: K-means++ smart initialization
  - Probabilistic selection for better initial centroids
  - Reduces likelihood of poor clustering
  - Generally superior to random K-means initialization

- **`"som++"`**: SOM++ farthest-first traversal
  - Selects neurons with maximum spacing
  - Ensures good coverage of input space
  - Similar to K-means++ but for SOM

- **`"naive_sharding"`**: Naive sharding method
  - Sorts data by feature sums and partitions
  - Simple but effective for well-distributed data
  - Computationally efficient

### `DISTANCE_METHOD_LIST`

List of available distance functions for BMU (Best Matching Unit) calculation.

```python
DISTANCE_METHOD_LIST = ["euclidean", "cosine"]
```

**Available Distance Functions:**

#### `"euclidean"`
- **Formula**: `||x - y||₂ = √(Σᵢ(xᵢ - yᵢ)²)`
- **Properties**: 
  - Measures straight-line distance in feature space
  - Sensitive to feature magnitude and scale
  - Assumes features are independent and equally important
- **Use Cases**:
  - Continuous features with similar scales
  - When preserving geometric relationships is important
  - Default choice for most clustering tasks

#### `"cosine"`
- **Formula**: `1 - (x·y)/(||x||·||y||)`
- **Properties**:
  - Measures angle between vectors (orientation similarity)
  - Invariant to vector magnitude/scale
  - Values in range [0, 2], where 0 = identical direction
- **Use Cases**:
  - High-dimensional sparse data
  - Text/document similarity
  - When vector direction matters more than magnitude

### `EVAL_METHOD_LIST`

List of available unsupervised clustering evaluation metrics.

```python
EVAL_METHOD_LIST = [
    "davies_bouldin", "silhouette", "calinski_harabasz", 
    "dunn", "bcubed_recall", "bcubed_precision", "all"
]
```

**Available Evaluation Metrics:**

#### Internal Validation Metrics

- **`"silhouette"`**: Silhouette Coefficient
  - **Range**: [-1, 1], higher is better
  - **Measures**: Both cohesion (within-cluster) and separation (between-cluster)
  - **Interpretation**: Values near +1 indicate well-separated clusters

- **`"davies_bouldin"`**: Davies-Bouldin Index
  - **Range**: [0, ∞), lower is better
  - **Measures**: Average similarity ratio of each cluster with its most similar cluster
  - **Interpretation**: Lower values indicate better clustering

- **`"calinski_harabasz"`**: Calinski-Harabasz Index (Variance Ratio Criterion)
  - **Range**: [0, ∞), higher is better
  - **Measures**: Ratio of between-cluster to within-cluster variance
  - **Interpretation**: Higher values indicate better defined clusters

- **`"dunn"`**: Dunn Index
  - **Range**: [0, ∞), higher is better
  - **Measures**: Ratio of minimum inter-cluster distance to maximum intra-cluster distance
  - **Interpretation**: Higher values indicate better separation and compactness

#### External Validation Metrics (require ground truth)

- **`"bcubed_precision"`**: BCubed Precision
  - **Range**: [0, 1], higher is better
  - **Measures**: Precision of cluster assignments compared to ground truth
  - **Usage**: When true labels are available

- **`"bcubed_recall"`**: BCubed Recall
  - **Range**: [0, 1], higher is better
  - **Measures**: Recall of cluster assignments compared to ground truth
  - **Usage**: When true labels are available

#### Special Options

- **`"all"`**: Compute all available metrics
  - Returns results as a dictionary
  - Useful for comprehensive evaluation
  - May include only applicable metrics based on context

### `CLASSIFICATION_EVAL_METHOD_LIST`

List of available supervised classification evaluation metrics.

```python
CLASSIFICATION_EVAL_METHOD_LIST = ["accuracy", "f1_score", "recall", "all"]
```

**Available Classification Metrics:**

#### `"accuracy"`
- **Formula**: `(TP + TN) / (TP + TN + FP + FN)`
- **Range**: [0, 1], higher is better
- **Interpretation**: Fraction of correct predictions
- **Use Cases**: Balanced datasets, overall performance assessment

#### `"f1_score"`
- **Formula**: `2 × (Precision × Recall) / (Precision + Recall)`
- **Range**: [0, 1], higher is better
- **Interpretation**: Harmonic mean of precision and recall
- **Use Cases**: Imbalanced datasets, when both precision and recall matter

#### `"recall"`
- **Formula**: `TP / (TP + FN)`
- **Range**: [0, 1], higher is better
- **Interpretation**: Fraction of actual positives correctly identified
- **Use Cases**: When missing positive cases is costly

#### `"all"`
- Returns all classification metrics as a dictionary
- Provides comprehensive performance overview

## Usage Examples

### Method Validation

```python
from modules.variables import (
    INITIATION_METHOD_LIST, DISTANCE_METHOD_LIST, 
    EVAL_METHOD_LIST, CLASSIFICATION_EVAL_METHOD_LIST
)

def validate_som_parameters(init_method, distance_function, eval_methods):
    """Validate SOM parameters against available options."""
    
    # Validate initialization method
    if init_method not in INITIATION_METHOD_LIST:
        raise ValueError(f"Invalid initialization method: {init_method}")
    
    # Validate distance function
    if distance_function not in DISTANCE_METHOD_LIST:
        raise ValueError(f"Invalid distance function: {distance_function}")
    
    # Validate evaluation methods
    if not all(method in EVAL_METHOD_LIST for method in eval_methods):
        invalid = set(eval_methods) - set(EVAL_METHOD_LIST)
        raise ValueError(f"Invalid evaluation methods: {invalid}")
    
    print("All parameters are valid!")

# Example usage
try:
    validate_som_parameters("kmeans++", "euclidean", ["silhouette", "davies_bouldin"])
    print("✓ Parameters validated successfully")
except ValueError as e:
    print(f"✗ Validation error: {e}")
```

### Configuration Generator

```python
from modules.variables import INITIATION_METHOD_LIST, DISTANCE_METHOD_LIST
import itertools

def generate_som_configurations():
    """Generate all possible SOM configurations."""
    
    grid_sizes = [(5, 5), (8, 8), (10, 10)]
    learning_rates = [0.3, 0.5, 0.7]
    neighbor_radii = [2, 3, 4]
    
    configurations = []
    
    for (m, n), init_method, distance_func, lr, radius in itertools.product(
        grid_sizes, INITIATION_METHOD_LIST, DISTANCE_METHOD_LIST, 
        learning_rates, neighbor_radii
    ):
        config = {
            'm': m, 'n': n,
            'initiate_method': init_method,
            'distance_function': distance_func,
            'learning_rate': lr,
            'neighbour_rad': radius
        }
        configurations.append(config)
    
    return configurations

# Generate configurations
configs = generate_som_configurations()
print(f"Generated {len(configs)} different SOM configurations")

# Show first few configurations
for i, config in enumerate(configs[:3]):
    print(f"Config {i+1}: {config}")
```

### Method Comparison Framework

```python
from modules.variables import INITIATION_METHOD_LIST, EVAL_METHOD_LIST
import numpy as np

class MethodComparison:
    """Framework for comparing different initialization methods."""
    
    def __init__(self):
        self.results = {}
    
    def compare_initialization_methods(self, X, som_params):
        """Compare all initialization methods."""
        
        for method in INITIATION_METHOD_LIST:
            print(f"Testing {method} initialization...")
            
            try:
                # Create SOM with specific initialization method
                som_params['initiate_method'] = method
                # som = SOM(**som_params)  # Would create SOM instance
                # som.fit(X, epoch=100)
                # scores = som.evaluate(X, method=["silhouette", "davies_bouldin"])
                
                # Placeholder for actual results
                scores = np.random.rand(2)  # Mock scores
                
                self.results[method] = {
                    'silhouette': scores[0],
                    'davies_bouldin': scores[1]
                }
                
            except Exception as e:
                print(f"Error with {method}: {e}")
                self.results[method] = {'error': str(e)}
    
    def print_results(self):
        """Print comparison results."""
        print("\n=== Initialization Method Comparison ===")
        
        for method, scores in self.results.items():
            if 'error' in scores:
                print(f"{method:15s}: ERROR - {scores['error']}")
            else:
                print(f"{method:15s}: Silhouette={scores['silhouette']:.3f}, "
                      f"Davies-Bouldin={scores['davies_bouldin']:.3f}")
    
    def get_best_method(self, metric='silhouette'):
        """Get best performing method for a specific metric."""
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            return None
        
        if metric == 'silhouette':
            # Higher is better for silhouette
            best_method = max(valid_results, key=lambda x: valid_results[x][metric])
        elif metric == 'davies_bouldin':
            # Lower is better for Davies-Bouldin
            best_method = min(valid_results, key=lambda x: valid_results[x][metric])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_method, valid_results[best_method][metric]

# Example usage
comparison = MethodComparison()
X = np.random.rand(500, 10)  # Sample data
som_params = {
    'm': 8, 'n': 8, 'dim': 10,
    'learning_rate': 0.5, 'neighbour_rad': 3,
    'distance_function': 'euclidean'
}

comparison.compare_initialization_methods(X, som_params)
comparison.print_results()

best_method, best_score = comparison.get_best_method('silhouette')
print(f"\nBest method for silhouette: {best_method} (score: {best_score:.3f})")
```

### Evaluation Metrics Selection

```python
from modules.variables import EVAL_METHOD_LIST, CLASSIFICATION_EVAL_METHOD_LIST

def select_evaluation_metrics(task_type, has_ground_truth=False):
    """Select appropriate evaluation metrics based on task type."""
    
    if task_type == "clustering":
        if has_ground_truth:
            # Use both internal and external metrics
            metrics = ["silhouette", "davies_bouldin", "bcubed_precision", "bcubed_recall"]
        else:
            # Use only internal metrics
            metrics = ["silhouette", "davies_bouldin", "calinski_harabasz", "dunn"]
    
    elif task_type == "classification":
        # Use classification metrics
        metrics = ["accuracy", "f1_score", "recall"]
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Validate metrics are available
    if task_type == "clustering":
        available_metrics = EVAL_METHOD_LIST
    else:
        available_metrics = CLASSIFICATION_EVAL_METHOD_LIST
    
    invalid_metrics = set(metrics) - set(available_metrics)
    if invalid_metrics:
        raise ValueError(f"Invalid metrics for {task_type}: {invalid_metrics}")
    
    return metrics

# Example usage
clustering_metrics = select_evaluation_metrics("clustering", has_ground_truth=False)
print(f"Clustering metrics (no ground truth): {clustering_metrics}")

supervised_clustering_metrics = select_evaluation_metrics("clustering", has_ground_truth=True)
print(f"Clustering metrics (with ground truth): {supervised_clustering_metrics}")

classification_metrics = select_evaluation_metrics("classification")
print(f"Classification metrics: {classification_metrics}")
```

## Best Practices

### Initialization Method Selection

1. **For general clustering**: Start with `"kmeans++"` or `"som++"`
2. **For high-dimensional data**: Consider `"he"` or `"lecun"`
3. **For specific data distributions**: Use `"kde"` for multimodal data
4. **For reproducible results**: Use `"naive_sharding"` or deterministic methods

### Distance Function Selection

1. **Euclidean distance**: Default for most numerical data
2. **Cosine distance**: Use for:
   - High-dimensional sparse data
   - Text/document analysis
   - When magnitude is less important than direction

### Evaluation Metrics Selection

1. **Single metric**: Use `"silhouette"` for general clustering evaluation
2. **Multiple metrics**: Combine internal metrics for robust evaluation
3. **With ground truth**: Include external metrics like BCubed precision/recall
4. **Comparative analysis**: Use `"all"` for comprehensive assessment

## Module Integration

These variables are used throughout the package:

- **SOM classes**: Parameter validation and configuration
- **Model picker**: Method iteration and comparison
- **Evaluation modules**: Metric selection and validation
- **Utility functions**: Configuration checking and setup

## Extension Guidelines

When adding new methods or metrics:

1. **Add to appropriate list**: Append new options to relevant constant lists
2. **Update documentation**: Document new methods with usage guidelines
3. **Implement validation**: Ensure new methods work with existing validation logic
4. **Add tests**: Include test cases for new functionality
5. **Update examples**: Provide usage examples for new options