# Model Picker Module

## Overview

The `model_picker.py` module provides functionality for evaluating and selecting the best performing SOM model across different initialization methods. It automates the process of testing multiple initialization strategies and identifies the optimal configuration based on evaluation metrics.

## Classes

### model_picker

A utility class for comparing SOM models with different initialization methods and selecting the best performing one.

#### Constructor

```python
model_picker()
```

Creates a new model picker instance with empty model storage.

#### Attributes

- `models` (List): List storing trained SOM model instances
- `model_evaluation` (List): List storing evaluation scores corresponding to each model

#### Methods

##### `pick_best_model() -> SOM`

Selects and returns the best performing model based on evaluation scores.

**Returns:**
- `SOM`: The SOM model instance with the highest evaluation score

**Algorithm:**
1. Finds the index of the maximum evaluation score using `np.argmax`
2. Returns the corresponding model from the models list

**Note:**
- Assumes higher evaluation scores indicate better performance
- Returns the first model if multiple models have the same maximum score

##### `evaluate_initiate_method(X: np.array, m: int, n: int, learning_rate: float, neighbor_rad: int, distance_function: str, max_iter: int = None, epoch: int = 1) -> None`

Evaluates SOM performance across all available initialization methods.

**Parameters:**
- `X` (np.ndarray): Input training data
- `m` (int): Height of the SOM grid
- `n` (int): Width of the SOM grid  
- `learning_rate` (float): Initial learning rate for training
- `neighbor_rad` (int): Initial neighborhood radius
- `distance_function` (str): Distance function to use ("euclidean" or "cosine")
- `max_iter` (int, optional): Maximum number of iterations. Defaults to None
- `epoch` (int): Number of training epochs. Defaults to 1

**Algorithm:**
1. Iterates through all initialization methods in `INITIATION_METHOD_LIST`
2. For each method:
   - Creates a new SOM instance with specified parameters
   - Trains the model using `fit()` method
   - Evaluates the model using `evaluate()` method
   - Stores both model and evaluation score
3. All results are stored in class attributes for later comparison

**Initialization Methods Tested:**
- `"random"`: Random uniform initialization
- `"kde"`: Kernel Density Estimation initialization
- `"kmeans"`: K-means clustering initialization
- `"kmeans++"`: K-means++ initialization
- `"som++"`: SOM++ initialization
- `"zero"`: Zero initialization
- `"he"`: He initialization
- `"naive_sharding"`: Naive sharding initialization
- `"lecun"`: LeCun initialization
- `"lsuv"`: Layer-wise Sequential Unit-Variance initialization

## Dependencies

The module imports from:
- `numpy`: Numerical operations
- `math`: Mathematical functions
- `.som`: SOM class implementation
- `.utils`: Utility functions (euc_distance)
- `.variables`: Configuration constants (INITIATION_METHOD_LIST)

## Usage Examples

### Basic Model Selection

```python
import numpy as np
from modules.model_picker import model_picker

# Generate sample data
X = np.random.rand(1000, 5)

# Create model picker instance
picker = model_picker()

# Evaluate all initialization methods
picker.evaluate_initiate_method(
    X=X,
    m=10,
    n=10, 
    learning_rate=0.5,
    neighbor_rad=3,
    distance_function="euclidean",
    epoch=50
)

# Select best model
best_model = picker.pick_best_model()
print(f"Best model uses: {best_model.init_method}")
print(f"Best score: {max(picker.model_evaluation)}")
```

### Comprehensive Evaluation with Multiple Configurations

```python
import numpy as np
from modules.model_picker import model_picker

# Generate clustered sample data
np.random.seed(42)
cluster1 = np.random.normal([2, 2], 0.5, (300, 2))
cluster2 = np.random.normal([-2, -2], 0.5, (300, 2)) 
cluster3 = np.random.normal([2, -2], 0.5, (300, 2))
X = np.vstack([cluster1, cluster2, cluster3])

# Test different SOM configurations
configurations = [
    {"m": 8, "n": 8, "learning_rate": 0.3, "neighbor_rad": 2},
    {"m": 10, "n": 10, "learning_rate": 0.5, "neighbor_rad": 3},
    {"m": 12, "n": 12, "learning_rate": 0.7, "neighbor_rad": 4}
]

best_models = []
best_scores = []

for i, config in enumerate(configurations):
    print(f"Testing configuration {i+1}: {config}")
    
    picker = model_picker()
    picker.evaluate_initiate_method(
        X=X,
        distance_function="euclidean",
        epoch=100,
        **config
    )
    
    best_model = picker.pick_best_model()
    best_score = max(picker.model_evaluation)
    
    best_models.append(best_model)
    best_scores.append(best_score)
    
    print(f"Best method: {best_model.init_method}, Score: {best_score:.4f}")

# Find overall best configuration
overall_best_idx = np.argmax(best_scores)
overall_best_model = best_models[overall_best_idx]
print(f"\nOverall best: Configuration {overall_best_idx+1}")
print(f"Method: {overall_best_model.init_method}")
print(f"Score: {best_scores[overall_best_idx]:.4f}")
```

### Detailed Analysis of Results

```python
import numpy as np
import matplotlib.pyplot as plt
from modules.model_picker import model_picker
from modules.variables import INITIATION_METHOD_LIST

# Generate data
X = np.random.rand(500, 3)

# Evaluate models
picker = model_picker()
picker.evaluate_initiate_method(
    X=X,
    m=8,
    n=8,
    learning_rate=0.4,
    neighbor_rad=2,
    distance_function="euclidean",
    epoch=75
)

# Analyze results
print("=== Model Evaluation Results ===")
for i, (method, score) in enumerate(zip(INITIATION_METHOD_LIST, picker.model_evaluation)):
    print(f"{method:15s}: {score:.6f}")

# Find best and worst
best_idx = np.argmax(picker.model_evaluation)
worst_idx = np.argmin(picker.model_evaluation)

print(f"\nBest method:  {INITIATION_METHOD_LIST[best_idx]} ({picker.model_evaluation[best_idx]:.6f})")
print(f"Worst method: {INITIATION_METHOD_LIST[worst_idx]} ({picker.model_evaluation[worst_idx]:.6f})")

# Get best model for further analysis
best_model = picker.pick_best_model()
print(f"\nBest model details:")
print(f"  Grid size: {best_model.m} x {best_model.n}")
print(f"  Dimensions: {best_model.dim}")
print(f"  Final learning rate: {best_model.cur_learning_rate:.6f}")
print(f"  Final neighbor radius: {best_model.cur_neighbour_rad:.6f}")
```

### Cross-Validation with Model Picker

```python
import numpy as np
from sklearn.model_selection import KFold
from modules.model_picker import model_picker

def cross_validate_som_init(X, n_splits=5):
    """Cross-validate SOM initialization methods."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for each method
    method_scores = {method: [] for method in INITIATION_METHOD_LIST}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Processing fold {fold + 1}/{n_splits}")
        
        X_train = X[train_idx]
        X_val = X[val_idx]
        
        picker = model_picker()
        picker.evaluate_initiate_method(
            X=X_train,
            m=6,
            n=6,
            learning_rate=0.5,
            neighbor_rad=2,
            distance_function="euclidean",
            epoch=30
        )
        
        # Evaluate each model on validation set
        for i, model in enumerate(picker.models):
            method = INITIATION_METHOD_LIST[i]
            val_score = model.evaluate()  # Implement validation scoring
            method_scores[method].append(val_score)
    
    # Calculate average scores
    avg_scores = {}
    for method, scores in method_scores.items():
        avg_scores[method] = np.mean(scores)
        print(f"{method:15s}: {avg_scores[method]:.6f} ± {np.std(scores):.6f}")
    
    # Return best method
    best_method = max(avg_scores, key=avg_scores.get)
    return best_method, avg_scores

# Example usage
X = np.random.rand(1000, 4)
best_init_method, scores = cross_validate_som_init(X)
print(f"\nBest initialization method: {best_init_method}")
```

## Implementation Notes

### Current Limitations

1. **Single Evaluation Metric**: Uses only one evaluation metric from `model.evaluate()`
2. **No Parameter Tuning**: Fixed SOM parameters for all initialization methods
3. **Memory Usage**: Stores all trained models in memory
4. **No Parallel Processing**: Sequential evaluation of initialization methods

### Evaluation Method

The `evaluate()` method called on each SOM model should return a numerical score where higher values indicate better performance. The specific metric depends on the SOM implementation but typically includes clustering quality measures like:

- Silhouette coefficient
- Davies-Bouldin index
- Calinski-Harabasz index
- Custom distance-based metrics

### Best Practices

1. **Sufficient Training**: Use adequate number of epochs for fair comparison
2. **Consistent Parameters**: Keep SOM parameters constant across initialization methods
3. **Representative Data**: Ensure training data is representative of the problem domain
4. **Multiple Runs**: Consider averaging results over multiple random seeds
5. **Validation Split**: Use separate validation data for unbiased evaluation

### Extension Opportunities

1. **Multi-metric Evaluation**: Support multiple evaluation criteria
2. **Parameter Grid Search**: Automatic tuning of SOM hyperparameters
3. **Parallel Evaluation**: Speed up evaluation using multiprocessing
4. **Cross-validation**: Built-in k-fold cross-validation support
5. **Statistical Testing**: Significance tests for method comparison
6. **Visualization**: Plotting and comparison tools for results analysis

## Performance Considerations

- **Training Time**: Varies significantly across initialization methods
- **Memory Usage**: Linear in number of methods and model complexity
- **Reproducibility**: Set random seeds for consistent results
- **Scalability**: Consider computational cost for large datasets or many methods