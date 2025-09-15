# SOM Classification Module

## Overview

The `som_classification.py` module implements a specialized Self-Organizing Map (SOM) for supervised learning tasks. Unlike the standard unsupervised SOM, this implementation includes label information during training and can perform classification on new data points.

## Key Features

- **Supervised SOM Training**: Incorporates label information during neuron weight updates
- **Label Assignment**: Automatic neuron labeling based on closest data points
- **Parallel Processing**: Uses joblib for efficient parallel training
- **Multiple Distance Functions**: Supports both Euclidean and cosine distance metrics
- **Comprehensive Evaluation**: Includes accuracy, F1-score, and recall metrics

## Classes

### SOM

Supervised Self-Organizing Map implementation for classification tasks.

#### Constructor

```python
SOM(m: int, n: int, dim: int, initiate_method: str, learning_rate: float, 
    neighbour_rad: int, distance_function: str, max_iter: Union[int, float] = np.inf)
```

**Parameters:**
- `m` (int): Height of the SOM grid
- `n` (int): Width of the SOM grid
- `dim` (int): Input dimension of data
- `initiate_method` (str): Neuron initialization method
- `learning_rate` (float): Initial learning rate
- `neighbour_rad` (int): Initial neighbourhood radius
- `distance_function` (str): Distance function for BMU calculation ("euclidean" or "cosine")
- `max_iter` (Union[int, float]): Maximum number of iterations. Defaults to np.inf

#### Key Attributes

- `neurons` (np.ndarray): Weight matrix of shape (m, n, dim)
- `neuron_label` (np.ndarray): Label assignments for each neuron, shape (m, n)
- `_trained` (bool): Indicates if the model has been trained
- `initial_neurons` (np.ndarray): Copy of initial neuron weights
- `cur_learning_rate` (float): Current learning rate
- `cur_neighbour_rad` (float): Current neighbourhood radius

#### Methods

##### `initiate_neuron(data: np.ndarray) -> np.ndarray`

Initialize neuron weights using the specified method.

**Parameters:**
- `data` (np.ndarray): Input training data

**Returns:**
- `np.ndarray`: Initialized neuron weights

**Supported initialization methods:**
- `"random"`: Random uniform initialization within data bounds
- `"kde"`: Kernel Density Estimation initialization
- `"kmeans"`, `"kde_kmeans"`, `"kmeans++"`: K-means based initialization
- `"som++"`: SOM++ initialization using farthest-first traversal

##### `index_bmu(x: np.ndarray) -> Tuple[int, int]`

Find the index of the best matching unit (BMU) for a given input.

**Parameters:**
- `x` (np.ndarray): Input data point

**Returns:**
- `Tuple[int, int]`: (row, column) indices of the BMU

**Distance Functions:**
- **Euclidean**: `||neurons - x||₂`
- **Cosine**: `1 - (neurons · x) / (||neurons|| * ||x||)`

##### `gaussian_neighbourhood(x1: int, y1: int, x2: int, y2: int) -> float`

Calculate the Gaussian neighbourhood function between two neurons.

**Parameters:**
- `x1`, `y1` (int): Coordinates of first neuron
- `x2`, `y2` (int): Coordinates of second neuron

**Returns:**
- `float`: Neighbourhood influence strength

**Mathematical Formula:**
```
h = learning_rate * exp(-0.5 * distance² / neighbourhood_radius²)
```

##### `update_neuron(x: np.ndarray) -> None`

Update neuron weights based on input data and neighbourhood function.

**Parameters:**
- `x` (np.ndarray): Input data point

**Algorithm:**
1. Find Best Matching Unit (BMU)
2. For each neuron in the grid:
   - Calculate neighbourhood influence
   - Update weights based on distance function:
     - **Euclidean**: `neuron += h * (x - neuron)`
     - **Cosine**: `neuron += h * (cos(angle) * neuron - neuron)`

##### `fit(x: np.ndarray, y: np.ndarray, epoch: int, shuffle: bool = True, verbose: bool = True, n_jobs: int = -1) -> None`

Train the supervised SOM on labeled data.

**Parameters:**
- `x` (np.ndarray): Input training data, shape (n_samples, n_features)
- `y` (np.ndarray): Training labels, shape (n_samples,)
- `epoch` (int): Number of training epochs
- `shuffle` (bool): Whether to shuffle data each epoch. Defaults to True
- `verbose` (bool): Whether to show progress. Defaults to True
- `n_jobs` (int): Number of parallel jobs. Defaults to -1 (all cores)

**Algorithm:**
1. **Initialization**: Initialize neuron weights if not already trained
2. **Parallel Training**: Split data across multiple workers
3. **Neuron Updates**: Each worker updates neurons using subset of data
4. **Result Combination**: Average neuron weights from all workers
5. **Label Assignment**: Assign labels to neurons based on closest data points

**Label Assignment Process:**
1. Compute distances between all neurons and data points
2. Find closest data point for each neuron
3. Assign corresponding label to each neuron

##### `predict(x: np.ndarray) -> np.ndarray`

Predict class labels for new data points.

**Parameters:**
- `x` (np.ndarray): Input data for prediction

**Returns:**
- `np.ndarray`: Predicted class labels

**Algorithm:**
1. For each input point, find BMU coordinates
2. Return the label assigned to that neuron location

**Raises:**
- `RuntimeError`: If SOM hasn't been trained yet
- `AssertionError`: If input dimensions don't match training data

##### `fit_predict(x: np.ndarray, y: np.ndarray, epoch: int, shuffle: bool = True, verbose: bool = True, n_jobs: int = -1) -> np.ndarray`

Train the SOM and predict labels for the training data.

**Parameters:**
- Same as `fit()` method

**Returns:**
- `np.ndarray`: Predicted labels for training data

##### `evaluate(x: np.ndarray, y: np.ndarray, method: List[str]) -> Union[List[float], dict]`

Evaluate the classification performance using specified metrics.

**Parameters:**
- `x` (np.ndarray): Input data for evaluation
- `y` (np.ndarray): True labels
- `method` (List[str]): Evaluation methods to use

**Returns:**
- `Union[List[float], dict]`: Evaluation scores

**Available evaluation methods:**
- `"accuracy"`: Classification accuracy
- `"f1_score"`: F1 score (for binary classification)
- `"recall"`: Recall score
- `"all"`: Returns all metrics as dictionary

##### `save(path: str) -> None`

Save the trained SOM model to a file.

**Parameters:**
- `path` (str): File path for saving the model

##### `load(path: str) -> 'SOM'` (class method)

Load a previously saved SOM model.

**Parameters:**
- `path` (str): File path of the saved model

**Returns:**
- `SOM`: Loaded SOM instance

#### Properties

##### `cluster_center_`

Get the cluster centers (neuron weights) as a flattened array.

**Returns:**
- `np.ndarray`: Neuron weights reshaped to (m*n, dim)

## Utility Functions

### `validate_configuration(initiate_method: str, learning_rate: float, distance_function: str) -> None`

Validate SOM configuration parameters.

**Parameters:**
- `initiate_method` (str): Initialization method
- `learning_rate` (float): Learning rate value
- `distance_function` (str): Distance function name

**Raises:**
- `ValueError`: If any parameter is invalid

**Validation Rules:**
- Learning rate must be ≤ 1.76
- Initialization method must be in `INITIATION_METHOD_LIST`
- Distance function must be in `DISTANCE_METHOD_LIST`

### `initiate_plus_plus(m: int, n: int, x: np.ndarray) -> np.ndarray`

Initialize centroids using SOM++ algorithm (farthest-first traversal).

**Parameters:**
- `m` (int): Grid height
- `n` (int): Grid width
- `x` (np.ndarray): Input data

**Returns:**
- `np.ndarray`: Initialized centroids

**Algorithm:**
1. Start with the most edge point (farthest from dataset mean)
2. Iteratively select point farthest from already selected points
3. Continue until k = m × n points are selected

## Worker Functions

### `_worker(x: np.ndarray, epoch: int, shuffle: bool) -> np.ndarray`

Worker function for parallel processing during training.

**Parameters:**
- `x` (np.ndarray): Data subset for this worker
- `epoch` (int): Number of training epochs
- `shuffle` (bool): Whether to shuffle data

**Returns:**
- `np.ndarray`: Updated neuron weights

**Features:**
- Independent neuron weight updates for data subset
- Learning rate and neighbourhood radius decay
- Supports both Euclidean and cosine distance functions

## Usage Examples

### Basic Classification

```python
import numpy as np
from modules.som_classification import SOM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample classification data
X, y = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                          n_informative=8, random_state=42)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train SOM classifier
som = SOM(m=8, n=8, dim=10, initiate_method="kmeans++", 
          learning_rate=0.5, neighbour_rad=3, distance_function="euclidean")

# Train the model
som.fit(X_train, y_train, epoch=100, n_jobs=-1)

# Make predictions
y_pred = som.predict(X_test)

# Evaluate performance
scores = som.evaluate(X_test, y_test, method=["accuracy", "f1_score", "recall"])
print(f"Accuracy: {scores[0]:.3f}")
print(f"F1 Score: {scores[1]:.3f}")
print(f"Recall: {scores[2]:.3f}")
```

### Comprehensive Evaluation

```python
import numpy as np
from modules.som_classification import SOM
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create SOM classifier
som = SOM(m=6, n=6, dim=4, initiate_method="som++",
          learning_rate=0.3, neighbour_rad=2, distance_function="euclidean")

# Train and evaluate
som.fit(X_scaled, y, epoch=200, shuffle=True, verbose=True)

# Get comprehensive evaluation
eval_results = som.evaluate(X_scaled, y, method=["all"])
print("=== Classification Results ===")
for metric, score in eval_results.items():
    print(f"{metric.capitalize()}: {score:.4f}")

# Analyze neuron labels
print(f"\nNeuron label distribution:")
unique_labels, counts = np.unique(som.neuron_label, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} neurons")
```

### Cross-Validation Example

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from modules.som_classification import SOM

def cross_validate_som(X, y, n_splits=5):
    """Perform cross-validation on SOM classifier."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracy_scores = []
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Processing fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create and train SOM
        som = SOM(m=5, n=5, dim=X.shape[1], initiate_method="kmeans++",
                  learning_rate=0.4, neighbour_rad=2, distance_function="euclidean")
        
        som.fit(X_train, y_train, epoch=50, verbose=False)
        
        # Evaluate on validation set
        scores = som.evaluate(X_val, y_val, method=["accuracy", "f1_score"])
        accuracy_scores.append(scores[0])
        f1_scores.append(scores[1])
    
    print(f"\nCross-validation results:")
    print(f"Accuracy: {np.mean(accuracy_scores):.3f} ± {np.std(accuracy_scores):.3f}")
    print(f"F1 Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    
    return accuracy_scores, f1_scores

# Example usage
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500, n_features=8, n_classes=2, random_state=42)
accuracy_scores, f1_scores = cross_validate_som(X, y)
```

### Comparison with Different Initialization Methods

```python
import numpy as np
from modules.som_classification import SOM
from modules.variables import INITIATION_METHOD_LIST
from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

# Test different initialization methods
results = {}

for method in INITIATION_METHOD_LIST:
    try:
        print(f"Testing initialization method: {method}")
        
        som = SOM(m=6, n=6, dim=X.shape[1], initiate_method=method,
                  learning_rate=0.5, neighbour_rad=3, distance_function="euclidean")
        
        som.fit(X, y, epoch=100, verbose=False)
        scores = som.evaluate(X, y, method=["accuracy"])
        
        results[method] = scores[0]
        print(f"Accuracy: {scores[0]:.3f}")
        
    except Exception as e:
        print(f"Error with {method}: {e}")
        results[method] = 0.0

# Find best method
best_method = max(results, key=results.get)
print(f"\nBest initialization method: {best_method}")
print(f"Best accuracy: {results[best_method]:.3f}")
```

## Mathematical Background

### Neighbourhood Function

The Gaussian neighbourhood function determines the influence of the BMU on surrounding neurons:

```
h(t) = α(t) * exp(-d²ᵢⱼ / (2σ²(t)))
```

where:
- `α(t)` is the learning rate at time t
- `d²ᵢⱼ` is the squared distance between neurons i and j
- `σ(t)` is the neighbourhood radius at time t

### Learning Rate Decay

```
α(t) = α₀ * (1 - t/T) * exp(-t/α₀)
```

### Neighbourhood Radius Decay

```
σ(t) = σ₀ * (1 - t/T) * exp(-t/σ₀)
```

where:
- `α₀`, `σ₀` are initial values
- `T` is the total number of iterations
- `t` is the current iteration

## Performance Considerations

### Computational Complexity
- **Training**: O(T × N × M × N_grid) where T is epochs, N is data size, M is dimensions, N_grid is grid size
- **Prediction**: O(N × N_grid × M) for N predictions
- **Parallel Training**: Reduces wall-clock time with multiple cores

### Memory Usage
- **Neuron Storage**: O(m × n × dim) for weight matrix
- **Label Storage**: O(m × n) for neuron labels
- **Parallel Processing**: Additional memory per worker process

### Optimization Tips
1. **Grid Size**: Balance between resolution and computation time
2. **Parallel Jobs**: Use all available cores for large datasets
3. **Epochs**: More epochs improve convergence but increase training time
4. **Learning Rate**: Higher values speed convergence but may cause instability
5. **Neighbourhood Radius**: Larger radius promotes global organization

## Limitations and Considerations

1. **Binary F1 Score**: F1 score implementation assumes binary classification
2. **Label Assignment**: Simple distance-based assignment may not be optimal
3. **Memory Scaling**: Grid size quadratically affects memory usage
4. **Hyperparameter Sensitivity**: Performance sensitive to learning rate and radius
5. **Categorical Features**: Designed for continuous features, categorical data needs preprocessing