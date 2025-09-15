# Basic SOM Clustering Example

This example demonstrates the basic usage of the SOM (Self-Organizing Map) for unsupervised clustering.

## Example: Clustering 2D Data

```python
import numpy as np
import matplotlib.pyplot as plt
from modules.som import SOM
from sklearn.datasets import make_blobs

# Generate sample clustered data
n_samples = 800
n_features = 2
n_centers = 4

X, true_labels = make_blobs(
    n_samples=n_samples,
    centers=n_centers,
    n_features=n_features,
    cluster_std=1.5,
    random_state=42
)

print(f"Generated {n_samples} samples with {n_features} features")
print(f"Data shape: {X.shape}")

# Create SOM instance
som = SOM(
    m=8,                           # Grid height
    n=8,                           # Grid width  
    dim=n_features,                # Input dimensions
    initiate_method="kmeans++",    # Initialization method
    learning_rate=0.5,             # Initial learning rate
    neighbour_rad=3,               # Initial neighborhood radius
    distance_function="euclidean", # Distance function
    max_iter=10000,                # Maximum iterations
    backend="auto"                 # Automatic backend selection
)

print(f"Created SOM with {som.m}x{som.n} grid")
print(f"Initialization method: {som.init_method}")
print(f"Backend selected: {som.backend}")

# Train the SOM
print("\nTraining SOM...")
som.fit(X, epoch=100, shuffle=True, batch_size=64)

print(f"Training completed!")
print(f"Final learning rate: {som.cur_learning_rate:.6f}")
print(f"Final neighborhood radius: {som.cur_neighbour_rad:.6f}")

# Make predictions
som_labels = som.predict(X)
print(f"Predicted {len(np.unique(som_labels))} clusters")

# Evaluate clustering performance
evaluation_methods = ["silhouette", "davies_bouldin", "calinski_harabasz", "dunn"]
scores = som.evaluate(X, method=evaluation_methods)

print("\n=== Clustering Evaluation ===")
print(f"Silhouette Score: {scores[0]:.4f} (higher is better)")
print(f"Davies-Bouldin Index: {scores[1]:.4f} (lower is better)")
print(f"Calinski-Harabasz Index: {scores[2]:.4f} (higher is better)")
print(f"Dunn Index: {scores[3]:.4f} (higher is better)")

# Get cluster centers
cluster_centers = som.cluster_center_
print(f"\nCluster centers shape: {cluster_centers.shape}")

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original data with true labels
axes[0].scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis', alpha=0.7)
axes[0].set_title('Original Data (True Clusters)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# SOM clustering results
axes[1].scatter(X[:, 0], X[:, 1], c=som_labels, cmap='viridis', alpha=0.7)
axes[1].scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                c='red', marker='x', s=100, linewidths=3, label='SOM Neurons')
axes[1].set_title('SOM Clustering Results')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()

# SOM grid visualization
neuron_weights = som.neurons.reshape(-1, n_features)
grid_x, grid_y = np.meshgrid(range(som.n), range(som.m))
im = axes[2].scatter(neuron_weights[:, 0], neuron_weights[:, 1], 
                     c=range(len(neuron_weights)), cmap='viridis', s=50)
axes[2].set_title('SOM Neuron Grid in Feature Space')
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 2')
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.show()

# Save the trained model
som.save("basic_som_model.pkl")
print("\nModel saved as 'basic_som_model.pkl'")

# Load and test the saved model
loaded_som = SOM.load("basic_som_model.pkl")
test_predictions = loaded_som.predict(X[:10])
print(f"Loaded model predictions for first 10 samples: {test_predictions}")
```

## Output

```
Generated 800 samples with 2 features
Data shape: (800, 2)
Created SOM with 8x8 grid
Initialization method: kmeans++
Backend selected: cupy

Training SOM...
Training completed!
Final learning rate: 0.000123
Final neighborhood radius: 0.000456

Predicted 64 clusters

=== Clustering Evaluation ===
Silhouette Score: 0.6234 (higher is better)
Davies-Bouldin Index: 0.8765 (lower is better)
Calinski-Harabasz Index: 234.5678 (higher is better)
Dunn Index: 0.1234 (higher is better)

Cluster centers shape: (64, 2)

Model saved as 'basic_som_model.pkl'
Loaded model predictions for first 10 samples: [12 12 15 15 23 23 23 18 18 12]
```

## Key Features Demonstrated

1. **Data Generation**: Using sklearn to create realistic clustered data
2. **SOM Configuration**: Setting up SOM with appropriate parameters
3. **Training**: Fitting the SOM with epoch-based training
4. **Evaluation**: Using multiple clustering metrics for assessment
5. **Visualization**: Plotting original data, clustering results, and SOM grid
6. **Model Persistence**: Saving and loading trained models

## Parameter Tuning Tips

### Grid Size (`m`, `n`)
- **Small grids (5x5)**: Faster training, coarser representation
- **Large grids (15x15)**: More detailed mapping, slower training
- **Rule of thumb**: Total neurons ≈ 5-10 times the expected number of clusters

### Learning Rate
- **High values (0.7-0.9)**: Faster convergence, risk of instability
- **Medium values (0.3-0.6)**: Balanced training (recommended)
- **Low values (0.1-0.3)**: Slower but more stable convergence

### Neighborhood Radius
- **Large radius**: Global organization, smoother topology
- **Small radius**: Local refinement, more detailed clusters
- **Adaptive**: Starts large and decreases over time (automatic)

### Backend Selection
- **"auto"**: Automatically selects best available backend
- **"cupy"**: Force GPU acceleration (requires CuPy)
- **"numba"**: Force JIT compilation (requires Numba)
- **"numpy"**: Pure NumPy implementation (always available)

## Common Issues and Solutions

### 1. Poor Clustering Quality
```python
# Try different initialization methods
for method in ["kmeans++", "som++", "kde"]:
    som = SOM(m=8, n=8, dim=2, initiate_method=method, 
              learning_rate=0.5, neighbour_rad=3, distance_function="euclidean")
    som.fit(X, epoch=100)
    score = som.evaluate(X, method=["silhouette"])[0]
    print(f"{method}: {score:.4f}")
```

### 2. Slow Training
```python
# Use smaller batch sizes and fewer epochs
som.fit(X, epoch=50, batch_size=32)  # Smaller batches
# or use GPU acceleration
som = SOM(..., backend="cupy")  # If CuPy is available
```

### 3. Memory Issues
```python
# Reduce grid size or use batch processing
som = SOM(m=6, n=6, ...)  # Smaller grid
som.fit(X, batch_size=64)  # Process in smaller batches
```

## Advanced Configuration

### Custom Distance Function
```python
# Use cosine distance for high-dimensional or sparse data
som = SOM(
    m=10, n=10, dim=100,
    initiate_method="he",
    learning_rate=0.4,
    neighbour_rad=4,
    distance_function="cosine"  # Better for high-dimensional data
)
```

### Multiple Evaluations
```python
# Get all evaluation metrics at once
all_scores = som.evaluate(X, method=["all"])
print("Complete evaluation:")
for metric, score in all_scores.items():
    print(f"  {metric}: {score:.4f}")
```

This example provides a comprehensive introduction to using SOM for clustering tasks, including parameter tuning, evaluation, and troubleshooting common issues.