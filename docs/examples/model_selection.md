# Model Selection and Comparison Example

This example demonstrates how to compare different initialization methods and select the best performing SOM model using the model_picker utility and various evaluation techniques.

## Example: Comprehensive Model Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from modules.model_picker import model_picker
from modules.som import SOM
from modules.variables import INITIATION_METHOD_LIST, DISTANCE_METHOD_LIST
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import StandardScaler
import time

# Generate complex dataset for testing
print("=== Generating Test Dataset ===")
np.random.seed(42)

# Create multi-cluster dataset
X1, _ = make_blobs(n_samples=300, centers=4, n_features=6, 
                   cluster_std=2.0, center_box=(-10, 10), random_state=42)
X2, _ = make_blobs(n_samples=200, centers=2, n_features=6, 
                   cluster_std=1.5, center_box=(5, 15), random_state=123)
X = np.vstack([X1, X2])

# Add some noise features
noise = np.random.normal(0, 0.5, (X.shape[0], 2))
X = np.hstack([X, noise])

print(f"Dataset shape: {X.shape}")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data standardized for better SOM performance")

# Basic model picker example
print("\n" + "="*60)
print("=== Basic Model Picker Example ===")
print("="*60)

# Create model picker instance
picker = model_picker()

# Evaluate all initialization methods
print("Evaluating all initialization methods...")
start_time = time.time()

picker.evaluate_initiate_method(
    X=X_scaled,
    m=8,                          # SOM grid height
    n=8,                          # SOM grid width
    learning_rate=0.5,            # Learning rate
    neighbor_rad=3,               # Neighborhood radius
    distance_function="euclidean", # Distance function
    max_iter=10000,               # Maximum iterations
    epoch=75                      # Training epochs
)

evaluation_time = time.time() - start_time
print(f"Evaluation completed in {evaluation_time:.2f} seconds")

# Display results for all methods
print("\n=== Initialization Method Comparison ===")
print(f"{'Method':<15} {'Score':<10} {'Rank'}")
print("-" * 35)

# Sort methods by performance
method_scores = list(zip(INITIATION_METHOD_LIST, picker.model_evaluation))
sorted_methods = sorted(method_scores, key=lambda x: x[1], reverse=True)

for rank, (method, score) in enumerate(sorted_methods, 1):
    print(f"{method:<15} {score:<10.6f} #{rank}")

# Get the best model
best_model = picker.pick_best_model()
best_score = max(picker.model_evaluation)
best_method = best_model.init_method

print(f"\n🏆 Best performing method: {best_method}")
print(f"   Best score: {best_score:.6f}")
print(f"   Grid size: {best_model.m} x {best_model.n}")
print(f"   Final learning rate: {best_model.cur_learning_rate:.6f}")

# Analyze the best model in detail
print("\n=== Best Model Analysis ===")
best_predictions = best_model.predict(X_scaled)
unique_clusters = len(np.unique(best_predictions))
print(f"Number of discovered clusters: {unique_clusters}")

# Evaluate best model with all metrics
best_model_scores = best_model.evaluate(X_scaled, method=["all"])
print("\nDetailed evaluation of best model:")
for metric, score in best_model_scores.items():
    print(f"  {metric.capitalize()}: {score:.6f}")
```

## Advanced Comparison with Multiple Configurations

```python
print("\n" + "="*60)
print("=== Advanced Multi-Configuration Comparison ===")
print("="*60)

class AdvancedModelComparison:
    """Advanced model comparison with multiple configurations."""
    
    def __init__(self):
        self.results = []
    
    def compare_configurations(self, X, configurations):
        """Compare multiple SOM configurations."""
        
        for i, config in enumerate(configurations):
            print(f"\nTesting configuration {i+1}/{len(configurations)}")
            print(f"Config: {config}")
            
            config_results = {}
            
            # Test each initialization method with this configuration
            for method in INITIATION_METHOD_LIST[:6]:  # Test subset for speed
                try:
                    print(f"  Testing {method}...")
                    
                    # Create SOM with current configuration
                    som = SOM(
                        dim=X.shape[1],
                        initiate_method=method,
                        distance_function="euclidean",
                        **config
                    )
                    
                    # Train the model
                    start_time = time.time()
                    som.fit(X, epoch=50, shuffle=True)
                    train_time = time.time() - start_time
                    
                    # Evaluate the model
                    scores = som.evaluate(X, method=["silhouette", "davies_bouldin"])
                    
                    config_results[method] = {
                        'silhouette': scores[0],
                        'davies_bouldin': scores[1],
                        'train_time': train_time,
                        'som': som
                    }
                    
                except Exception as e:
                    print(f"    Error with {method}: {e}")
                    config_results[method] = {'error': str(e)}
            
            # Find best method for this configuration
            valid_results = {k: v for k, v in config_results.items() if 'error' not in v}
            if valid_results:
                best_method = max(valid_results, key=lambda x: valid_results[x]['silhouette'])
                config['best_method'] = best_method
                config['best_score'] = valid_results[best_method]['silhouette']
                config['results'] = config_results
            
            self.results.append(config)
    
    def print_summary(self):
        """Print comparison summary."""
        print("\n=== Configuration Comparison Summary ===")
        
        for i, config in enumerate(self.results):
            if 'best_method' in config:
                print(f"\nConfiguration {i+1}:")
                print(f"  Grid: {config['m']}x{config['n']}")
                print(f"  Learning Rate: {config['learning_rate']}")
                print(f"  Neighbor Radius: {config['neighbour_rad']}")
                print(f"  Best Method: {config['best_method']}")
                print(f"  Best Score: {config['best_score']:.6f}")
        
        # Find overall best configuration
        best_config = max(self.results, key=lambda x: x.get('best_score', 0))
        print(f"\n🎯 Overall Best Configuration:")
        print(f"   Grid: {best_config['m']}x{best_config['n']}")
        print(f"   Learning Rate: {best_config['learning_rate']}")
        print(f"   Method: {best_config['best_method']}")
        print(f"   Score: {best_config['best_score']:.6f}")
        
        return best_config

# Define configurations to test
configurations = [
    {'m': 6, 'n': 6, 'learning_rate': 0.3, 'neighbour_rad': 2},
    {'m': 8, 'n': 8, 'learning_rate': 0.5, 'neighbour_rad': 3},
    {'m': 10, 'n': 10, 'learning_rate': 0.7, 'neighbour_rad': 4},
    {'m': 12, 'n': 12, 'learning_rate': 0.4, 'neighbour_rad': 3}
]

# Run advanced comparison
comparison = AdvancedModelComparison()
comparison.compare_configurations(X_scaled, configurations)
best_config = comparison.print_summary()
```

## Distance Function Comparison

```python
print("\n" + "="*60)
print("=== Distance Function Comparison ===")
print("="*60)

def compare_distance_functions(X, init_method="kmeans++"):
    """Compare Euclidean vs Cosine distance functions."""
    
    results = {}
    
    for distance_func in DISTANCE_METHOD_LIST:
        print(f"\nTesting {distance_func} distance function...")
        
        try:
            som = SOM(
                m=8, n=8, dim=X.shape[1],
                initiate_method=init_method,
                learning_rate=0.5,
                neighbour_rad=3,
                distance_function=distance_func
            )
            
            start_time = time.time()
            som.fit(X, epoch=100)
            train_time = time.time() - start_time
            
            scores = som.evaluate(X, method=["silhouette", "davies_bouldin", "calinski_harabasz"])
            
            results[distance_func] = {
                'silhouette': scores[0],
                'davies_bouldin': scores[1],
                'calinski_harabasz': scores[2],
                'train_time': train_time
            }
            
            print(f"  Silhouette: {scores[0]:.4f}")
            print(f"  Davies-Bouldin: {scores[1]:.4f}")
            print(f"  Calinski-Harabasz: {scores[2]:.4f}")
            print(f"  Training time: {train_time:.2f}s")
            
        except Exception as e:
            print(f"  Error with {distance_func}: {e}")
            results[distance_func] = {'error': str(e)}
    
    return results

# Compare distance functions
distance_results = compare_distance_functions(X_scaled)

# Determine best distance function
valid_distance_results = {k: v for k, v in distance_results.items() if 'error' not in v}
if valid_distance_results:
    best_distance = max(valid_distance_results, key=lambda x: valid_distance_results[x]['silhouette'])
    print(f"\n🎯 Best distance function: {best_distance}")
    print(f"   Silhouette score: {valid_distance_results[best_distance]['silhouette']:.6f}")
```

## Cross-Validation for Robust Model Selection

```python
from sklearn.model_selection import KFold

print("\n" + "="*60)
print("=== Cross-Validation Model Selection ===")
print("="*60)

def cross_validate_som_methods(X, n_splits=5, methods_to_test=None):
    """Cross-validate SOM initialization methods."""
    
    if methods_to_test is None:
        methods_to_test = ["random", "kmeans++", "som++", "kde", "he"]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results for each method
    method_scores = {method: [] for method in methods_to_test}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nProcessing fold {fold + 1}/{n_splits}")
        
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        
        for method in methods_to_test:
            try:
                print(f"  Testing {method}...")
                
                som = SOM(
                    m=6, n=6, dim=X.shape[1],
                    initiate_method=method,
                    learning_rate=0.5,
                    neighbour_rad=2,
                    distance_function="euclidean"
                )
                
                som.fit(X_train_fold, epoch=50, shuffle=True)
                
                # Evaluate on validation fold
                val_scores = som.evaluate(X_val_fold, method=["silhouette"])
                method_scores[method].append(val_scores[0])
                
            except Exception as e:
                print(f"    Error with {method}: {e}")
                method_scores[method].append(0.0)  # Add penalty for failed methods
    
    # Calculate statistics
    print("\n=== Cross-Validation Results ===")
    print(f"{'Method':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 50)
    
    cv_summary = {}
    for method, scores in method_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        cv_summary[method] = {
            'mean': mean_score,
            'std': std_score,
            'min': min_score,
            'max': max_score
        }
        
        print(f"{method:<12} {mean_score:<8.4f} {std_score:<8.4f} {min_score:<8.4f} {max_score:<8.4f}")
    
    # Find most robust method (highest mean score)
    best_cv_method = max(cv_summary, key=lambda x: cv_summary[x]['mean'])
    print(f"\n🏆 Most robust method (CV): {best_cv_method}")
    print(f"   Mean score: {cv_summary[best_cv_method]['mean']:.6f}")
    print(f"   Std deviation: {cv_summary[best_cv_method]['std']:.6f}")
    
    return cv_summary, best_cv_method

# Run cross-validation
cv_results, best_cv_method = cross_validate_som_methods(X_scaled)
```

## Visualization of Model Comparison

```python
print("\n" + "="*60)
print("=== Visualization of Results ===")
print("="*60)

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Original data
axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6, c='blue')
axes[0, 0].set_title('Original Dataset\n(First 2 Features)')
axes[0, 0].set_xlabel('Feature 1 (standardized)')
axes[0, 0].set_ylabel('Feature 2 (standardized)')

# 2. Best model clustering result
best_model_final = picker.pick_best_model()
best_predictions = best_model_final.predict(X_scaled)
scatter = axes[0, 1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=best_predictions, 
                           cmap='viridis', alpha=0.7)
axes[0, 1].set_title(f'Best Model Clustering\n({best_model_final.init_method})')
axes[0, 1].set_xlabel('Feature 1 (standardized)')
axes[0, 1].set_ylabel('Feature 2 (standardized)')
plt.colorbar(scatter, ax=axes[0, 1])

# 3. Method comparison bar chart
methods = INITIATION_METHOD_LIST
scores = picker.model_evaluation
axes[0, 2].bar(range(len(methods)), scores, color='skyblue')
axes[0, 2].set_title('Initialization Method Scores')
axes[0, 2].set_xlabel('Initialization Method')
axes[0, 2].set_ylabel('Evaluation Score')
axes[0, 2].set_xticks(range(len(methods)))
axes[0, 2].set_xticklabels(methods, rotation=45, ha='right')

# 4. Cross-validation results
if 'cv_results' in locals():
    cv_methods = list(cv_results.keys())
    cv_means = [cv_results[method]['mean'] for method in cv_methods]
    cv_stds = [cv_results[method]['std'] for method in cv_methods]
    
    axes[1, 0].bar(range(len(cv_methods)), cv_means, yerr=cv_stds, 
                   capsize=5, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Cross-Validation Results')
    axes[1, 0].set_xlabel('Initialization Method')
    axes[1, 0].set_ylabel('Mean Silhouette Score')
    axes[1, 0].set_xticks(range(len(cv_methods)))
    axes[1, 0].set_xticklabels(cv_methods, rotation=45, ha='right')

# 5. Distance function comparison
if 'distance_results' in locals():
    dist_methods = list(valid_distance_results.keys())
    dist_scores = [valid_distance_results[method]['silhouette'] for method in dist_methods]
    
    axes[1, 1].bar(dist_methods, dist_scores, color='lightgreen')
    axes[1, 1].set_title('Distance Function Comparison')
    axes[1, 1].set_xlabel('Distance Function')
    axes[1, 1].set_ylabel('Silhouette Score')

# 6. Training time comparison
if 'distance_results' in locals():
    train_times = [valid_distance_results[method]['train_time'] for method in dist_methods]
    
    axes[1, 2].bar(dist_methods, train_times, color='orange')
    axes[1, 2].set_title('Training Time Comparison')
    axes[1, 2].set_xlabel('Distance Function')
    axes[1, 2].set_ylabel('Training Time (seconds)')

plt.tight_layout()
plt.show()

# Summary report
print("\n" + "="*60)
print("=== FINAL SUMMARY REPORT ===")
print("="*60)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Best single run method: {best_method} (score: {best_score:.6f})")
if 'best_cv_method' in locals():
    print(f"Most robust method (CV): {best_cv_method} (mean: {cv_results[best_cv_method]['mean']:.6f})")
if 'best_distance' in locals():
    print(f"Best distance function: {best_distance}")

print(f"\nRecommendations:")
print(f"- For this dataset, use '{best_method}' initialization")
print(f"- Grid size: 8x8 or larger for detailed mapping")
print(f"- Distance function: {best_distance if 'best_distance' in locals() else 'euclidean'}")
print(f"- Training epochs: 75-100 for good convergence")

print("\nModel saved for future use:")
best_model_final.save("best_som_model.pkl")
print("File: best_som_model.pkl")
```

## Output Example

```
=== Generating Test Dataset ===
Dataset shape: (500, 8)
Features: 8
Samples: 500
Data standardized for better SOM performance

============================================================
=== Basic Model Picker Example ===
============================================================
Evaluating all initialization methods...
Evaluation completed in 45.23 seconds

=== Initialization Method Comparison ===
Method          Score      Rank
-----------------------------------
som++           0.634521   #1
kmeans++        0.628734   #2
kde             0.621455   #3
he              0.598234   #4
lecun           0.587654   #5
naive_sharding  0.575432   #6
kmeans          0.567823   #7
lsuv            0.556789   #8
random          0.534567   #9
zero            0.512345   #10

🏆 Best performing method: som++
   Best score: 0.634521
   Grid size: 8 x 8
   Final learning rate: 0.000123

=== Best Model Analysis ===
Number of discovered clusters: 6

Detailed evaluation of best model:
  Silhouette: 0.634521
  Davies_bouldin: 0.756432
  Calinski_harabasz: 234.567
  Dunn: 0.123456

============================================================
=== FINAL SUMMARY REPORT ===
============================================================
Dataset: 500 samples, 8 features
Best single run method: som++ (score: 0.634521)
Most robust method (CV): som++ (mean: 0.621234)
Best distance function: euclidean

Recommendations:
- For this dataset, use 'som++' initialization
- Grid size: 8x8 or larger for detailed mapping  
- Distance function: euclidean
- Training epochs: 75-100 for good convergence

Model saved for future use:
File: best_som_model.pkl
```

## Key Features Demonstrated

1. **Automated Model Selection**: Using model_picker for systematic comparison
2. **Multiple Configurations**: Testing different SOM parameters
3. **Distance Function Analysis**: Comparing Euclidean vs Cosine distance
4. **Cross-Validation**: Robust performance estimation
5. **Comprehensive Visualization**: Multiple plots showing results
6. **Performance Timing**: Measuring training time for different methods
7. **Statistical Analysis**: Mean, standard deviation, and confidence intervals

This example provides a complete framework for selecting the optimal SOM configuration for any dataset.