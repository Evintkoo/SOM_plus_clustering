# SOM Classification Example

This example demonstrates how to use the supervised SOM for classification tasks, including training with labeled data and evaluating classification performance.

## Example: Multi-class Classification

```python
import numpy as np
import matplotlib.pyplot as plt
from modules.som_classification import SOM
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Generate synthetic classification data
print("=== Generating Classification Data ===")
X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=6,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=2,
    class_sep=1.5,
    random_state=42
)

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardize features (important for SOM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Create supervised SOM
print("\n=== Creating Supervised SOM ===")
som = SOM(
    m=8,                           # Grid height
    n=8,                           # Grid width
    dim=X.shape[1],                # Input dimensions
    initiate_method="som++",       # Initialization method
    learning_rate=0.5,             # Initial learning rate
    neighbour_rad=3,               # Initial neighborhood radius
    distance_function="euclidean", # Distance function
    max_iter=15000                 # Maximum iterations
)

print(f"SOM grid size: {som.m} x {som.n}")
print(f"Total neurons: {som.m * som.n}")
print(f"Input dimensions: {som.dim}")

# Train the supervised SOM
print("\n=== Training SOM with Labels ===")
som.fit(
    x=X_train_scaled, 
    y=y_train, 
    epoch=150,
    shuffle=True,
    verbose=True,
    n_jobs=-1  # Use all available cores
)

print("Training completed!")
print(f"Final learning rate: {som.cur_learning_rate:.6f}")
print(f"Final neighborhood radius: {som.cur_neighbour_rad:.6f}")

# Analyze neuron label distribution
print("\n=== Neuron Label Analysis ===")
unique_labels, counts = np.unique(som.neuron_label, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} neurons ({count/(som.m*som.n)*100:.1f}%)")

# Make predictions on test set
print("\n=== Making Predictions ===")
y_pred = som.predict(X_test_scaled)

# Evaluate classification performance
print("\n=== Classification Evaluation ===")
evaluation_results = som.evaluate(X_test_scaled, y_test, method=["all"])

print("Performance Metrics:")
for metric, score in evaluation_results.items():
    print(f"  {metric.capitalize()}: {score:.4f}")

# Detailed classification report
print("\n=== Detailed Classification Report ===")
print(classification_report(y_test, y_pred, target_names=[f"Class {i}" for i in range(3)]))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Training data distribution
axes[0, 0].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='viridis', alpha=0.7)
axes[0, 0].set_title('Training Data Distribution')
axes[0, 0].set_xlabel('Feature 1 (standardized)')
axes[0, 0].set_ylabel('Feature 2 (standardized)')

# 2. Test predictions vs true labels
axes[0, 1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='viridis', alpha=0.7, label='True')
axes[0, 1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.3, marker='x', label='Predicted')
axes[0, 1].set_title('Test Set: True vs Predicted Labels')
axes[0, 1].set_xlabel('Feature 1 (standardized)')
axes[0, 1].set_ylabel('Feature 2 (standardized)')
axes[0, 1].legend()

# 3. SOM neuron labels visualization
neuron_positions = np.array([(i, j) for i in range(som.m) for j in range(som.n)])
im = axes[1, 0].scatter(neuron_positions[:, 1], neuron_positions[:, 0], 
                        c=som.neuron_label.flatten(), cmap='viridis', s=100, marker='s')
axes[1, 0].set_title('SOM Grid - Neuron Labels')
axes[1, 0].set_xlabel('Grid X')
axes[1, 0].set_ylabel('Grid Y')
plt.colorbar(im, ax=axes[1, 0])

# 4. Confusion matrix heatmap
im2 = axes[1, 1].imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
axes[1, 1].set_title('Confusion Matrix')
axes[1, 1].set_xlabel('Predicted Label')
axes[1, 1].set_ylabel('True Label')

# Add text annotations to confusion matrix
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        axes[1, 1].text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")

plt.colorbar(im2, ax=axes[1, 1])
plt.tight_layout()
plt.show()

# Save the trained model
model_path = "som_classification_model.pkl"
som.save(model_path)
print(f"\nModel saved as '{model_path}'")
```

## Real-World Example: Iris Dataset

```python
# Load the classic Iris dataset
print("\n" + "="*50)
print("=== Real-World Example: Iris Dataset ===")
print("="*50)

from sklearn.datasets import load_iris

# Load and prepare Iris data
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

print(f"Iris dataset shape: {X_iris.shape}")
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")
print(f"Class distribution: {np.bincount(y_iris)}")

# Split and standardize
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

scaler_iris = StandardScaler()
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
X_test_iris_scaled = scaler_iris.transform(X_test_iris)

# Create and train SOM for Iris
som_iris = SOM(
    m=6, n=6, dim=4,
    initiate_method="kmeans++",
    learning_rate=0.4,
    neighbour_rad=2,
    distance_function="euclidean"
)

som_iris.fit(X_train_iris_scaled, y_train_iris, epoch=200, shuffle=True, verbose=False)

# Evaluate on Iris
y_pred_iris = som_iris.predict(X_test_iris_scaled)
iris_results = som_iris.evaluate(X_test_iris_scaled, y_test_iris, method=["all"])

print("\nIris Classification Results:")
for metric, score in iris_results.items():
    print(f"  {metric.capitalize()}: {score:.4f}")

print("\nIris Classification Report:")
print(classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names))
```

## Cross-Validation Example

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_som_classification(X, y, n_splits=5):
    """Perform cross-validation on SOM classifier."""
    print(f"\n=== {n_splits}-Fold Cross-Validation ===")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    accuracy_scores = []
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Processing fold {fold + 1}/{n_splits}...")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Standardize each fold separately
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold)
        
        # Create and train SOM
        som_fold = SOM(
            m=6, n=6, dim=X.shape[1],
            initiate_method="som++",
            learning_rate=0.4,
            neighbour_rad=2,
            distance_function="euclidean"
        )
        
        som_fold.fit(X_train_fold_scaled, y_train_fold, epoch=100, verbose=False)
        
        # Evaluate on validation set
        scores = som_fold.evaluate(X_val_fold_scaled, y_val_fold, method=["accuracy", "f1_score"])
        accuracy_scores.append(scores[0])
        f1_scores.append(scores[1])
    
    print(f"\nCross-validation results:")
    print(f"Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    return accuracy_scores, f1_scores

# Run cross-validation on the synthetic dataset
cv_accuracy, cv_f1 = cross_validate_som_classification(X, y)
```

## Hyperparameter Tuning

```python
def tune_som_hyperparameters(X_train, y_train, X_val, y_val):
    """Simple grid search for SOM hyperparameters."""
    print("\n=== Hyperparameter Tuning ===")
    
    # Define parameter grid
    param_grid = {
        'grid_size': [(6, 6), (8, 8), (10, 10)],
        'learning_rate': [0.3, 0.5, 0.7],
        'neighbour_rad': [2, 3, 4],
        'initiate_method': ['kmeans++', 'som++', 'he']
    }
    
    best_score = 0
    best_params = {}
    
    for grid_size in param_grid['grid_size']:
        for lr in param_grid['learning_rate']:
            for radius in param_grid['neighbour_rad']:
                for init_method in param_grid['initiate_method']:
                    try:
                        # Create SOM with current parameters
                        som_tune = SOM(
                            m=grid_size[0], n=grid_size[1], dim=X_train.shape[1],
                            initiate_method=init_method,
                            learning_rate=lr,
                            neighbour_rad=radius,
                            distance_function="euclidean"
                        )
                        
                        # Train and evaluate
                        som_tune.fit(X_train, y_train, epoch=50, verbose=False)
                        accuracy = som_tune.evaluate(X_val, y_val, method=["accuracy"])[0]
                        
                        # Check if this is the best so far
                        if accuracy > best_score:
                            best_score = accuracy
                            best_params = {
                                'grid_size': grid_size,
                                'learning_rate': lr,
                                'neighbour_rad': radius,
                                'initiate_method': init_method
                            }
                        
                        print(f"Grid: {grid_size}, LR: {lr}, Radius: {radius}, "
                              f"Init: {init_method} -> Accuracy: {accuracy:.4f}")
                        
                    except Exception as e:
                        print(f"Error with params {grid_size}, {lr}, {radius}, {init_method}: {e}")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_score:.4f}")
    
    return best_params, best_score

# Run hyperparameter tuning
best_params, best_score = tune_som_hyperparameters(
    X_train_scaled, y_train, X_test_scaled, y_test
)
```

## Output Example

```
=== Generating Classification Data ===
Dataset shape: (1000, 8)
Number of classes: 3
Class distribution: [334 333 333]
Training set: (700, 8)
Test set: (300, 8)

=== Creating Supervised SOM ===
SOM grid size: 8 x 8
Total neurons: 64
Input dimensions: 8

=== Training SOM with Labels ===
Training completed!
Final learning rate: 0.000234
Final neighborhood radius: 0.000567

=== Neuron Label Analysis ===
Class 0: 22 neurons (34.4%)
Class 1: 21 neurons (32.8%)
Class 2: 21 neurons (32.8%)

=== Classification Evaluation ===
Performance Metrics:
  Accuracy: 0.8967
  F1_score: 0.8954
  Recall: 0.8967

=== Detailed Classification Report ===
              precision    recall  f1-score   support

     Class 0       0.90      0.89      0.89       100
     Class 1       0.88      0.91      0.89       100
     Class 2       0.91      0.89      0.90       100

    accuracy                           0.90       300
   macro avg       0.90      0.90      0.90       300
weighted avg       0.90      0.90      0.90       300

Confusion Matrix:
[[89  6  5]
 [ 5 91  4]
 [ 7  4 89]]

Model saved as 'som_classification_model.pkl'
```

## Key Features Demonstrated

1. **Supervised Learning**: Training SOM with labeled data
2. **Data Preprocessing**: Feature standardization for better performance
3. **Model Evaluation**: Comprehensive classification metrics
4. **Visualization**: Multiple plots showing training data, predictions, and SOM structure
5. **Cross-Validation**: Robust performance estimation
6. **Hyperparameter Tuning**: Systematic parameter optimization
7. **Real-World Application**: Using classic Iris dataset

## Best Practices for SOM Classification

1. **Feature Scaling**: Always standardize features before training
2. **Grid Size**: Balance between detail and computational cost
3. **Training Epochs**: More epochs generally improve performance
4. **Evaluation**: Use multiple metrics for comprehensive assessment
5. **Cross-Validation**: Validate performance across different data splits