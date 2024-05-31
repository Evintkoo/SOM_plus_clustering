---

# Self-Organizing Map (SOM) Implementation

This repository contains an implementation of a Self-Organizing Map (SOM) algorithm, inspired by the paper "A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons" by Chaudhary, Bhatia, and Ahlawat (2014). The implementation includes various initialization methods and evaluation metrics for clustering quality.

## Features

- Initialization methods: random, kde, kmeans, kde_kmeans, kmeans++, SOM++
- Distance functions: Euclidean, cosine
- Evaluation metrics: Silhouette score, Davies-Bouldin index, Calinski-Harabasz score, Dunn index
- Support for multiprocessing to speed up training

## Installation

To use this implementation, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/som-implementation.git
cd som-implementation
pip install -r requirements.txt
```

## Usage

### Importing the SOM Class

```python
from som import SOM
```

### Creating an SOM Instance

```python
som = SOM(m=10, n=10, dim=3, initiate_method='random', learning_rate=0.5, neighbour_rad=1.0, distance_function='euclidean', max_iter=1000)
```

### Training the SOM

```python
# Example data
import numpy as np
data = np.random.random((100, 3))

# Fit the model
som.fit(X=data, epoch=100, shuffle=True, verbose=True)
```

### Making Predictions

```python
labels = som.predict(data)
print(labels)
```

### Multiprocessing for Faster Training

```python
som.fit(X=data, epoch=100, shuffle=True, verbose=True, use_multiprocessing=True)
```

### Evaluating the SOM

```python
silhouette_score = som.evaluate(data, method='silhouette')
print(silhouette_score)

all_scores = som.evaluate(data, method='all')
print(all_scores)
```

## Detailed Documentation

### SOM Class

#### Initialization

```python
SOM(m, n, dim, initiate_method, learning_rate, neighbour_rad, distance_function, max_iter=None)
```

- `m`: Height of the SOM matrix.
- `n`: Width of the SOM matrix.
- `dim`: Dimension of the input data.
- `initiate_method`: Method for initializing neurons ('random', 'kde', 'kmeans', 'kde_kmeans', 'kmeans++', 'SOM++').
- `learning_rate`: Initial learning rate for the SOM.
- `neighbour_rad`: Initial neighbourhood radius.
- `distance_function`: Distance function to use ('euclidean', 'cosine').
- `max_iter`: Maximum number of iterations for training.

#### Methods

- `fit(X, epoch, shuffle=True, verbose=True, use_multiprocessing=False)`: Train the SOM on the input data `X` for a specified number of `epoch`s.
- `predict(X)`: Predict the labels for the input data `X` based on the trained SOM.
- `fit_predict(X, epoch, shuffle=True, verbose=True)`: Train the SOM and return the predicted labels.
- `evaluate(X, method='silhouette')`: Evaluate the SOM using the specified evaluation method. Available methods are 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'dunn', and 'all'.
- `cluster_center_`: Property to get the list of all neurons in the SOM.

### References

Chaudhary, V., Bhatia, R. S., & Ahlawat, A. K. (2014). A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons. Alexandria Engineering Journal, 53(4), 827-831. [https://doi.org/10.1016/j.aej.2014.09.007](https://doi.org/10.1016/j.aej.2014.09.007)

---