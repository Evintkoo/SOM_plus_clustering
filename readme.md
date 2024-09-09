---

# Self-Organizing Map (SOM) Implementation

This repository contains an advanced implementation of the Self-Organizing Map (SOM) algorithm for unsupervised learning and clustering tasks. The SOM algorithm is particularly useful for visualizing high-dimensional data, performing dimensionality reduction, and clustering. This implementation is inspired by the paper "A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons" by Chaudhary, Bhatia, and Ahlawat (2014) and includes several unique features to enhance performance and flexibility.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Importing the SOM Class](#importing-the-som-class)
  - [Creating an SOM Instance](#creating-an-som-instance)
  - [Training the SOM](#training-the-som)
  - [Making Predictions](#making-predictions)
  - [Multiprocessing for Faster Training](#multiprocessing-for-faster-training)
  - [Evaluating the SOM](#evaluating-the-som)
  - [Visualization and Analysis](#visualization-and-analysis)
- [Advanced Use Cases](#advanced-use-cases)
- [Performance Optimization](#performance-optimization)
- [Evaluation Metrics](#evaluation-metrics)
- [Error Handling and Debugging](#error-handling-and-debugging)
- [Contribution Guidelines](#contribution-guidelines)
- [Licensing and Acknowledgments](#licensing-and-acknowledgments)
- [References](#references)

## Overview

### Purpose and Benefits
The main purpose of this SOM implementation is to provide an efficient and flexible tool for clustering and visualizing high-dimensional data. This implementation includes various enhancements, such as multiple initialization methods, distance metrics, and evaluation criteria, making it suitable for a wide range of applications, from data visualization to anomaly detection.

### Algorithm Description
The Self-Organizing Map (SOM) is a type of artificial neural network trained using unsupervised learning to produce a low-dimensional (typically two-dimensional) representation of input data. It uses competitive learning to find the best matching unit (BMU) and updates the neighborhood of the BMU using a Gaussian function to preserve the topological properties of the input space.

## Features

- **Initialization Methods:** Random, KDE, KMeans, KDE-KMeans, KMeans++, SOM++
- **Distance Functions:** Euclidean, Cosine
- **Evaluation Metrics:** Silhouette score, Davies-Bouldin index, Calinski-Harabasz score, Dunn index
- **Multiprocessing Support:** Leveraging joblib for parallel processing to accelerate training
- **Customizability:** Allows for customization of initialization methods, distance functions, learning rate, neighborhood functions, and more

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/som-implementation.git
cd som-implementation
pip install -r requirements.txt
```

### Requirements
- Python 3.7 or higher
- Libraries: `numpy`, `joblib`, `matplotlib`, `scipy` (for KDE), and other dependencies listed in `requirements.txt`.

## Usage

### Importing the SOM Class

```python
from som import SOM
```

### Creating an SOM Instance

```python
som = SOM(
    m=10, 
    n=10, 
    dim=3, 
    initiate_method='random', 
    learning_rate=0.5, 
    neighbour_rad=1.0, 
    distance_function='euclidean', 
    max_iter=1000
)
```

### Training the SOM

```python
import numpy as np
data = np.random.random((100, 3))  # Example data

som.fit(x=data, epoch=100, shuffle=True, verbose=True)
```

### Making Predictions

```python
labels = som.predict(data)
print(labels)
```

### Multiprocessing for Faster Training

```python
som.fit(x=data, epoch=100, shuffle=True, verbose=True, n_jobs=-1)
```

### Evaluating the SOM

```python
silhouette_score = som.evaluate(data, method=['silhouette'])
print(silhouette_score)

all_scores = som.evaluate(data, method=['all'])
print(all_scores)
```

### Visualization and Analysis

To visualize the trained SOM, you can use Python libraries like `matplotlib`:

```python
import matplotlib.pyplot as plt

# Visualize the neurons
plt.imshow(som.cluster_center_.reshape(som.m, som.n, som.dim))
plt.title('Self-Organizing Map Neurons')
plt.show()
```

## Advanced Use Cases

1. **Anomaly Detection:** Use SOM to identify anomalies in time series data or financial transactions by detecting clusters that differ significantly from the norm.
2. **Customer Segmentation:** Segment customers based on purchasing patterns, demographics, or behavior data.
3. **Dimensionality Reduction:** Reduce high-dimensional data into a lower-dimensional space while preserving its topological properties.
4. **Integration with Machine Learning Tools:** Use the SOM output as features for downstream machine learning tasks, such as classification or regression.

## Performance Optimization

- **Use Multiprocessing:** Utilize the `n_jobs` parameter for parallel processing on multi-core systems to speed up training.
- **Data Preprocessing:** Normalize input data to ensure faster convergence and better clustering performance.
- **Memory Management:** For large datasets, consider using batch processing or splitting data into chunks.

## Evaluation Metrics

- **Silhouette Score:** Measures how similar each point is to its own cluster compared to other clusters.
- **Davies-Bouldin Index:** Computes the average similarity ratio of each cluster with the most similar cluster.
- **Calinski-Harabasz Score:** Evaluates the ratio of between-cluster variance to within-cluster variance.
- **Dunn Index:** Determines the distance between clusters divided by the size of the largest cluster.

## Error Handling and Debugging

- **Common Errors:**
  - **ValueError:** Raised when an invalid parameter is provided. Check your inputs against the valid options listed in the documentation.
  - **RuntimeError:** Thrown if the SOM is used before fitting the data.
  - **Dimension Mismatch:** Ensure that the input data dimensions match the expected dimensions specified during SOM initialization.
- **Debugging Tips:**
  - Use verbose mode (`verbose=True`) during training to see progress and intermediate results.
  - Check input data for NaN or infinite values which may cause unexpected behavior.

## Contribution Guidelines

We welcome contributions from the community! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request and describe the changes you made.

## Licensing and Acknowledgments

This project is licensed under the MIT License. See the `LICENSE` file for more details.

### Acknowledgments

- This implementation is inspired by the paper: Chaudhary, V., Bhatia, R. S., & Ahlawat, A. K. (2014). "A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons." Alexandria Engineering Journal, 53(4), 827-831. [Link to paper](https://doi.org/10.1016/j.aej.2014.09.007)

## References

- Chaudhary, V., Bhatia, R. S., & Ahlawat, A. K. (2014). "A novel self-organizing map (SOM) learning algorithm with nearest and farthest neurons." Alexandria Engineering Journal, 53(4), 827-831. [Link to paper](https://doi.org/10.1016/j.aej.2014.09.007)

---