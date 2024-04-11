# SOM Clustering Library

The SOM Clustering Library is a Python implementation of the Self-Organizing Map (SOM) algorithm for clustering and visualization of high-dimensional data.

## Features

- Flexible initialization methods: random or PCA-based
- Customizable neighborhood radius and learning rate
- Support for different distance functions: Euclidean, Manhattan, or custom
- Visualization of the trained SOM grid
- Prediction of cluster assignments for new data points

## Installation

To install the SOM Clustering Library, clone the repository using the following command:

```
git clone https://github.com/Evintkoo/SOM_plus_clustering.git
```

## Usage

### 1. Import the Library

```python
from modules.som import SOM
```

### 2. Create an SOM object

```python
model = SOM(m=2, n=2, dim=X.shape[1], initiate_method="random", neighbour_rad=0.1, learning_rate=0.1, distance_function="euclidean")
```

Parameters:
- `m`: Number of rows in the SOM grid
- `n`: Number of columns in the SOM grid
- `dim`: Dimension of the input data
- `initiate_method`: Initialization method for the SOM grid ("random" or "pca")
- `neighbour_rad`: Initial neighborhood radius
- `learning_rate`: Initial learning rate
- `distance_function`: Distance function to use ("euclidean", "manhattan", or a custom function)

### 3. Fit the model

```python
model.fit(X)
```

Parameters:
- `X`: Input data matrix (numpy array)

### 4. Predict cluster assignments

```python
labels = model.predict(X)
```

Parameters:
- `X`: Input data matrix (numpy array)

Returns:
- `labels`: Cluster assignments for each data point

### Additional Utilities

- `model.plot_som()`: Visualize the trained SOM grid
- `model.get_weights()`: Get the learned weights of the SOM grid
- `model.get_bmus(X)`: Get the Best Matching Units (BMUs) for each data point

## Examples

For detailed examples and usage, please refer to the [examples](examples/) directory.

## Contributing

Contributions to the SOM Clustering Library are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the [GitHub repository](https://github.com/Evintkoo/SOM_plus_clustering).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

The SOM Clustering Library is based on the Self-Organizing Map algorithm proposed by Teuvo Kohonen. We would like to acknowledge the contributions of the open-source community and the developers of the libraries used in this project.