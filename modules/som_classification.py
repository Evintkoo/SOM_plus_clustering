import numpy as np
import math
from .utils import one_hot_encode, normalize_column, euc_distance
from .som import SOM

class som_classification:
    def __init__(self, m: int, n: int, X: np.array, y: np.array) -> None:
        # Normalize each column
        for col in range(X.shape[1]):
            X[:, col] = normalize_column(X, col)
        
        y = np.reshape(y, -1)
        self.total_neurons_representation = m * n
        self.neuron_shape = (m, n)
        self.classes = np.unique(y)
        self.dataset = []
        self.all_dataset = X
        self.true = y
        self.true_encoded = one_hot_encode(y)
        self.best_models = None
        self.trained = False
        for c in self.classes:
            X_filtered = X[(y == c)]
            self.dataset.append((X_filtered, c))
        # Initialize weights for the models

    def predict(self, X: np.array):
        predictions = []

        for data in X:
            list_samples = [
                model.neurons[model.index_bmu(np.array([data]))[0]][model.index_bmu(np.array([data]))[1]]
                for model in self.models
            ]
            distances = [math.exp(1 / (euc_distance(data, sample) + 1)) for sample in list_samples]
            dist_sum = sum(distances)
            normalized_distances = np.array([dist / dist_sum for dist in distances])
            # Apply weights to normalized distances
            predictions.append(normalized_distances)
        
        return np.array(predictions)
    
    def fit(self, epoch: int = 10, initiate_method = 'kde', learning_rate=1.5, distance_function = "euclidean"):
        if not self.trained:
            self.trained = True
            self.models = []
            for data_points, label in self.dataset:
                som = SOM(m=self.neuron_shape[0], n=self.neuron_shape[1], dim=self.all_dataset.shape[1], 
                          initiate_method=initiate_method, learning_rate=learning_rate, 
                          neighbour_rad=2.0, distance_function=distance_function, max_iter=None)
                som.fit(X=data_points, epoch=epoch, verbose=False)
                self.models.append(som)
        else:
            for datasets, som_model in zip(self.dataset, self.models):
                data_points, label = datasets
                som_model.fit(X=data_points, epoch=epoch, verbose=False)
        
    def eval(self):
        pred = self.predict(self.all_dataset)
        mean_squared_error = np.mean(np.sum((self.true_encoded - pred) ** 2, axis=1))
        accuracy = np.sum([1 if self.classes[np.argmax(pred_data)] == true_data else 0 for pred_data, true_data in zip(pred, self.true)]) / len(pred)
        return mean_squared_error, accuracy
    
    def train(self, retry: int = 10, train_batch: int = 24):
        self.fit(1)
        best_mse = float('inf')
        best_acc = 0
        while retry > 0:
            mse, acc = self.eval()
            self.fit(train_batch)
            print(mse, acc)
            if acc > best_acc:
                best_acc = acc
                self.best_models = self.models
                retry = 10
            else:
                retry -= 1
        self.models = self.best_models
        print("done training, update weight")
        return self.eval(), (best_mse, best_acc)