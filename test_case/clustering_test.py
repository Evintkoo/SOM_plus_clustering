import numpy as np
import logging
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from modules.som import SOM 

# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_report.log"),
        logging.StreamHandler()
    ]
)

def test_initiate_neuron():
    # Testing different initiation methods
    data = np.random.rand(100, 3)  # Sample data with 100 samples and 3 features
    som = SOM(m=5, n=5, dim=3, initiate_method="random", learning_rate=0.5, neighbour_rad=2, distance_function="euclidean")
    
    neurons_random = som.initiate_neuron(data)
    logging.info(f"Neuron initialization with 'random': {neurons_random.shape}")

    som.init_method = "kde"
    neurons_kde = som.initiate_neuron(data)
    logging.info(f"Neuron initialization with 'kde': {neurons_kde.shape}")

    som.init_method = "kmeans++"
    neurons_kmeans = som.initiate_neuron(data)
    logging.info(f"Neuron initialization with 'kmeans++': {neurons_kmeans.shape}")

    som.init_method = "SOM++"
    neurons_sompp = som.initiate_neuron(data)
    logging.info(f"Neuron initialization with 'SOM++': {neurons_sompp.shape}")

def test_fit():
    # Testing the fit method
    data = np.random.rand(200, 3)  # Sample data with 200 samples and 3 features
    som = SOM(m=5, n=5, dim=3, initiate_method="random", learning_rate=0.5, neighbour_rad=2, distance_function="euclidean", max_iter=1000)
    som.fit(data, epoch=10, shuffle=True, verbose=True)
    logging.info("SOM training completed.")

def test_predict():
    # Testing the predict method
    data = np.random.rand(200, 3)
    som = SOM(m=5, n=5, dim=3, initiate_method="random", learning_rate=0.5, neighbour_rad=2, distance_function="euclidean", max_iter=1000)
    som.fit(data, epoch=10, shuffle=True)
    predictions = som.predict(data)
    logging.info(f"Predictions: {predictions}")

def test_fit_predict():
    # Testing fit_predict method
    data = np.random.rand(200, 3)
    som = SOM(m=5, n=5, dim=3, initiate_method="random", learning_rate=0.5, neighbour_rad=2, distance_function="euclidean", max_iter=1000)
    predictions = som.fit_predict(data, epoch=10, shuffle=True, verbose=True)
    logging.info(f"Fit and predict completed. Predictions: {predictions}")

def test_evaluate():
    # Testing evaluation methods
    data = np.random.rand(200, 3)
    som = SOM(m=5, n=5, dim=3, initiate_method="random", learning_rate=0.5, neighbour_rad=2, distance_function="euclidean", max_iter=1000)
    som.fit(data, epoch=10, shuffle=True)
    scores = som.evaluate(data, method=["silhouette", "davies_bouldin", "calinski_harabasz", "dunn"])
    logging.info(f"Evaluation Scores: {scores}")

def test_save_load():
    # Testing save and load methods
    data = np.random.rand(200, 3)
    som = SOM(m=5, n=5, dim=3, initiate_method="random", learning_rate=0.5, neighbour_rad=2, distance_function="euclidean", max_iter=1000)
    som.fit(data, epoch=10, shuffle=True)
    
    som.save("som_model.pkl")
    logging.info("SOM model saved.")

    loaded_som = SOM.load("som_model.pkl")
    logging.info("SOM model loaded. Checking attributes...")
    logging.info(f"Initial Learning Rate: {loaded_som.initial_learning_rate}")
    logging.info(f"Current Neighbourhood Radius: {loaded_som.cur_neighbour_rad}")

def test_with_iris():
    # Load and standardize the Iris dataset
    iris = load_iris()
    X = iris.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize and train the SOM
    som = SOM(m=10, n=10, dim=4, initiate_method="random", learning_rate=0.5, neighbour_rad=2, distance_function="euclidean", max_iter=1000)
    som.fit(X_scaled, epoch=100, shuffle=True, verbose=True)
    
    # Predict cluster labels
    predictions = som.predict(X_scaled)
    logging.info(f"Predicted clusters for Iris dataset: {predictions}")

    # Evaluate the SOM model on the Iris dataset
    scores = som.evaluate(X_scaled, method=["silhouette", "davies_bouldin", "calinski_harabasz", "dunn"])
    logging.info(f"Evaluation scores for Iris dataset: {scores}")

def clust_test():
    # Run all tests
    logging.info("Testing SOM Clustering Model")
    logging.info("Testing initiate_neuron...")
    test_initiate_neuron()

    logging.info("Testing fit...")
    test_fit()

    logging.info("Testing predict...")
    test_predict()

    logging.info("Testing fit_predict...")
    test_fit_predict()

    logging.info("Testing evaluate...")
    test_evaluate()

    logging.info("Testing save and load...")
    test_save_load()

    logging.info("Testing with Iris dataset...")
    test_with_iris()

if __name__ == "__main__":
    main()
