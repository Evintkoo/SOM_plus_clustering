import numpy as np
import logging
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# Assuming the SOM class and other required components are available in the same directory
from modules.som_classification import SOM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def class_test():
    # Create synthetic data
    X, y = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize SOM instance
    som = SOM(m=3, n=3, dim=2, initiate_method='random', learning_rate=0.5, neighbour_rad=2, distance_function='euclidean', max_iter=1000)
    
    # Fit the SOM
    logger.info("Fitting the SOM model...")
    som.fit(X_train, y_train, epoch=10, shuffle=True, verbose=True, n_jobs=-1)
    
    # Predict the labels for test data
    logger.info("Predicting the cluster labels...")
    y_pred = som.predict(X_test)
    assert y_pred is not None, "Prediction failed, y_pred is None"
    assert len(y_pred) == len(X_test), "Prediction length mismatch"
    
    # Fit and predict simultaneously
    logger.info("Fitting and predicting in one go...")
    y_pred_combined = som.fit_predict(X_train, y_train, epoch=10, shuffle=True, verbose=True, n_jobs=-1)
    assert y_pred_combined is not None, "Fit and predict failed, y_pred_combined is None"
    assert len(y_pred_combined) == len(X_train), "Fit and predict length mismatch"

    # Evaluate the SOM
    logger.info("Evaluating the SOM...")
    evaluation_metrics = som.evaluate(X_test, y_test, method=['accuracy', 'f1_score', 'recall'])
    assert isinstance(evaluation_metrics, (list, dict)), "Evaluation metrics type mismatch"
    assert len(evaluation_metrics) > 0, "Evaluation metrics are empty"

    # Save and Load the SOM
    logger.info("Saving the SOM model...")
    som.save("som_model.pkl")
    
    logger.info("Loading the SOM model...")
    loaded_som = SOM.load("som_model.pkl")
    logger.info("Loaded SOM model, predicting again...")
    y_pred_loaded = loaded_som.predict(X_test)
    assert y_pred_loaded is not None, "Prediction with loaded model failed, y_pred_loaded is None"
    assert len(y_pred_loaded) == len(X_test), "Prediction length mismatch for loaded model"

if __name__ == "__main__":
    test_som_functionality()