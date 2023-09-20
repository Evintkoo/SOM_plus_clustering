import numpy as np
import math
from .som import SOM
from .utils import euc_distance
from .variables import METHOD_LIST

class model_picker:
    def __init__(self) -> None:
        self.models = []
        self.model_evaluation = []
    
    def pick_best_model(self):
        return self.models[np.argmax(self.model_evaluation)]
    
    def evaluate_initiate_method(self, X:np.array, m:int, n:int, learning_rate:float, neighbor_rad:int, max_iter:int=None, epoch:int = 1):
        for methods in METHOD_LIST:
            som_model = SOM(m=m,
                        n=n,
                        dim=X.shape[1],
                        initiate_method=methods,
                        learning_rate=learning_rate,
                        max_iter=max_iter,
                        neighbour_rad=neighbor_rad)
            som_model.fit(X=X, epoch=epoch, verbose=False)
            dist_matrix_evaluation = som_model.evaluate()
            self.models.append(som_model)
            self.model_evaluation.append(dist_matrix_evaluation)