from modules.som import SOM
from modules.model_picker import model_picker
import pandas as pd
import time

if __name__ == "__main__":
    cur = time.time()
    df = pd.read_csv("data.csv", header=None).iloc[:, :-1]
    X = df.values
    model1 = SOM(m=2,n=2,dim=X.shape[1], initiate_method="random", neighbour_rad=0.1,learning_rate=0.1,distance_function="euclidean")
    model1.fit_predict(X, 10)
    model2 = SOM(m=2,n=2,dim=X.shape[1], initiate_method="random", neighbour_rad=0.1,learning_rate=0.1,distance_function="cosine")
    model2.fit_predict(X, 10)
    print()
    print(model1.cluster_center_)
    print(model2.cluster_center_)
    print(model1.evaluate("euclidean"))
    print(model1.evaluate("cosine"))
    print(model2.evaluate("euclidean"))
    print(model2.evaluate("cosine"))
    print(time.time()-cur)