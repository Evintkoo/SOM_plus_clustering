from modules.som import SOM
from modules.model_picker import model_picker
import pandas as pd
import time

if __name__ == "__main__":
    df = pd.read_csv("data.csv", header=None).iloc[:, :-1]
    X = df.values
    m1 = 2
    m2 = 2
    n1 = 2
    n2 = 2
    model1 = SOM(m=m1,n=n1,dim=X.shape[1], initiate_method="random", neighbour_rad=0.1,learning_rate=0.1,distance_function="euclidean")
    cur = time.time()
    model1.fit_predict(X, 10, verbose=False)
    ex_time = time.time() - cur
    model2 = SOM(m=m2,n=n2,dim=X.shape[1], initiate_method="SOM++", neighbour_rad=0.1,learning_rate=0.1,distance_function="cosine")
    cur = time.time()
    model2.fit_predict(X, 10, verbose=False)
    ex_time2 = time.time() - cur
    print("time for", m1*n1, "neurons:", ex_time)
    print("time for", m2*n2, "neurons", ex_time2)
    print("time difference:", (ex_time2-ex_time)/ex_time*100,"%")