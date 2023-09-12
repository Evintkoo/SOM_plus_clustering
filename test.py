from som import SOM
import pandas as pd
import time

df = pd.read_csv("data.csv", header=None).iloc[:, :-1]
X = df.values
methods = ["random", "kde", "kmeans", "kde_kmeans", "kmeans++", "SOM++"]
for i in methods:
    starttime = time.time()
    model = SOM(2,2,X.shape[1],initiate_method=i, learning_rate=1, neighbour_rad=1)
    pred = model.fit_predict(X=X, epoch=1, verbose=False)
    print(i, "method executed in", time.time()-starttime)
    