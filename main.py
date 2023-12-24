from modules.som import SOM
from modules.model_picker import model_picker
import pandas as pd
import time

cur = time.time()
df = pd.read_csv("data.csv", header=None).iloc[:, :-1]
X = df.values
model1 = SOM(2,2,X.shape[1], "random", 0.1,0.1,"euclidean")
model1.fit_predict(X, 10)
model2 = SOM(2,2,X.shape[1], "random", 0.1,0.1,"cosine")
model2.fit_predict(X, 10)
print()
print(model1.cluster_center_)
print(model2.cluster_center_)
print(model1.evaluate("euclidean"))
print(model1.evaluate("cosine"))
print(model2.evaluate("euclidean"))
print(model2.evaluate("cosine"))
print(time.time()-cur)