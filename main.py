from modules.som import SOM
from modules.som_classification import som_classification
import pandas as pd
import time

if __name__ == "__main__":
    df = pd.read_csv("data.csv", header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:,-1].values
    X.shape, y.shape
    m = 2
    n = 2
    init1 = "random"
    init2 = "SOM++"
    init3 = "kde"
    epochs = 1000
    
    model1 = SOM(m=m,n=n,dim=X.shape[1], initiate_method=init1, neighbour_rad=0.1,learning_rate=0.5,distance_function="euclidean")
    cur = time.time()
    model1.fit_predict(X, epochs, verbose=True)
    ex_time = time.time() - cur
    model2 = SOM(m=m,n=n,dim=X.shape[1], initiate_method=init2, neighbour_rad=0.1,learning_rate=0.5,distance_function="euclidean")
    cur = time.time()
    model2.fit_predict(X, epochs, verbose=True)
    ex_time2 = time.time() - cur
    #model3 = SOM(m=m,n=n,dim=X.shape[1], initiate_method=init3, neighbour_rad=0.1,learning_rate=0.1,distance_function="euclidean")
    cur = time.time()
    #model3.fit_predict(X, 10, verbose=False)
    ex_time3 = time.time() - cur
    print("time for", init1, "neurons:", ex_time)
    print("time for", init2, "neurons", ex_time2)
    print("time for", init3, "neurons", ex_time3)
    print("time difference:", (ex_time2)/ex_time*100 - 100,"%")
    #print("time difference:", (ex_time3)/ex_time*100 - 100,"%")
    print("eval for model 1")
    print(model1.evaluate(X=X, method=["davies_bouldin", "silhouette", "calinski_harabasz", "dunn"]))
    print(model2.evaluate(X=X, method=["davies_bouldin", "silhouette", "calinski_harabasz", "dunn"]))
    #print(model3.evaluate(X=X, method="all"))
    
    classification_model = som_classification(2,2,X,y)
    classification_model.train()