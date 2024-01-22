# SOM Clustering Library


## Installation

``git clone https://github.com/Evintkoo/SOM_plus_clustering.git``

## 1. Import Library

``from modules.som import SOM``

## 2. Create an SOM object

``model = SOM(m=2,n=2,dim=X.shape[1], initiate_method="random", neighbour_rad=0.1,learning_rate=0.1,distance_function="euclidean")``

