from modules.som import SOM
from modules.model_picker import model_picker
import pandas as pd
import time

cur = time.time()
df = pd.read_csv("data.csv", header=None).iloc[:, :-1]
X = df.values
picker = model_picker()
picker.evaluate_initiate_method(X, 2,2,0.5,1,10)
print(picker.pick_best_model())
print(picker.model_evaluation)
print(time.time() - cur)