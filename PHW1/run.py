import numpy as np
import os
from sklearn.datasets import fetch_openml
from phw1 import *

print("Reading data...")
X, y = fetch_openml("mnist_784", return_X_y=True)

# Convert dataframe to ndarray
X = X.to_numpy(copy=True)
y = y.to_numpy(copy=True).astype(np.int8)

print("Data size: ", X.shape)
print("Target size: ", y.shape)

if not os.path.exists("./fig"):
    os.makedirs("./fig")

# Q1(X)
Q2(X, y)
Q3(X, y)
Q4(X, y)
Q5(X)
Q6(X)
# Q7(X, y)
