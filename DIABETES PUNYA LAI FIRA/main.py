import sklearn
import numpy as np


#load the iris dataset
iris = sklearn.datasets.load_iris()

# print the shape of te data and the target arrays
print(f"Data shape: {iris.data.shape}")
print(f"Target shape: {iris.target.shape}")

# print the names of the target classes
print(f"Target names: {iris.target_names}")
