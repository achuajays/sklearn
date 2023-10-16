from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

n = datasets.load_iris()

# split data with label and feature 

x = n.data
y = n.target

print(x.shape)
print(y.shape)

# test and train with train - 80  and test - 20 

x_t , x_te , y_t , y_te = train_test_split(x,y,test_size=0.2)

print(x_t.shape , x_te.shape )
print(y_t.shape , y_te.shape)