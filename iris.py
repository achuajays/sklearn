from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics , neighbors
from sklearn.preprocessing import LabelEncoder
from p.vtok import find_key_by_value
import pandas as pd
import numpy as np
import os

n = pd.read_csv('Iris.csv')
x = n[[
    'SepalLengthCm',
    'SepalWidthCm',
    'PetalLengthCm',
    'PetalWidthCm'
]]

y = n[['Species']]

l = LabelEncoder()
# for i in range(len(x[0])):
#     x[:,i] = l.fit_transform(x[:,i])
x = x.apply(LabelEncoder().fit_transform)

label = {
    'Iris-setosa' : 0,
    'Iris-versicolor' : 1,
    'Iris-virginica' : 2 
}

y['Species'] = y['Species'].map(label)
x_t , x_te , y_t , y_te = train_test_split(x,y,test_size=0.2)


# print(type(n))

knn = neighbors.KNeighborsClassifier(n_neighbors=20 , weights='uniform')
knn.fit(x_t , y_t)

p = knn.predict(x_te)

acc = metrics.accuracy_score(y_te , p )
print(acc)
l = knn.predict([[5.0,3.3,1.4,0.2]])
l = l[0]
l = find_key_by_value(label , l)
os.system('cls')
print('predict = ' , l)
k = input()
os.system('cls')