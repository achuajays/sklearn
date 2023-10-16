import numpy as np
import pandas as pd 
from sklearn import neighbors , metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder

n = pd.read_csv('car_evaluation.csv')
# print(n)

x = n[[
    'buying',
    'maint',
    'safety'
]].values

y=n[['class']]



# label encoding

l = LabelEncoder()
for i in range(len(x[0])):
    x[:,i] = l.fit_transform(x[:,i])
# print(x)

label = {
    'unacc'  : 0,
    'acc' : 1 ,
    'good' : 2 ,
    'vgood' : 3
}

y['class'] = y['class'].map(label)

# print(y)

# text and train  


x_t , x_te , y_t , y_te = train_test_split(x,y,test_size=0.2)


# knn train 

knn = neighbors.KNeighborsClassifier(n_neighbors=25 , weights='uniform')

knn.fit(x_t , y_t.values.ravel())



# predict 

p  = knn.predict(x_te)




# accureacy 

a = metrics.accuracy_score(y_te , p)
print ( a )
aa = 1

# print('actual value = ',y[a])
print('predicted value ',knn.predict([[1,2,0]]))