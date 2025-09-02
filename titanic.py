import sklearn.datasets as sklearn
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

iris =  sklearn.load_iris()
# print(iris)
train_target = iris.target
train_data   = iris.data

# print(test_target)
# print(test_data)
# lovePYTHON = np.zeros((1,3,1))
# lovePYTHON = [2,3,4]
# lovePYTHON = np.arange(6)
# print(type(lovePYTHON))
testpior = [0,50,100]
test_data = iris.data[testpior]

model = tree.DecisionTreeClassifier()
model.fit(train_data,train_target)
# print(model.predict(test_data))

lemme = model.predict(test_data)
# print(iris.keys()) #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
for i in lemme :
    print(iris.target_names[i])