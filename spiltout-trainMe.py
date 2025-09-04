import pandas as pd
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# pointed to train.csv
src = pd.read_csv("Dataset-Titanic/train.csv")
#   checking by graph i think this so fine to use let try another time
# sns.countplot(x="Sex",data=src)
# plt.show()



#   cleansing data      data = data, test = target
locked_data = src[["Sex","Age"]]
locked_data = pd.get_dummies(locked_data,columns=["Sex"])
locked_test = src[["Survived"]]
#   เราจะให้ target มีแค่ 0,1 ด้วยความต้องการของ GU
# locked_test = pd.get_dummies(locked_test,columns=["Survived"]) # get_dummies( จำเป็นต้องมี ,columns=["ด้วยยยย"])

#   split-out for train and test like @inoutTestTitanic.py, But this way better Cool asf!
x = locked_data
y = locked_test
#      mentioned ตรงนี้ data,data,  target,target
data_train_X,data_test_Y,target_train_X,target_test_Y = train_test_split(x,y,test_size=0.33,random_state=42)    # mentioned importมาจาก sklearn.model_selection
#                     train_test_split("data","target",test_size=0.33,random_state=42)    mentioned test_size= , random_state= 

#   DecisionTreeClassifier
model = tree.DecisionTreeClassifier()
model.fit(data_train_X,target_train_X)

#   prediction  go get percent% for me
# print(f"Y{y_test}   ")
# print(model.predict(data_test_Y))
pppdd = model.predict(data_test_Y)

print(model.score(data_test_Y,target_test_Y))
acc = accuracy_score(pppdd,target_test_Y)
print(acc)




# for checking Face Datas
# print(type(locked_data))
# print(locked_data.head())
# print(y.head())
# print(f"{x_train.head(3)}   X_train = ")
# print(f"{x_test.head(3)}    X_test =")
# print(f"{y_train.head(3)}   Y_train =")
# print(f"{y_test.head(3)}    Y_test =")   # hard to read but okay
# print(x_train.tail())
# print(x_test.tail())
# print(type(x_train))
# print(type(x_test))