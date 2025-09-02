import pandas as pd
from sklearn import tree

## import .csv file
src = pd.read_csv("Dataset-Titanic/train.csv")

## test-for predict         # .iloc[6] จะได้มาเป็น Series ณ ตรงนี้เราจะ train ต้องเป็น dataframe
test_data = src[["Age","Sex"]].iloc[[6]].copy()   # crop มาแค่ Mr. Timothy J  อายุ 54 เพศ ชาย ไม่รอด

## test-prepare
test_data["Sex"] = test_data["Sex"].map({"male":0,"female":1})
## test_data["Sex"] = test_data["Sex"].map({"male":0,"female":1}) // ทำ .map ได้แค่ตอนมีหลายๆตัว เรียกว่า colums รึป่าววะ //ตรงนี้เรามี test_data เป็น dataframe แล้ว .map ได้เลย
## test_data["Sex"] = {"male":0,"female":1}[test_data["Sex"]]  # ทำให้ เพศเป็น int เพราะยัด string เข้า model_train ไม่ได้

## Train-prepare
train_data = src.drop(src.index[6]).loc[:,["Age","Sex"]].copy()     # เรา need to .copy() เพราะว่าเราจะเปลี่ยนค่าใน dataframe ตรงๆไม่ได้
train_data["Sex"] = train_data["Sex"].map({"male":0,"female":1})    # same we need to casting lol
train_target = src.drop(src.index[6]).loc[:,"Survived"]     #.loc[จะต้องใส่เป็น index, ชื่อ colums] พอไปใช้ .loc["Survived"] มันเป็นแค่ชื่อ colums มันไม่ได้เว้ย

## Model
model_train = tree.DecisionTreeClassifier()
model_train.fit(train_data,train_target)
print(model_train.predict(test_data))

# // just for checking
# print(test_data)
# print(src.iloc[[6]])
# print(train_data.index[6])
# print(type(test_data))
# print(test_data)
# print(train_target.head())