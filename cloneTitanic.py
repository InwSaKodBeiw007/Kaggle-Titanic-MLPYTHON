import pandas as pd
from sklearn import tree

model_piod = pd.read_csv("Dataset-Titanic/train.csv")

train_data = model_piod[["Age","Sex"]].copy()
train_data["Sex"] = train_data["Sex"].map({"male":0,"female":1})
train_target = model_piod["Survived"]

test_data = model_piod[["Age","Sex"]].iloc[[2]].copy()
test_data["Sex"] = test_data["Sex"].map({"male":0,"female":1})

model = tree.DecisionTreeClassifier()
model.fit(train_data,train_target)

print(model.predict(test_data))
# print(model_piod.iloc[2])
# print(train_data)