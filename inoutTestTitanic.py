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
# print(train_data.head())
# print(type(train_data.head()))

for i,zeries in train_data.head().iterrows():
    # model_train.predict([zeries])
    print(model_train.predict(train_data.iloc[[i]]))

    # print(zeries)       # type = series ใช้ predict ไม้ได้  :  Age    22.0 Sex     0.0  Name: 0, dtype: float64
    # print([zeries])     # type = list                   : [Age    22.0    Sex     0.0 Name: 0, dtype: float64]
    # print(type(zeries))     # <class 'pandas.core.series.Series'>
    # print(type([zeries]))  # ตรงนี้ มาเป็น list แต่ใช้ไม่ได้ #   # <class 'list'>

    # print(type([zeries.values]))    # <class 'list'> เช่นกัน
    # scenario = model_train.predict([zeries.values]) # ทดลอง เอาแทนค่ามารับไว้
    # print(scenario) # บ่ได้

    #print(zeries.values.reshape(1,-1))
    #print(type(zeries.values.reshape(1,-1))) # [[22.  0.]] <class 'numpy.ndarray'> ควรจะได้หรือไม่? แต่ว่ามันก็ไม่ได้
    # spt =  model_train.predict(zeries.values.reshape(1,-1))
    # print(spt)

    # model_train.predict(train_data.loc[i:]) # จบข่าว 
    # print(train_data.loc[i])    #   Age    22.0 Sex     0.0 Name: 0, dtype: float64
    # print(type(train_data.loc[i]))# <class 'pandas.core.series.Series'>

    # i found it! 
    # print(train_data.loc[i:])   #  Age  Sex    0  22.0    0  [890 rows x 2 columns] หลายตัวไปหน่อย
    # print(type(train_data.loc[i:]))#    <class 'pandas.core.frame.DataFrame'> เนี่ย มันเป็น Dataframe เว้ย ที่มันต้องการอะ 
    # print(train_data.iloc[[i]])   #       Age  Sex    0  22.0    0 ตัวเดียวในแต่ละ loop
    # print(type(train_data.iloc[[i]]))#  <class 'pandas.core.frame.DataFrame'>
    # model_train.predict(train_data.iloc[[i]])   # its! work

    # gpt give
    # row = zeries.to_frame().T  # แปลง Series → DataFrame 1 แถว
    # prediction = model_train.predict(row)
    # print(prediction)

    # print(type(row))    #   <class 'pandas.core.frame.DataFrame'>
    # print(row)  #    Age  Sex   0  22.0  0.0

#   check if what wrong :: how ever THIS IS OVERFITTING i mean not sure to be sure
print(src.drop(6).head())   # everything alright