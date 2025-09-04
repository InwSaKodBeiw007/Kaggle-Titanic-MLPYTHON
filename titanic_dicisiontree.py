from sklearn import tree
import pandas as pd
from sklearn.metrics import accuracy_score



####   import Titanic Datasets

src_train = pd.read_csv("Dataset-Titanic/train.csv")
src_test = pd.read_csv("Dataset-Titanic/test.csv")


####   frist check
# print(src_test.info())
# print(src_train.info())



####   let the me pointed some features this is also part of cleansing data

pointed_train = src_train[["Sex","Age"]]
pointed_test = src_test[["Sex","Age"]]
target_pointed = src_train["Survived"]  # Series or Dataframe can be the target i've tried or u can try urself

pointed_train = pd.get_dummies(pointed_train,columns=["Sex"])
pointed_train["Age"] = pointed_train["Age"].fillna(pointed_train["Age"].mean()) # i think this can improve, something idk. / i dunno if i just let it N/a can better but this time.
# pointed_train["Cabin"] = pointed_train["Cabin"].fillna(pointed_train["Cabin"].mean()) ## i try to make chimera 5555



####   second check
# print(pointed_train.head(15))    # i will let Cabin N/A   ## i tried io run it but "Cabin" can't be data its have text inside i will have a look how to deal with these type of dataframe
# print(type(pointed_train))
# print(target_pointed.head(4))
# print(type(target_pointed)) #   Series no dataframe



####   Model Creationn!!

model_titanic_found_dead_count = tree.DecisionTreeClassifier()
model_titanic_found_dead_count.fit(pointed_train,target_pointed)


#### we need to cleansing pointed_test like we do on pointed_train because i stuck with this fkkkk!

pointed_test = pd.get_dummies(pointed_test,columns=["Sex"]) # i will let another feature N/a lol



####   Make to see percent(%) of your model with another datasets here Named "src_test"

predict = model_titanic_found_dead_count.predict(pointed_test)
# acc = accuracy_score(predict," here we need feature Survived of src_test")    #   so we can't figure it out now ><
#                                                                               #    cause src_test have no Survived feature
print(type(predict[:10])) #   [0 0 0 0 1 0 1 0 1 0]   cool right!?


#### Make .csv file to summission my model 

ml_certificate = pd.DataFrame({
    "PassengerId": src_test["PassengerId"],
    "Survived": predict
})
# print(ml_certificate.head(11))
# print(type(ml_certificate))


## export here
ml_certificate.to_csv("Dataset-Titanic/submissionmyscore.csv", index=False)
# print("Now you need to update your model score below HERE")
#   0.722 scoreboard


####   how can i Graph my model TT