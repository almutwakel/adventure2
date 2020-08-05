from typing import Any, Union

import sklearn
from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.neighbors import  KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pickle
from matplotlib import style

data = pd.read_csv("adult.data", sep = ",")
data = data[["age", "work_sector", "edunum", "marital", "occupational", "family", "race", "sex", "gain", "loss", "hours", "nationality", "income"]]

# transform irregular data into integer types
le = preprocessing.LabelEncoder()
age = list(data["age"])
work_sector = le.fit_transform(list(data["work_sector"]))
edunum = list(data["edunum"])
marital = le.fit_transform(list(data["marital"]))
occupational = le.fit_transform(list(data["occupational"]))
family = le.fit_transform(list(data["family"]))
race = le.fit_transform(list(data["race"]))
sex = le.fit_transform((list(data["sex"])))
gain = list(data["gain"])
loss = list(data["loss"])
hours = list(data["hours"])
nationality = le.fit_transform(list(data["nationality"]))
income = le.fit_transform((data["income"]))

predict = "income"

x = list(zip(age, edunum, marital, work_sector, family,  race, sex, gain, loss, ))
y = income

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

best = 0

# preload saved model - breakpoint for model collection
"""
pickle_in = open("adultmodelKNN.pickle", "rb")
model = pickle.load(pickle_in)

# predict accuracy score of loaded model
best = model.score(x_test, y_test)
print("loaded",best)

for z in range(10):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    # loaded acc
    pickle_in = open("adultmodelKNN.pickle", "rb")
    model = pickle.load(pickle_in)
    best = model.score(x_test, y_test)
    print("loaded", best)

    # make model using K nearest neighbors
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(x_train, y_train)

    # test for accuracy
    acc = model.score(x_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        with open("adultmodelKNN.pickle", "wb") as f:
            pickle.dump(model, f)
"""

# linear regression - DO NOT USE

"""
linear = linear_model.LinearRegression()

#linear modeling
average = 0
best = 0
for _ in range(100):
    #create best fit line based on data

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    #accuracy score
    acc = linear.score(x_test, y_test)
    print(acc)
    average += acc
    # save it
    if acc > best:
        best = acc
        with open("adultmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
print("avg", average / 100)"""

# load saved KNN model

pickle_in = open("adultmodelKNN.pickle", "rb")
model = pickle.load(pickle_in)

# predict accuracy score of loaded model
acc = model.score(x, y)
print("Model accuracy is ",(int(acc * 10000000))/100000, "%", sep="")

# predict results of rest of data and print
predictions = model.predict(x_test)
names = ["income < 50K","income > 50K"]

for x in range(len(predictions)):
    print("Predicted",names[predictions[x]],"|","Data:",x_test[x],"|","Actual",names[y_test[x]])


"""
outcomes: >50K, <=50K.

data description:
age: continuous.
work_sector: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
"""