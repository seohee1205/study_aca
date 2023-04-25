# n_component > 0.95 이상
# xgboost, gridSearch 또는 RandomSearch를 쓸 것

# m33_2 결과를 뛰어넘을 것

# parameters = {
#     {"_estimators": [100, 200, 300],
#      "learning_rate": [0.1, 0.3, 0.001, 0.01],
#     "max_depth": [4, 5, 6]},
#     {"_estimators": [90, 100, 110],
#     "learning_rate": [0.1, 0.001, 0.01],
#     "max _depth": [4,5,6],
#     "colsample_bytree": [0.6, 0.9, 1]},
#     {"_estimators": [90, 110],
#     "learning rate": [0.1, 0.001, 0.5],
#     "max _depth": [4,5,6],
#     "colsample _bytree": [0.6, 0.9, 1]},
#     {"colsample_bylevel": [0.6, 0.7, 0.9]}
# }
# n_jobs = -1
# tree_method ='gpu_hist'
# predictor ='gpu_predictor'
# gpu_id =0,

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
#n_conponent > 0.95 이상
# xgboost, girdSearch 또는 RandomSearch 를 쓸것

#m33_2 결과를 뛰어넘기!

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)

pca = PCA(n_components= 9)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

parameters = [
    {"n_estimators": [100,200,300], "learning_rate" : [0.1, 0.3, 0.001, 0.01],
    "max_depth":[4,5,6]},
    {"n_estimators": [90,100,110], "learning_rate" : [0.1, 0.001, 0.01],
    "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1]},
    {"n_estimators": [90,110], "learning_rate" : [0.1, 0.001, 0.5],
    "max_depth":[4,5,6], "colsample_bytree":[0.6, 0.9, 1],
    "colsample_bylevel":[0.6,0.7,0.9]}
]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=413)

#n_jobs = -1
#tree_method = 'gpu_hist'
#predictor = gpu_predictor
#gpu_id = 0
    
#2.모델
model = RandomizedSearchCV(XGBClassifier(),parameters,
                        n_jobs=-1,
                        verbose=1,
                        refit=True,
                        n_iter=10,
                        cv=kfold)

# 3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test,y_test)
print("acc : ", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test,y_predict)
print("accuracy_score : ", acc)


# acc :  0.9141
# accuracy_score :  0.9141