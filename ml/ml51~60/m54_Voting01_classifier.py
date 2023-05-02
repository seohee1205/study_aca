# 실습



import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype]
data_name_list = ['아이리스', '캔서', '디지트', '와인', '패치콥타입']

model_list = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]
model_name_list = ['로지스틱', '케이네이볼', '디시전트리']

lr = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=8)
dt = DecisionTreeClassifier()

for i, v in enumerate(data_list):
    x, y = v(return_X_y= True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size= 0.8, random_state=123, shuffle=True
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    list =[]
    for j, v2 in enumerate(model_list):
        
        model = VotingClassifier(
            estimators=[('LR', lr), ('KNN', knn), ('DT', dt)],
        )
        classifiers = [lr, knn, dt]
        for model2 in classifiers:
            model2.fit(x_train, y_train)
            y_predict = model2.predict(x_test)
            score2 = accuracy_score(y_test, y_predict)
            class_name = model2.__class__.__name__
            print('==========', data_name_list[i], '==========')
            print("{0} 정확도 : {1:.4f}".format(class_name, score2))
            # print('model.score : ', model.score(x_test, y_test))
            print('voting acc : ', accuracy_score(y_test, y_predict))
            list.append(score2)
            

