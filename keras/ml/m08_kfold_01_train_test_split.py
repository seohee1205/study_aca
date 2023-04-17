import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import all_estimators
import sklearn as sk



#1. 데이터
datasets = [load_iris(return_X_y=True),
            load_breast_cancer(return_X_y=True),
            load_wine(return_X_y=True),
            load_digits(return_X_y=True),
            ]

data_name= ['아이리스', '캔서', '와인', '디지트']

n_splits= 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 123)


#1. 데이터
for index, value in enumerate(datasets):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, shuffle= True, random_state=123
    )
    
    
    #2. 모델구성
    allAlgorithms= all_estimators(type_filter= 'classifier')

    max_score = 0
    max_name = '바보'
    max_acc = 0
    max_acc_name = '바보'

    for (name, algorithm) in allAlgorithms:
        try:
            model= algorithm()
            
            scores = cross_val_score(model, x_train, y_train, cv=kfold)  #n_jobs= -1
            results = cross_val_score(model, x_train, y_train, cv=kfold)
            if max_score < results:
                max_score = results
                max_name = name  
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = accuracy_score(y_test, y_predict)
            print(name, 'acc : ', acc)
            if max_acc < acc:
                max_acc = acc
                max_acc_name = name
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = accuracy_score(y_test, y_predict)
            print(name, 'acc : ', acc)    
        except:
            continue
    print("=============", data_name[index], "================")
    print('최고모델 : ', max_name, max_score)
    print("=============================")


