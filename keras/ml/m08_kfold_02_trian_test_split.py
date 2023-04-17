import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import all_estimators    # all_estimators : 모든 모델에 대한 평가 (분류 41개 모델)
import sklearn as sk


path_ddarung = './_data/ddarung/'
path_kaggle_train = './_data/kaggle_bike/'

ddarung_data= pd.read_csv(path_ddarung + 'train.csv', index_col=0).dropna()
kaggle_data= pd.read_csv(path_kaggle_train + 'train.csv', index_col=0).dropna()

#1. 데이터
datasets = [load_diabetes(return_X_y=True),
            fetch_california_housing(return_X_y=True),
            ddarung_data,
            kaggle_data
            ]

data_name= ['디아벳', '캘리포니아', '따릉', '캐글']

n_splits= 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 123)



#1. 데이터
for index, value in enumerate(datasets):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size = 0.8, shuffle= True, random_state=123
    )
    
    
    #2. 모델구성
    allAlgorithms= all_estimators(type_filter= 'regressor')

    max_score = 0
    max_name = 'max_model'
    

    for (name, algorithm) in allAlgorithms:
        try:
            model= algorithm()
            
            scores = cross_val_score(model, x_train, y_train, cv=kfold) 
            mean = round(np.mean(scores), 4)
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = round(accuracy_score(y_test, y_predict), 4)
            
            if max_score < mean:
                max_score = mean
                max_name = name
        except:
            continue
    print("=============", data_name[index], "================")
    print('최고모델:', max_name, '\nmean_acc:', max_score, '\nprd_acc:', acc)
    print("=============================")



# ============= 디아벳 ================
# 최고모델: max_model
# mean_acc: 0
# prd_acc: 0.0
# =============================