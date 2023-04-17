import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils import all_estimators
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
for i in range(len(datasets)):
    if i < 2:
       x, y = datasets[i]
    elif i ==2:
        x = ddarung_data.drop(['count'], axis = 1) 
        y = ddarung_data['count']
    else:
        x = kaggle_data.drop(['count', 'casual', 'registered'], axis = 1)
        y = kaggle_data['count']
    
    #2. 모델구성
    allAlgorithms= all_estimators(type_filter= 'regressor')

    max_score = 0
    max_name = '바보'

    for (name, algorithm) in allAlgorithms:
        try:
            model= algorithm()
            
            scores = cross_val_score(model, x, y, cv=kfold)  #n_jobs= -1
            results = round(np.mean(scores), 4)

            if max_score < results:
                max_score = results
                max_name = name
        except:
            continue
    print("=============", data_name[i], "================")
    print('최고모델 : ', max_name, max_score)
    print("==================================")

# ============= 디아벳 ================
# 최고모델 :  LinearRegression 0.4832  
# ==================================


