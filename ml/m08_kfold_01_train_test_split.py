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
    allAlgorithms = all_estimators(type_filter='classifier')
    max_score = 0
    max_name = 'max_model'
    
    for (name, algorithm) in allAlgorithms:
        try: #예외처리
            model = algorithm()

            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            mean = round(np.mean(scores),4)
            # print('acc:', scores, '\ncross_val_score 평균:', mean)
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            acc = round(accuracy_score(y_test, y_predict),4)
            
            if max_score < mean: 
               max_score = mean
               max_name = name
        except:
            continue #continue: error 무시하고 계속 for문 돌리기 #break = for문 중단해라

    #dataset name , 최고모델, 성능
    print('========', data_name[index],'========')        
    print('최고모델:', max_name, '\nmean_acc:', max_score, '\nprd_acc:',acc)
    print('================================')  


'''
======== 아이리스 ========
최고모델: LinearDiscriminantAnalysis
mean_acc: 0.9917
prd_acc: 0.8
================================
======== 캔서 ========
최고모델: AdaBoostClassifier
mean_acc: 0.9714
prd_acc: 0.9123
================================
======== 와인 ========
최고모델: ExtraTreesClassifier
mean_acc: 0.9929
prd_acc: 0.6944
================================
======== 디지트 ========
최고모델: ExtraTreesClassifier
mean_acc: 0.9854
prd_acc: 0.9528
================================
'''

