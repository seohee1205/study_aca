import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine, fetch_covtype
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

#1. 데이터
x, y = fetch_covtype(return_X_y=True)

y = y - 1  # 클래스 레이블을 0부터 시작하는 정수로 변환

print(x.shape[1])   # 컬럼 수: 54

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 337, train_size= 0.8,  stratify=y, 
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {'n_estimators' : 1000,               # = epochs
              'learning_rate' : 0.3,
              'max_depth' : 2,
              'gamma' : 0,
              'min_child_weight' : 0,
              'subsample' : 0.2,                     # dropout
              'colsample_bytree' : 0.5,
              'colsample_bylevel' : 0,
              'colsample_bynode' : 0,
              'reg_alpha' : 1,                    # 절대값: 레이어에서 양수만들겠다/ 라쏘 / 머신러닝 모델
              'reg_lambda' : 1,                   # 제곱: 레이어에서 양수만들겠다/ 리지   / 머신러닝 모델
              'random_state' : 337,
              # 'verbose' : 0
}

model = XGBClassifier()
model.set_params(**parameters,
                 early_stopping_rounds = 10,
                 eval_metric = 'merror')

model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=0
          )
results = model.score(x_test, y_test)
print("최종점수 :", results)

y_predict = model.predict(x_test)
acc =  accuracy_score(y_test, y_predict)
print("acc", acc)

mse = mean_squared_error(y_test, y_predict)
print("RMSE : ", np.sqrt(mse))



####################################################
# print(model.feature_importances_)
# [0.1204448  0.03181682 0.07185043 0.10543638 0.08690013 0.0775187
#  0.14735512 0.06484857 0.13835293 0.15547612]
thresholds = np.sort(model.feature_importances_)
# print(thresholds)
# [0.03181682 0.06484857 0.07185043 0.0775187  0.08690013 0.10543638
#  0.1204448  0.13835293 0.14735512 0.15547612]

from sklearn.feature_selection import SelectFromModel

for i in thresholds:
    selection = SelectFromModel(model, threshold=i, prefit= True)    # False면 다시 훈련

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print('변형된 x_train : ', select_x_train.shape, '변형된 x_test : ', select_x_test.shape)

    selection_model = XGBClassifier()
    selection_model.set_params(early_stopping_rounds= 10, **parameters,
                               eval_metric = 'merror'
                               )
    selection_model.fit(select_x_train, y_train,
          eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
          verbose= 0)
    
    
    select_y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    
    print("Thres=%.3f, n= %d, R2: %.2f%%" %(i, select_x_train.shape[1], score*100))
    # 소수 3번째 자리까지 숫자를 넣어라 
    

# 최종점수 : 0.6331592127569856
# acc 0.6331592127569856
# RMSE :  1.4812349415464234
# 변형된 x_train :  (464809, 54) 변형된 x_test :  (116203, 54)
# Thres=0.000, n= 54, R2: 63.32%
# 변형된 x_train :  (464809, 54) 변형된 x_test :  (116203, 54)
# Thres=0.000, n= 54, R2: 63.32%
# 변형된 x_train :  (464809, 54) 변형된 x_test :  (116203, 54)
# Thres=0.000, n= 54, R2: 63.32%
# 변형된 x_train :  (464809, 54) 변형된 x_test :  (116203, 54)
# Thres=0.000, n= 54, R2: 63.32%
# 변형된 x_train :  (464809, 54) 변형된 x_test :  (116203, 54)
# Thres=0.000, n= 54, R2: 63.32%
# 변형된 x_train :  (464809, 54) 변형된 x_test :  (116203, 54)
# Thres=0.000, n= 54, R2: 63.32%
# 변형된 x_train :  (464809, 54) 변형된 x_test :  (116203, 54)
# Thres=0.000, n= 54, R2: 63.32%
# 변형된 x_train :  (464809, 47) 변형된 x_test :  (116203, 47)
# Thres=0.000, n= 47, R2: 71.65%
# 변형된 x_train :  (464809, 46) 변형된 x_test :  (116203, 46)
# Thres=0.000, n= 46, R2: 74.17%
# 변형된 x_train :  (464809, 45) 변형된 x_test :  (116203, 45)
# Thres=0.000, n= 45, R2: 74.00%
# 변형된 x_train :  (464809, 44) 변형된 x_test :  (116203, 44)
# Thres=0.000, n= 44, R2: 72.63%
# 변형된 x_train :  (464809, 43) 변형된 x_test :  (116203, 43)
# Thres=0.000, n= 43, R2: 72.87%
# 변형된 x_train :  (464809, 42) 변형된 x_test :  (116203, 42)
# Thres=0.000, n= 42, R2: 73.90%
# 변형된 x_train :  (464809, 41) 변형된 x_test :  (116203, 41)
# Thres=0.000, n= 41, R2: 73.75%
# 변형된 x_train :  (464809, 40) 변형된 x_test :  (116203, 40)
# Thres=0.001, n= 40, R2: 73.98%
# 변형된 x_train :  (464809, 39) 변형된 x_test :  (116203, 39)
# Thres=0.001, n= 39, R2: 72.88%
# 변형된 x_train :  (464809, 38) 변형된 x_test :  (116203, 38)
# Thres=0.001, n= 38, R2: 52.61%
# 변형된 x_train :  (464809, 37) 변형된 x_test :  (116203, 37)
# Thres=0.001, n= 37, R2: 73.81%
# 변형된 x_train :  (464809, 36) 변형된 x_test :  (116203, 36)
# Thres=0.001, n= 36, R2: 73.47%
# 변형된 x_train :  (464809, 35) 변형된 x_test :  (116203, 35)
# Thres=0.002, n= 35, R2: 62.46%
# 변형된 x_train :  (464809, 34) 변형된 x_test :  (116203, 34)
# Thres=0.002, n= 34, R2: 73.94%
# 변형된 x_train :  (464809, 33) 변형된 x_test :  (116203, 33)
# Thres=0.002, n= 33, R2: 72.35%
# 변형된 x_train :  (464809, 32) 변형된 x_test :  (116203, 32)
# Thres=0.002, n= 32, R2: 73.43%
# 변형된 x_train :  (464809, 31) 변형된 x_test :  (116203, 31)
# Thres=0.003, n= 31, R2: 70.98%
# 변형된 x_train :  (464809, 30) 변형된 x_test :  (116203, 30)
# Thres=0.003, n= 30, R2: 62.71%
# 변형된 x_train :  (464809, 29) 변형된 x_test :  (116203, 29)
# Thres=0.003, n= 29, R2: 74.55%
# 변형된 x_train :  (464809, 28) 변형된 x_test :  (116203, 28)
# Thres=0.004, n= 28, R2: 73.96%
# 변형된 x_train :  (464809, 27) 변형된 x_test :  (116203, 27)
# Thres=0.004, n= 27, R2: 73.10%
# 변형된 x_train :  (464809, 26) 변형된 x_test :  (116203, 26)
# Thres=0.005, n= 26, R2: 74.05%
# 변형된 x_train :  (464809, 25) 변형된 x_test :  (116203, 25)
# Thres=0.005, n= 25, R2: 74.29%
# 변형된 x_train :  (464809, 24) 변형된 x_test :  (116203, 24)
# Thres=0.006, n= 24, R2: 72.28%
# 변형된 x_train :  (464809, 23) 변형된 x_test :  (116203, 23)
# Thres=0.007, n= 23, R2: 73.41%
# 변형된 x_train :  (464809, 22) 변형된 x_test :  (116203, 22)
# Thres=0.009, n= 22, R2: 73.23%
# 변형된 x_train :  (464809, 21) 변형된 x_test :  (116203, 21)
# Thres=0.009, n= 21, R2: 72.31%
# 변형된 x_train :  (464809, 20) 변형된 x_test :  (116203, 20)
# Thres=0.011, n= 20, R2: 72.10%
# 변형된 x_train :  (464809, 19) 변형된 x_test :  (116203, 19)
# Thres=0.011, n= 19, R2: 71.26%
# 변형된 x_train :  (464809, 18) 변형된 x_test :  (116203, 18)
# Thres=0.011, n= 18, R2: 71.17%
# 변형된 x_train :  (464809, 17) 변형된 x_test :  (116203, 17)
# Thres=0.011, n= 17, R2: 71.61%
# 변형된 x_train :  (464809, 16) 변형된 x_test :  (116203, 16)
# Thres=0.013, n= 16, R2: 72.45%
# 변형된 x_train :  (464809, 15) 변형된 x_test :  (116203, 15)
# Thres=0.016, n= 15, R2: 72.24%
# 변형된 x_train :  (464809, 14) 변형된 x_test :  (116203, 14)
# Thres=0.018, n= 14, R2: 69.80%
# 변형된 x_train :  (464809, 13) 변형된 x_test :  (116203, 13)
# Thres=0.022, n= 13, R2: 67.57%
# 변형된 x_train :  (464809, 12) 변형된 x_test :  (116203, 12)
# Thres=0.022, n= 12, R2: 68.80%
# 변형된 x_train :  (464809, 11) 변형된 x_test :  (116203, 11)
# Thres=0.023, n= 11, R2: 68.72%
# 변형된 x_train :  (464809, 10) 변형된 x_test :  (116203, 10)
# Thres=0.023, n= 10, R2: 68.20%
# 변형된 x_train :  (464809, 9) 변형된 x_test :  (116203, 9)
# Thres=0.024, n= 9, R2: 68.03%
# 변형된 x_train :  (464809, 8) 변형된 x_test :  (116203, 8)
# Thres=0.024, n= 8, R2: 67.82%
# 변형된 x_train :  (464809, 7) 변형된 x_test :  (116203, 7)
# Thres=0.027, n= 7, R2: 67.34%
# 변형된 x_train :  (464809, 6) 변형된 x_test :  (116203, 6)
# Thres=0.036, n= 6, R2: 66.98%
# 변형된 x_train :  (464809, 5) 변형된 x_test :  (116203, 5)
# Thres=0.055, n= 5, R2: 66.81%
# 변형된 x_train :  (464809, 4) 변형된 x_test :  (116203, 4)
# Thres=0.064, n= 4, R2: 66.81%
# 변형된 x_train :  (464809, 3) 변형된 x_test :  (116203, 3)
# Thres=0.085, n= 3, R2: 66.74%
# 변형된 x_train :  (464809, 2) 변형된 x_test :  (116203, 2)
# Thres=0.088, n= 2, R2: 67.29%
# 변형된 x_train :  (464809, 1) 변형된 x_test :  (116203, 1)
# Thres=0.345, n= 1, R2: 67.53%