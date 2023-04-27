import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_digits
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

#1. 데이터
x, y = load_digits(return_X_y=True)

print(x.shape[1])   # 컬럼 수: 64

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 337, train_size= 0.8, # stratify=y, 
)

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

model = XGBRegressor()
model.set_params(**parameters,
                 early_stopping_rounds = 10,
                 eval_metric = 'rmse')

model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          verbose=False
          )
results = model.score(x_test, y_test)
print("최종점수 :", results)

y_predict = model.predict(x_test)
r2 =  r2_score(y_test, y_predict)
print("r2는 ", r2)

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

    selection_model = XGBRegressor()
    selection_model.set_params(early_stopping_rounds= 10, **parameters,
                               eval_metric = 'rmse'
                               )
    selection_model.fit(select_x_train, y_train,
          eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
          verbose= 0)
    
    
    select_y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, select_y_predict)
    
    print("Thres=%.3f, n= %d, R2: %.2f%%" %(i, select_x_train.shape[1], score*100))
    # 소수 3번째 자리까지 숫자를 넣어라 
    

# 최종점수 : 0.615940423475908
# r2는  0.615940423475908
# RMSE :  1.8001264572689029
# 변형된 x_train :  (1437, 64) 변형된 x_test :  (360, 64)
# Thres=0.000, n= 64, R2: 61.59%
# 변형된 x_train :  (1437, 64) 변형된 x_test :  (360, 64)
# Thres=0.000, n= 64, R2: 61.59%
# 변형된 x_train :  (1437, 64) 변형된 x_test :  (360, 64)
# Thres=0.000, n= 64, R2: 61.59%
# 변형된 x_train :  (1437, 64) 변형된 x_test :  (360, 64)
# Thres=0.000, n= 64, R2: 61.59%
# 변형된 x_train :  (1437, 64) 변형된 x_test :  (360, 64)
# Thres=0.000, n= 64, R2: 61.59%
# 변형된 x_train :  (1437, 59) 변형된 x_test :  (360, 59)
# Thres=0.000, n= 59, R2: 60.42%
# 변형된 x_train :  (1437, 58) 변형된 x_test :  (360, 58)
# Thres=0.000, n= 58, R2: 63.09%
# 변형된 x_train :  (1437, 57) 변형된 x_test :  (360, 57)
# Thres=0.000, n= 57, R2: 60.71%
# 변형된 x_train :  (1437, 56) 변형된 x_test :  (360, 56)
# Thres=0.000, n= 56, R2: 71.28%
# 변형된 x_train :  (1437, 55) 변형된 x_test :  (360, 55)
# Thres=0.000, n= 55, R2: 59.66%
# 변형된 x_train :  (1437, 54) 변형된 x_test :  (360, 54)
# Thres=0.001, n= 54, R2: 62.89%
# 변형된 x_train :  (1437, 53) 변형된 x_test :  (360, 53)
# Thres=0.001, n= 53, R2: 63.59%
# 변형된 x_train :  (1437, 52) 변형된 x_test :  (360, 52)
# Thres=0.001, n= 52, R2: 59.45%
# 변형된 x_train :  (1437, 51) 변형된 x_test :  (360, 51)
# Thres=0.001, n= 51, R2: 56.06%
# 변형된 x_train :  (1437, 50) 변형된 x_test :  (360, 50)
# Thres=0.002, n= 50, R2: 62.58%
# 변형된 x_train :  (1437, 49) 변형된 x_test :  (360, 49)
# Thres=0.003, n= 49, R2: 70.48%
# 변형된 x_train :  (1437, 48) 변형된 x_test :  (360, 48)
# Thres=0.003, n= 48, R2: 63.52%
# 변형된 x_train :  (1437, 47) 변형된 x_test :  (360, 47)
# Thres=0.004, n= 47, R2: 62.69%
# 변형된 x_train :  (1437, 46) 변형된 x_test :  (360, 46)
# Thres=0.004, n= 46, R2: 69.06%
# 변형된 x_train :  (1437, 45) 변형된 x_test :  (360, 45)
# Thres=0.004, n= 45, R2: 61.82%
# 변형된 x_train :  (1437, 44) 변형된 x_test :  (360, 44)
# Thres=0.005, n= 44, R2: 66.31%
# 변형된 x_train :  (1437, 43) 변형된 x_test :  (360, 43)
# Thres=0.006, n= 43, R2: 61.74%
# 변형된 x_train :  (1437, 42) 변형된 x_test :  (360, 42)
# Thres=0.007, n= 42, R2: 64.58%
# 변형된 x_train :  (1437, 41) 변형된 x_test :  (360, 41)
# Thres=0.008, n= 41, R2: 63.59%
# 변형된 x_train :  (1437, 40) 변형된 x_test :  (360, 40)
# Thres=0.008, n= 40, R2: 60.67%
# 변형된 x_train :  (1437, 39) 변형된 x_test :  (360, 39)
# Thres=0.008, n= 39, R2: 63.95%
# 변형된 x_train :  (1437, 38) 변형된 x_test :  (360, 38)
# Thres=0.008, n= 38, R2: 55.96%
# 변형된 x_train :  (1437, 37) 변형된 x_test :  (360, 37)
# Thres=0.009, n= 37, R2: 64.73%
# 변형된 x_train :  (1437, 36) 변형된 x_test :  (360, 36)
# Thres=0.009, n= 36, R2: 62.19%
# 변형된 x_train :  (1437, 35) 변형된 x_test :  (360, 35)
# Thres=0.009, n= 35, R2: 62.72%
# 변형된 x_train :  (1437, 34) 변형된 x_test :  (360, 34)
# Thres=0.010, n= 34, R2: 65.51%
# 변형된 x_train :  (1437, 33) 변형된 x_test :  (360, 33)
# Thres=0.010, n= 33, R2: 66.38%
# 변형된 x_train :  (1437, 32) 변형된 x_test :  (360, 32)
# Thres=0.011, n= 32, R2: 64.16%
# 변형된 x_train :  (1437, 31) 변형된 x_test :  (360, 31)
# Thres=0.011, n= 31, R2: 60.57%
# 변형된 x_train :  (1437, 30) 변형된 x_test :  (360, 30)
# Thres=0.012, n= 30, R2: 54.16%
# 변형된 x_train :  (1437, 29) 변형된 x_test :  (360, 29)
# Thres=0.013, n= 29, R2: 70.45%
# 변형된 x_train :  (1437, 28) 변형된 x_test :  (360, 28)
# Thres=0.013, n= 28, R2: 68.43%
# 변형된 x_train :  (1437, 27) 변형된 x_test :  (360, 27)
# Thres=0.014, n= 27, R2: 68.15%
# 변형된 x_train :  (1437, 26) 변형된 x_test :  (360, 26)
# Thres=0.014, n= 26, R2: 61.79%
# 변형된 x_train :  (1437, 25) 변형된 x_test :  (360, 25)
# Thres=0.015, n= 25, R2: 64.56%
# 변형된 x_train :  (1437, 24) 변형된 x_test :  (360, 24)
# Thres=0.015, n= 24, R2: 61.56%
# 변형된 x_train :  (1437, 23) 변형된 x_test :  (360, 23)
# Thres=0.015, n= 23, R2: 67.92%
# 변형된 x_train :  (1437, 22) 변형된 x_test :  (360, 22)
# Thres=0.017, n= 22, R2: 63.71%
# 변형된 x_train :  (1437, 21) 변형된 x_test :  (360, 21)
# Thres=0.017, n= 21, R2: 60.35%
# 변형된 x_train :  (1437, 20) 변형된 x_test :  (360, 20)
# Thres=0.019, n= 20, R2: 61.34%
# 변형된 x_train :  (1437, 19) 변형된 x_test :  (360, 19)
# Thres=0.022, n= 19, R2: 65.02%
# 변형된 x_train :  (1437, 18) 변형된 x_test :  (360, 18)
# Thres=0.025, n= 18, R2: 63.19%
# 변형된 x_train :  (1437, 17) 변형된 x_test :  (360, 17)
# Thres=0.025, n= 17, R2: 64.90%
# 변형된 x_train :  (1437, 16) 변형된 x_test :  (360, 16)
# Thres=0.026, n= 16, R2: 64.03%
# 변형된 x_train :  (1437, 15) 변형된 x_test :  (360, 15)
# Thres=0.027, n= 15, R2: 62.39%
# 변형된 x_train :  (1437, 14) 변형된 x_test :  (360, 14)
# Thres=0.028, n= 14, R2: 60.27%
# 변형된 x_train :  (1437, 13) 변형된 x_test :  (360, 13)
# Thres=0.028, n= 13, R2: 59.71%
# 변형된 x_train :  (1437, 12) 변형된 x_test :  (360, 12)
# Thres=0.028, n= 12, R2: 56.99%
# 변형된 x_train :  (1437, 11) 변형된 x_test :  (360, 11)
# Thres=0.028, n= 11, R2: 50.50%
# 변형된 x_train :  (1437, 10) 변형된 x_test :  (360, 10)
# Thres=0.029, n= 10, R2: 54.06%
# 변형된 x_train :  (1437, 9) 변형된 x_test :  (360, 9)
# Thres=0.029, n= 9, R2: 58.97%
# 변형된 x_train :  (1437, 8) 변형된 x_test :  (360, 8)
# Thres=0.030, n= 8, R2: 54.18%
# 변형된 x_train :  (1437, 7) 변형된 x_test :  (360, 7)
# Thres=0.032, n= 7, R2: 53.15%
# 변형된 x_train :  (1437, 6) 변형된 x_test :  (360, 6)
# Thres=0.032, n= 6, R2: 52.21%
# 변형된 x_train :  (1437, 5) 변형된 x_test :  (360, 5)
# Thres=0.034, n= 5, R2: 42.83%
# 변형된 x_train :  (1437, 4) 변형된 x_test :  (360, 4)
# Thres=0.039, n= 4, R2: 35.91%
# 변형된 x_train :  (1437, 3) 변형된 x_test :  (360, 3)
# Thres=0.049, n= 3, R2: 10.83%
# 변형된 x_train :  (1437, 2) 변형된 x_test :  (360, 2)
# Thres=0.059, n= 2, R2: 11.06%
# 변형된 x_train :  (1437, 1) 변형된 x_test :  (360, 1)
# Thres=0.121, n= 1, R2: 5.53%