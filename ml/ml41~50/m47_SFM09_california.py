import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

print(x.shape[1])   # 컬럼 수: 8

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 337, train_size= 0.8, # stratify=y, 
)

parameters = {'n_estimators' : 1000,               # = epochs
              'learning_rate' : 0.3,
              'max_depth' : 2,
              'gamma' : 0,
              'min_child_weight' : 0,
              'subsample' : 0.2,                     # dropout과 비슷함
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
    

# 최종점수 : 0.76592117840493
# r2는  0.76592117840493
# RMSE :  0.5596686228510324
# 변형된 x_train :  (16512, 8) 변형된 x_test :  (4128, 8)
# Thres=0.034, n= 8, R2: 76.59%
# 변형된 x_train :  (16512, 7) 변형된 x_test :  (4128, 7)
# Thres=0.040, n= 7, R2: 75.58%
# 변형된 x_train :  (16512, 6) 변형된 x_test :  (4128, 6)
# Thres=0.059, n= 6, R2: 77.40%
# 변형된 x_train :  (16512, 5) 변형된 x_test :  (4128, 5)
# Thres=0.136, n= 5, R2: 74.90%
# 변형된 x_train :  (16512, 4) 변형된 x_test :  (4128, 4)
# Thres=0.144, n= 4, R2: 66.86%
# 변형된 x_train :  (16512, 3) 변형된 x_test :  (4128, 3)
# Thres=0.153, n= 3, R2: 59.67%
# 변형된 x_train :  (16512, 2) 변형된 x_test :  (4128, 2)
# Thres=0.163, n= 2, R2: 51.69%
# 변형된 x_train :  (16512, 1) 변형된 x_test :  (4128, 1)
# Thres=0.270, n= 1, R2: 48.27%