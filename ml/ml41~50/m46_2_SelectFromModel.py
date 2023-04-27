import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

#1. 데이터
x, y = load_diabetes(return_X_y=True)

# print(x.shape[1])   # 컬럼 수: 10

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
    

#     최종점수 : 0.4993781445646843
# r2는  0.4993781445646843
# RMSE :  51.056803869663895
# 변형된 x_train :  (353, 10) 변형된 x_test :  (89, 10)
# Thres=0.032, n= 10, R2: 49.94%
# 변형된 x_train :  (353, 9) 변형된 x_test :  (89, 9)
# Thres=0.065, n= 9, R2: 31.54%
# 변형된 x_train :  (353, 8) 변형된 x_test :  (89, 8)
# Thres=0.072, n= 8, R2: 40.08%
# 변형된 x_train :  (353, 7) 변형된 x_test :  (89, 7)
# Thres=0.078, n= 7, R2: 37.00%
# 변형된 x_train :  (353, 6) 변형된 x_test :  (89, 6)
# Thres=0.087, n= 6, R2: 44.07%
# 변형된 x_train :  (353, 5) 변형된 x_test :  (89, 5)
# Thres=0.105, n= 5, R2: 40.59%
# 변형된 x_train :  (353, 4) 변형된 x_test :  (89, 4)
# Thres=0.120, n= 4, R2: 31.31%
# 변형된 x_train :  (353, 3) 변형된 x_test :  (89, 3)
# Thres=0.138, n= 3, R2: 31.43%
# 변형된 x_train :  (353, 2) 변형된 x_test :  (89, 2)
# Thres=0.147, n= 2, R2: 11.42%
# 변형된 x_train :  (353, 1) 변형된 x_test :  (89, 1)
# Thres=0.155, n= 1, R2: 6.71%