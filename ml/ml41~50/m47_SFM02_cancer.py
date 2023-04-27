import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

print(x.shape[1])   # 컬럼 수: 30

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 337, train_size= 0.8,  stratify=y, 
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

model = XGBClassifier()
model.set_params(**parameters,
                 early_stopping_rounds = 10,
                 eval_metric = 'rmse')

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
                               eval_metric = 'rmse'
                               )
    selection_model.fit(select_x_train, y_train,
          eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
          verbose= 0)
    
    
    select_y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    
    print("Thres=%.3f, n= %d, R2: %.2f%%" %(i, select_x_train.shape[1], score*100))
    # 소수 3번째 자리까지 숫자를 넣어라 
    

# 최종점수 : 0.9385964912280702
# acc 0.9385964912280702
# RMSE :  0.24779731389167603
# 변형된 x_train :  (455, 30) 변형된 x_test :  (114, 30)
# Thres=0.000, n= 30, R2: 93.86%
# 변형된 x_train :  (455, 29) 변형된 x_test :  (114, 29)
# Thres=0.001, n= 29, R2: 93.86%
# 변형된 x_train :  (455, 28) 변형된 x_test :  (114, 28)
# Thres=0.003, n= 28, R2: 96.49%
# 변형된 x_train :  (455, 27) 변형된 x_test :  (114, 27)
# Thres=0.005, n= 27, R2: 92.98%
# 변형된 x_train :  (455, 26) 변형된 x_test :  (114, 26)
# Thres=0.005, n= 26, R2: 95.61%
# 변형된 x_train :  (455, 25) 변형된 x_test :  (114, 25)
# Thres=0.007, n= 25, R2: 94.74%
# 변형된 x_train :  (455, 24) 변형된 x_test :  (114, 24)
# Thres=0.008, n= 24, R2: 94.74%
# 변형된 x_train :  (455, 23) 변형된 x_test :  (114, 23)
# Thres=0.008, n= 23, R2: 93.86%
# 변형된 x_train :  (455, 22) 변형된 x_test :  (114, 22)
# Thres=0.008, n= 22, R2: 94.74%
# 변형된 x_train :  (455, 21) 변형된 x_test :  (114, 21)
# Thres=0.008, n= 21, R2: 96.49%
# 변형된 x_train :  (455, 20) 변형된 x_test :  (114, 20)
# Thres=0.008, n= 20, R2: 92.11%
# 변형된 x_train :  (455, 19) 변형된 x_test :  (114, 19)
# Thres=0.009, n= 19, R2: 94.74%
# 변형된 x_train :  (455, 18) 변형된 x_test :  (114, 18)
# Thres=0.009, n= 18, R2: 94.74%
# 변형된 x_train :  (455, 17) 변형된 x_test :  (114, 17)
# Thres=0.010, n= 17, R2: 96.49%
# 변형된 x_train :  (455, 16) 변형된 x_test :  (114, 16)
# Thres=0.010, n= 16, R2: 93.86%
# 변형된 x_train :  (455, 15) 변형된 x_test :  (114, 15)
# Thres=0.011, n= 15, R2: 94.74%
# 변형된 x_train :  (455, 14) 변형된 x_test :  (114, 14)
# Thres=0.015, n= 14, R2: 94.74%
# 변형된 x_train :  (455, 13) 변형된 x_test :  (114, 13)
# Thres=0.016, n= 13, R2: 95.61%
# 변형된 x_train :  (455, 12) 변형된 x_test :  (114, 12)
# Thres=0.021, n= 12, R2: 95.61%
# 변형된 x_train :  (455, 11) 변형된 x_test :  (114, 11)
# Thres=0.024, n= 11, R2: 93.86%
# 변형된 x_train :  (455, 10) 변형된 x_test :  (114, 10)
# Thres=0.032, n= 10, R2: 93.86%
# 변형된 x_train :  (455, 9) 변형된 x_test :  (114, 9)
# Thres=0.033, n= 9, R2: 92.98%
# 변형된 x_train :  (455, 8) 변형된 x_test :  (114, 8)
# Thres=0.039, n= 8, R2: 96.49%
# 변형된 x_train :  (455, 7) 변형된 x_test :  (114, 7)
# Thres=0.040, n= 7, R2: 91.23%
# 변형된 x_train :  (455, 6) 변형된 x_test :  (114, 6)
# Thres=0.047, n= 6, R2: 88.60%
# 변형된 x_train :  (455, 5) 변형된 x_test :  (114, 5)
# Thres=0.050, n= 5, R2: 92.11%
# 변형된 x_train :  (455, 4) 변형된 x_test :  (114, 4)
# Thres=0.058, n= 4, R2: 91.23%
# 변형된 x_train :  (455, 3) 변형된 x_test :  (114, 3)
# Thres=0.071, n= 3, R2: 89.47%
# 변형된 x_train :  (455, 2) 변형된 x_test :  (114, 2)
# Thres=0.172, n= 2, R2: 91.23%
# 변형된 x_train :  (455, 1) 변형된 x_test :  (114, 1)
# Thres=0.271, n= 1, R2: 81.58%