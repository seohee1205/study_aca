# classifier

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 337, train_size= 0.8, stratify=y, 
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits= n_splits, shuffle= True, random_state= 337)

# 'n_estimators' : [100, 200, 300, 400, 500, 1000], 디폴트 100 / 1~inf / 정수
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3 / 0~1 / eta
# 'max_depth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~inf / 정수
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0 / 0~inf
# 'min_child_weight' : [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100] 디폴트 1 / 0~inf
# 'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1 
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'colsample_bynode' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] / 디폴트 1 / 0~1
# 'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 0 / 0~inf / L1 절대값 가중치 규제 / alpha
# 'reg_lambda' : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 1 / 0~inf / L2 제곱 가중치 규제 / lambda
 

parameters = {'n_estimators' : 100,               # = epochs
              'learning_rate' : 0.3,
              'max_depth' : 3,
              'gamma' : 0,
              'min_child_weight' : 1,
              'subsample' : 0.5,                     # dropout과 비슷함
              'colsample_bytree' : 1,
              'colsample_bylevel' : 1,
              'colsample_bynode' : 1,
              'reg_alpha' : 1,                    # 절대값: 레이어에서 양수만들겠다/ 라쏘 / 머신러닝 모델
              'reg_lambda' : 1,                   # 제곱: 레이어에서 양수만들겠다/ 리지   / 머신러닝 모델
              'random_state' : 337,
}

#2. 모델
# model = XGBClassifier(**parameters)                                                                                                                                                                                                                                                                                                                  
model = XGBClassifier()


#3. 훈련
model.set_params(early_stopping_rounds=10, **parameters)
# model.set_params(early_stopping_rounds=10)

model.fit(x_train, y_train,
          eval_set = [(x_train, y_train), (x_test, y_test)],
        #   early_stopping_rounds = 10,
          verbose = 1 
          )

#4. 평가, 예측
# print("최상의 매개변수 : ", model.best_params_)
# print("최상의 점수 : ", model.best_score_)

results = model.score(x_test, y_test)
print("최종점수 : ", results)



