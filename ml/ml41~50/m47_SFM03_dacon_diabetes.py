import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler 
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
path = './_data/dacon_diabetes/'
train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)
test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

x = train_csv.drop(['Outcome'], axis = 1)
y = train_csv['Outcome']

print(x.shape[1])   # 컬럼 수: 8

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 337, train_size= 0.8,  #stratify=y, 
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
                 eval_metric = 'error')

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
                               eval_metric = 'error'
                               )
    selection_model.fit(select_x_train, y_train,
          eval_set=[(select_x_train, y_train), (select_x_test, y_test)],
          verbose= 0)
    
    
    select_y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_predict)
    
    print("Thres=%.3f, n= %d, R2: %.2f%%" %(i, select_x_train.shape[1], score*100))
    # 소수 3번째 자리까지 숫자를 넣어라 
    
# 최종점수 : 0.7709923664122137
# acc 0.7709923664122137
# RMSE :  0.47854742041702225
# 변형된 x_train :  (521, 8) 변형된 x_test :  (131, 8)
# Thres=0.055, n= 8, R2: 77.10%
# 변형된 x_train :  (521, 7) 변형된 x_test :  (131, 7)
# Thres=0.066, n= 7, R2: 80.92%
# 변형된 x_train :  (521, 6) 변형된 x_test :  (131, 6)
# Thres=0.077, n= 6, R2: 77.10%
# 변형된 x_train :  (521, 5) 변형된 x_test :  (131, 5)
# Thres=0.080, n= 5, R2: 80.15%
# 변형된 x_train :  (521, 4) 변형된 x_test :  (131, 4)
# Thres=0.122, n= 4, R2: 78.63%
# 변형된 x_train :  (521, 3) 변형된 x_test :  (131, 3)
# Thres=0.166, n= 3, R2: 76.34%
# 변형된 x_train :  (521, 2) 변형된 x_test :  (131, 2)
# Thres=0.198, n= 2, R2: 76.34%
# 변형된 x_train :  (521, 1) 변형된 x_test :  (131, 1)
# Thres=0.236, n= 1, R2: 76.34%
