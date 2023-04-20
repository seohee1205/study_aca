
# parameters = [
#     {'n_estimators': [100, 200]},
#     {'max_depth': [6, 8, 10, 12]},
#     {'min_samples_leaf': [3, 5, 7, 10]},
#     {'min_samples_split': [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]}
# ]

parameters = [
    {'n_estimators': [100, 200], 'max_depth': [6, 10, 12], 'min_samples_leaf': [3, 10]},
    {'max_depth': [6, 8, 10, 12], 'min_samples_leaf': [3, 5, 7, 10]},
    {'min_samples_leaf': [3, 5, 7, 10], 'min_samples_split': [2, 3, 5, 10]},
    {'n_jobs': [-1, 2, 4]}
]

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import time


#1. 데이터
path = './_data/dacon_diabetes/'
train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)
test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

x = train_csv.drop(['Outcome'], axis = 1)
y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size= 0.2, 
    stratify=y
)

n_splits= 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = RandomizedSearchCV(RandomForestClassifier(), parameters, 
                    cv=kfold, 
                    # cv=5,      # 분류의 디폴트는 StratifiedKFold야
                     verbose=1, 
                     refit= True,      # 디폴트값은 True, False하게 되면 최종 파라미터로 출력
                    # refit=False,       # True는 최상의 파라미터로 출력 
                     n_iter=5,
                     n_jobs= -1)

#3. 컴파일, 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)
print("best_score_ : ", model.best_score_)
print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ", accuracy_score(y_test, y_pred_best))

print("걸린시간 : ", round(end_time - start_time, 2), '초')


# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=3, n_estimators=200)
# 최적의 파라미터 :  {'n_estimators': 200, 'min_samples_leaf': 3, 'max_depth': 12}
# best_score_ :  0.7542307692307693
# model.score :  0.7786259541984732
# accuracy_score :  0.7786259541984732
# 최적 튠 ACC :  0.7786259541984732
# 걸린시간 :  4.28 초
