import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

#1. 데이터
path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)   
# print(train_csv.shape)      # (10886, 11)


# 결측치
train_csv = train_csv.dropna()      # 결측지 제거

x = train_csv.drop(['count'], axis = 1) 
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size= 0.2, 
    # stratify=y
)

n_splits= 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = [
    {'rf__n_estimators': [100, 200], 'rf__max_depth': [6, 8, 10], 'rf__min_samples_leaf': [1, 10]},
    {'rf__max_depth': [6, 8, 10, 12], 'rf__min_samples_leaf': [3, 5, 7, 10]},
    {'rf__min_samples_leaf': [3, 5, 7, 10], 'rf__min_samples_split': [2, 3, 5, 10]},
    {'rf__n_jobs': [-1, 2, 4], 'rf__min_samples_split': [2, 3, 5, 10]}
]

#2. 모델
pipe = Pipeline([("std", StandardScaler()), ("rf", RandomForestRegressor())]) 

model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs= -1)

# RandomFore 파라미터가 아니라 pipe 파라미터가 필요함. -> 파라미터 안에 rf__ 추가

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)
print('r2_score : ', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 r2 : ", r2_score(y_test, y_pred_best))

# r2_score :  0.7895141447618661
# 최적 튠 r2 :  0.7895141447618661

