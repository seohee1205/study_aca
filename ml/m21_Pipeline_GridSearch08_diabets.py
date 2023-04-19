import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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


parameters = [
    {'rf__n_estimators': [100, 200], 'rf__max_depth': [6, 8, 10], 'rf__min_samples_leaf': [1, 10]},
    {'rf__max_depth': [6, 8, 10, 12], 'rf__min_samples_leaf': [3, 5, 7, 10]},
    {'rf__min_samples_leaf': [3, 5, 7, 10], 'rf__min_samples_split': [2, 3, 5, 10]},
    {'rf__n_jobs': [-1, 2, 4], 'rf__min_samples_split': [2, 3, 5, 10]}
]

#2. 모델
pipe = Pipeline([("std", StandardScaler()), ("rf", RandomForestClassifier())]) 

model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs= -1)

# RandomFore 파라미터가 아니라 pipe 파라미터가 필요함. -> 파라미터 안에 rf__ 추가

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score : ", result)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)


# model.score :  0.7786259541984732
# accuracy_score :  0.7786259541984732
