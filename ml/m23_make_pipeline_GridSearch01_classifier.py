# 삼중포문!!
# 앞부분에는 데이터셋
# 두번째는 스케일러
# 세번째는 모델
# 스케일러 전부
# 모델 = 랜덤, SVC, 디시젼트리

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.svm import SVC

#1. 데이터
data_list = [load_iris, load_breast_cancer, load_digits, load_wine]
data_name_list = ['아이리스', '캔서', '디지트', '와인']

Grid_list = [GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV]
Grid_name_list = ['그리드서치', '랜더마이즈서치', '할빙그리드서치', '할빙랜덤서치']

scaler_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()]
scaler_name_list = ['민맥스', '스탠다드', '로버스트', '맥스앱스']

parameters = [
    {'randomforestclassifier__n_estimators': [100, 200], 'randomforestclassifier__max_depth': [6, 8, 10], 'randomforestclassifier__min_samples_leaf': [1, 10]},
    {'randomforestclassifier__max_depth': [6, 8, 10, 12], 'randomforestclassifier__min_samples_leaf': [3, 5, 7, 10]},
    {'randomforestclassifier__min_samples_leaf': [3, 5, 7, 10], 'randomforestclassifier__min_samples_split': [2, 3, 5, 10]},
    {'randomforestclassifier__n_jobs': [-1, 2, 4], 'randomforestclassifier__min_samples_split': [2, 3, 5, 10]}
]

#2. 모델
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())


# model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs= -1)

for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size= 0.8, shuffle=True, random_state= 337, stratify=y
    )
    max_score = 0
    
    for j, value2 in enumerate(scaler_list):
        
        for k, value3 in enumerate(Grid_list):
            model = value3(pipe, parameters, cv = 5, verbose = 1, n_jobs = -1)
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)
            y_predict = model.predict(x_test)
            acc = accuracy_score(y_test, y_predict)
            
            if max_score < score:
                max_score = score
                max_scaler_name = scaler_name_list[j]
                max_model_name = Grid_name_list[k]

    print('\n')
    print('===============',data_name_list[i],'================')
    print('최고모델 :', max_scaler_name, max_model_name, max_score)
    print('=============================================')
    



