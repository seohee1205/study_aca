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
    


'''
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 30
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 90
Fitting 5 folds for each of 19 candidates, totalling 95 fits
n_iterations: 2
n_required_iterations: 2
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 4
n_resources: 30
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 1
n_candidates: 2
n_resources: 90
Fitting 5 folds for each of 2 candidates, totalling 10 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 30
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 90
Fitting 5 folds for each of 19 candidates, totalling 95 fits
n_iterations: 2
n_required_iterations: 2
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 4
n_resources: 30
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 1
n_candidates: 2
n_resources: 90
Fitting 5 folds for each of 2 candidates, totalling 10 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 30
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 90
Fitting 5 folds for each of 19 candidates, totalling 95 fits
n_iterations: 2
n_required_iterations: 2
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 4
n_resources: 30
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 1
n_candidates: 2
n_resources: 90
Fitting 5 folds for each of 2 candidates, totalling 10 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 30
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 90
Fitting 5 folds for each of 19 candidates, totalling 95 fits
n_iterations: 2
n_required_iterations: 2
n_possible_iterations: 2
min_resources_: 30
max_resources_: 120
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 4
n_resources: 30
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 1
n_candidates: 2
n_resources: 90
Fitting 5 folds for each of 2 candidates, totalling 10 fits


=============== 아이리스 ================
최고모델 : 민맥스 그리드서치 0.9666666666666667
=============================================
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 20
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 20
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 60
Fitting 5 folds for each of 19 candidates, totalling 95 fits
----------
iter: 2
n_candidates: 7
n_resources: 180
Fitting 5 folds for each of 7 candidates, totalling 35 fits
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 20
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 22
n_resources: 20
Fitting 5 folds for each of 22 candidates, totalling 110 fits
----------
iter: 1
n_candidates: 8
n_resources: 60
Fitting 5 folds for each of 8 candidates, totalling 40 fits
----------
iter: 2
n_candidates: 3
n_resources: 180
Fitting 5 folds for each of 3 candidates, totalling 15 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 20
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 20
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 60
Fitting 5 folds for each of 19 candidates, totalling 95 fits
----------
iter: 2
n_candidates: 7
n_resources: 180
Fitting 5 folds for each of 7 candidates, totalling 35 fits
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 20
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 22
n_resources: 20
Fitting 5 folds for each of 22 candidates, totalling 110 fits
----------
iter: 1
n_candidates: 8
n_resources: 60
Fitting 5 folds for each of 8 candidates, totalling 40 fits
----------
iter: 2
n_candidates: 3
n_resources: 180
Fitting 5 folds for each of 3 candidates, totalling 15 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 20
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 20
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 60
Fitting 5 folds for each of 19 candidates, totalling 95 fits
----------
iter: 2
n_candidates: 7
n_resources: 180
Fitting 5 folds for each of 7 candidates, totalling 35 fits
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 20
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 22
n_resources: 20
Fitting 5 folds for each of 22 candidates, totalling 110 fits
----------
iter: 1
n_candidates: 8
n_resources: 60
Fitting 5 folds for each of 8 candidates, totalling 40 fits
----------
iter: 2
n_candidates: 3
n_resources: 180
Fitting 5 folds for each of 3 candidates, totalling 15 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 20
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 20
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 60
Fitting 5 folds for each of 19 candidates, totalling 95 fits
----------
iter: 2
n_candidates: 7
n_resources: 180
Fitting 5 folds for each of 7 candidates, totalling 35 fits
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 20
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 22
n_resources: 20
Fitting 5 folds for each of 22 candidates, totalling 110 fits
----------
iter: 1
n_candidates: 8
n_resources: 60
Fitting 5 folds for each of 8 candidates, totalling 40 fits
----------
iter: 2
n_candidates: 3
n_resources: 180
Fitting 5 folds for each of 3 candidates, totalling 15 fits


=============== 캔서 ================
최고모델 : 스탠다드 할빙그리드서치 0.9649122807017544
=============================================
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 100
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 300
Fitting 5 folds for each of 19 candidates, totalling 95 fits
----------
iter: 2
n_candidates: 7
n_resources: 900
Fitting 5 folds for each of 7 candidates, totalling 35 fits
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 14
n_resources: 100
Fitting 5 folds for each of 14 candidates, totalling 70 fits
----------
iter: 1
n_candidates: 5
n_resources: 300
Fitting 5 folds for each of 5 candidates, totalling 25 fits
----------
iter: 2
n_candidates: 2
n_resources: 900
Fitting 5 folds for each of 2 candidates, totalling 10 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 100
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 300
Fitting 5 folds for each of 19 candidates, totalling 95 fits
----------
iter: 2
n_candidates: 7
n_resources: 900
Fitting 5 folds for each of 7 candidates, totalling 35 fits
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 14
n_resources: 100
Fitting 5 folds for each of 14 candidates, totalling 70 fits
----------
iter: 1
n_candidates: 5
n_resources: 300
Fitting 5 folds for each of 5 candidates, totalling 25 fits
----------
iter: 2
n_candidates: 2
n_resources: 900
Fitting 5 folds for each of 2 candidates, totalling 10 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 100
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 300
Fitting 5 folds for each of 19 candidates, totalling 95 fits
----------
iter: 2
n_candidates: 7
n_resources: 900
Fitting 5 folds for each of 7 candidates, totalling 35 fits
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 14
n_resources: 100
Fitting 5 folds for each of 14 candidates, totalling 70 fits
----------
iter: 1
n_candidates: 5
n_resources: 300
Fitting 5 folds for each of 5 candidates, totalling 25 fits
----------
iter: 2
n_candidates: 2
n_resources: 900
Fitting 5 folds for each of 2 candidates, totalling 10 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 100
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 300
Fitting 5 folds for each of 19 candidates, totalling 95 fits
----------
iter: 2
n_candidates: 7
n_resources: 900
Fitting 5 folds for each of 7 candidates, totalling 35 fits
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 100
max_resources_: 1437
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 14
n_resources: 100
Fitting 5 folds for each of 14 candidates, totalling 70 fits
----------
iter: 1
n_candidates: 5
n_resources: 300
Fitting 5 folds for each of 5 candidates, totalling 25 fits
----------
iter: 2
n_candidates: 2
n_resources: 900
Fitting 5 folds for each of 2 candidates, totalling 10 fits


=============== 디지트 ================
최고모델 : 로버스트 할빙그리드서치 0.9861111111111112
=============================================
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 30
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 30
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 90
Fitting 5 folds for each of 19 candidates, totalling 95 fits
n_iterations: 2
n_required_iterations: 2
n_possible_iterations: 2
min_resources_: 30
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 4
n_resources: 30
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 1
n_candidates: 2
n_resources: 90
Fitting 5 folds for each of 2 candidates, totalling 10 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 30
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 30
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 90
Fitting 5 folds for each of 19 candidates, totalling 95 fits
n_iterations: 2
n_required_iterations: 2
n_possible_iterations: 2
min_resources_: 30
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 4
n_resources: 30
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 1
n_candidates: 2
n_resources: 90
Fitting 5 folds for each of 2 candidates, totalling 10 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 30
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 30
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 90
Fitting 5 folds for each of 19 candidates, totalling 95 fits
n_iterations: 2
n_required_iterations: 2
n_possible_iterations: 2
min_resources_: 30
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 4
n_resources: 30
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 1
n_candidates: 2
n_resources: 90
Fitting 5 folds for each of 2 candidates, totalling 10 fits
Fitting 5 folds for each of 56 candidates, totalling 280 fits
Fitting 5 folds for each of 10 candidates, totalling 50 fits
n_iterations: 2
n_required_iterations: 4
n_possible_iterations: 2
min_resources_: 30
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 30
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 90
Fitting 5 folds for each of 19 candidates, totalling 95 fits
n_iterations: 2
n_required_iterations: 2
n_possible_iterations: 2
min_resources_: 30
max_resources_: 142
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 4
n_resources: 30
Fitting 5 folds for each of 4 candidates, totalling 20 fits
----------
iter: 1
n_candidates: 2
n_resources: 90
Fitting 5 folds for each of 2 candidates, totalling 10 fits


=============== 와인 ================
최고모델 : 민맥스 그리드서치 1.0
=============================================
'''