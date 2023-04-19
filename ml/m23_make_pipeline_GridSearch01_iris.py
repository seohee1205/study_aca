# 랜덤서치, 그리드서치, 할빙그리드서치를
# for문으로 한방에 넣어라
# 단, 패치코타입처럼 느린 데이터는 랜덤이나 할빙 둘중에 하나만 넣어라


import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


data_list = [load_iris, load_breast_cancer, load_digits, load_wine]
data_name_list = ['아이리스', '캔서', '디지트', '와인']
model_list = [RandomForestClassifier()]
model_name_list = [RandomForestClassifier()]
scaler_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()]
scaler_name_list = ['민맥스', '스탠다드', '로버스트', '맥스앱스']

parameters = [
    {'n_estimators': [100, 200], 'max_depth': [6, 10, 12], 'min_samples_leaf': [3, 10]},
    {'max_depth': [6, 8, 10, 12], 'min_samples_leaf': [3, 5, 7, 10]},
    {'min_samples_leaf': [3, 5, 7, 10], 'min_samples_split': [2, 3, 5, 10]},
    {'n_jobs': [-1, 2, 4]}]
                    
max_data_name = '바보'
max_scaler_name = '바보'
max_model_name = '바보'
for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 337, stratify=y)
    max_score = 0
    
    for j, value2 in enumerate(scaler_list):
        
        for k, value3 in enumerate(model_list):
            for search_type in ['GridSearchCV', 'RandomizedSearchCV', 'HalvingGridSearchCV']:
                if search_type == 'GridSearchCV':
                    model = GridSearchCV(value2, value3)

                elif search_type == 'RandomizedSearchCV' :
                    model = RandomizedSearchCV(value2, value3)
                    
                elif search_type == 'HalvingGridSearchCV':

                    model = HalvingGridSearchCV(value2, value3)
                     
                    model.fit(x_train, y_train)
                    score = model.score(x_test, y_test)     # 평가
                    y_predict=model.predict(x_test)
                    acc=accuracy_score(y_test, y_predict)    # 예측
                
                if max_score < score:
                    max_score = score
                    max_scaler_name = scaler_name_list[j]
                    max_model_name = model_name_list[k]
                
    print('\n')
    print('===============',data_name_list[i],'================')
    print('최고모델 :', max_scaler_name, max_model_name, max_score)
    print('=============================================')
            