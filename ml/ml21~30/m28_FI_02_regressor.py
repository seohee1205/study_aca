# [실습]
# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거
# 재구성후
# 모델을 돌려서 결과 도출
# 기초모델들과 성능비교

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing, load_diabetes

#1 데이터
ddarung_path = 'c:/_study/_data/_ddarung/'
kaggle_bike_path = 'c:/_study/_data/_kaggle_bike/'

ddarung = pd.read_csv(ddarung_path + 'train.csv', index_col = 0).dropna()
kaggle_bike = pd.read_csv(kaggle_bike_path + 'train.csv', index_col = 0).dropna()

x1 = ddarung.drop(['count'], axis = 1).values
y1 = ddarung['count'].values

x2 = kaggle_bike.drop(['count', 'casual', 'registered'], axis = 1).values
y2 = kaggle_bike['count'].values

num_classes = len(np.unique(np.concatenate((y1, y2))))


data_list = [fetch_california_housing(return_X_y = True),
             load_diabetes(return_X_y = True),
             (x1, y1),
             (x2, y2)]

scaler_list = [MinMaxScaler(),
               MaxAbsScaler(),
               StandardScaler(),
               RobustScaler()]

model_list = [DecisionTreeRegressor(),
              RandomForestRegressor(),
              GradientBoostingRegressor(),
              XGBRFRegressor(num_class=num_classes, objective='multi:softmax')]

data_list_name = ['캘리포니아',
                  '디아뱃',
                  '따릉이',
                  '캐글 바이크']

model_list_name = ['DecisionTreeClassifier',
                   'RandomForestClassifier',
                   'GradientBoostingClassifier',
                   'XGBClassifier']

for i in range(len(data_list)):
    x, y = data_list[i]
    if i == 4:  # '데이콘 디아뱃' dataset
        x = x.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state = 1234)
    
    for j in range(len(scaler_list)):
        scaler = scaler_list[j]
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        for k in range(len(model_list)):
            model = model_list[k]
            model.fit(x_train, y_train)
            
            y_predict = model.predict(x_test)
            
            r2 = r2_score(y_test, y_predict)
            print(data_list_name[i], model_list_name[k], 'r2_score : ', r2)
            
            idx = np.argsort(model.feature_importances_)[int(len(model.feature_importances_) * 0.2) : int(len(model.feature_importances_) * 0.25)]
            keep_idx = np.delete(np.arange(x.shape[1]), idx)
            x_drop = x[:, keep_idx]

            x_train1, x_test1, y_train1, y_test1 = train_test_split(x_drop, y, train_size = 0.7, shuffle = True, random_state = 123)
            
            x_train1 = scaler.fit_transform(x_train1)
            x_test1 = scaler.transform(x_test1)
            
            model.fit(x_train1, y_train1)
            
            y_predict1 = model.predict(x_test1)
                        
            r22 = r2_score(y_test1, y_predict1)
            print(data_list_name[i], model_list_name[k], 'r2_score2 : ', r22)