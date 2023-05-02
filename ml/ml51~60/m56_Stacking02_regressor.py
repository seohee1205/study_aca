# 실습

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

#1. 데이터
data_list = [load_diabetes, fetch_california_housing]
data_name_list = ['디아벳', '캘리포니아']

model_list = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor()]
model_name_list = ['리니어', '케이네이볼', '디시전트리']

lr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=8)
dt = DecisionTreeRegressor()

for i, v in enumerate(data_list):
    x, y = v(return_X_y= True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size= 0.8, random_state=123, shuffle=True
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    list =[]
    for j, v2 in enumerate(model_list):
        
        model = StackingRegressor(
            estimators=[('LR', lr), ('KNN', knn), ('DT', dt)],
            final_estimator= RandomForestRegressor(),
        )
        regressor = [lr, knn, dt]
        for model2 in regressor:
            model2.fit(x_train, y_train)
            y_predict = model2.predict(x_test)
            score2 = r2_score(y_test, y_predict)
            class_name = model2.__class__.__name__
            print('==========', data_name_list[i], '==========')
            print("{0} r2 : {1:.4f}".format(class_name, score2))
            # print('model.score : ', model.score(x_test, y_test))
            print('Stacking r2 : ', r2_score(y_test, y_predict))
            list.append(score2)


# ========== 디아벳 ==========
# LinearRegression r2 : 0.5676
# Stacking r2 :  0.5675916622351822
# ========== 디아벳 ==========
# KNeighborsRegressor r2 : 0.4506
# Stacking r2 :  0.45059882929398765
# ========== 디아벳 ==========
# DecisionTreeRegressor r2 : 0.1464
# Stacking r2 :  0.1464360926398416
# ========== 디아벳 ==========
# LinearRegression r2 : 0.5676
# Stacking r2 :  0.5675916622351822
# ========== 디아벳 ==========
# KNeighborsRegressor r2 : 0.4506
# Stacking r2 :  0.45059882929398765
# ========== 디아벳 ==========
# DecisionTreeRegressor r2 : 0.1050
# Stacking r2 :  0.1049818704556661
# ========== 디아벳 ==========
# LinearRegression r2 : 0.5676
# Stacking r2 :  0.5675916622351822
# ========== 디아벳 ==========
# KNeighborsRegressor r2 : 0.4506
# Stacking r2 :  0.45059882929398765
# ========== 디아벳 ==========
# DecisionTreeRegressor r2 : 0.1118
# Stacking r2 :  0.11180886935951329
# ========== 캘리포니아 ==========
# LinearRegression r2 : 0.6105
# Stacking r2 :  0.6104546894797876
# ========== 캘리포니아 ==========
# KNeighborsRegressor r2 : 0.7007
# Stacking r2 :  0.700712982510346
# ========== 캘리포니아 ==========
# DecisionTreeRegressor r2 : 0.6097
# Stacking r2 :  0.6097139590336509
# ========== 캘리포니아 ==========
# LinearRegression r2 : 0.6105
# Stacking r2 :  0.6104546894797876
# ========== 캘리포니아 ==========
# KNeighborsRegressor r2 : 0.7007
# Stacking r2 :  0.700712982510346
# ========== 캘리포니아 ==========
# DecisionTreeRegressor r2 : 0.6071
# Stacking r2 :  0.607141754208331
# ========== 캘리포니아 ==========
# LinearRegression r2 : 0.6105
# Stacking r2 :  0.6104546894797876
# ========== 캘리포니아 ==========
# KNeighborsRegressor r2 : 0.7007
# Stacking r2 :  0.700712982510346
# ========== 캘리포니아 ==========
# DecisionTreeRegressor r2 : 0.6066
# Stacking r2 :  0.6066130277308892