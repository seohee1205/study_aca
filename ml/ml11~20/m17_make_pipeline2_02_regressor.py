import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, load_diabetes, fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC


data_list = [load_diabetes, fetch_california_housing]
data_name_list = ['디아벳', '캘리포니아']

model_list = [RandomForestRegressor(), DecisionTreeRegressor()]
model_name_list = ['RandomForestRegressor', 'DecisionTreeRegressor']

scaler_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()]
scaler_name_list = ['민맥스', '스탠다드', '로버스트', '맥스앱스']

max_data_name = '바보'
max_scaler_name = '바보'
max_model_name = '바보'
for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 337)
    max_score = 0
    
    for j, value2 in enumerate(scaler_list):
        
        for k, value3 in enumerate(model_list):
            
            model = make_pipeline(value2, value3)
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)     # 평가
            y_predict=model.predict(x_test)
            r2=r2_score(y_test, y_predict)      # 예측
            
            if max_score < score:
                max_score = score
                max_scaler_name = scaler_name_list[j]
                max_model_name = model_name_list[k]
                
    print('\n')
    print('===============',data_name_list[i],'================')
    print('최고모델 :', max_scaler_name, max_model_name, max_score)
    print('=============================================')



# =============== 디아벳 ================
# 최고모델 : 스탠다드 RandomForestRegressor 0.42775924522974385
# =============================================

# =============== 캘리포니아 ================
# 최고모델 : 스탠다드 RandomForestRegressor 0.800823378497722
# =============================================