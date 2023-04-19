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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC


data_list = [load_iris, load_breast_cancer, load_digits, load_wine]
data_name_list = ['아이리스', '캔서', '디지트', '와인']
model_list = [SVC(), RandomForestClassifier(), DecisionTreeClassifier()]
model_name_list = ['SVC', 'RandomForestClassifier', 'DecisionTreeClassifier']
scaler_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler()]
scaler_name_list = ['민맥스', '스탠다드', '로버스트', '맥스앱스']

max_data_name = '바보'
max_scaler_name = '바보'
max_model_name = '바보'
for i in range(len(data_list)):
    x, y = data_list[i](return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 337, stratify=y)
    max_score = 0
    
    for j, value2 in enumerate(scaler_list):
        
        for k, value3 in enumerate(model_list):
            
            model = Pipeline([('scaler:', value2), ('model', value3)])
            model.fit(x_train, y_train)
            score = model.score(x_test, y_test)     # 평가
            y_predict=model.predict(x_test)
            acc=accuracy_score(y_test, y_predict)      # 예측
            
            if max_score < score:
                max_score = score
                max_scaler_name = scaler_name_list[j]
                max_model_name = model_name_list[k]
                
    print('\n')
    print('===============',data_name_list[i],'================')
    print('최고모델 :', max_scaler_name, max_model_name, max_score)
    print('=============================================')


# =============== 아이리스 ================
# 최고모델 : 민맥스 SVC 0.9666666666666667
# =============================================


# =============== 캔서 ================
# 최고모델 : 민맥스 RandomForestClassifier 0.956140350877193
# =============================================


# =============== 디지트 ================
# 최고모델 : 민맥스 SVC 0.9861111111111112
# =============================================


# =============== 와인 ================
# 최고모델 : 민맥스 SVC 1.0
# =============================================