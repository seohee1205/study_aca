# 회귀로 맹그러
# 회귀데이터 올인 포문
# scaler 6개 올인 포문

# 정규분포로 만들고, 분위수를 기준으로 0-1 사이로 만들기 때문에
# 이상치에 자유롭다!


from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer   #  Quantile: 분위수 / 모든 값은 0~1사이로
import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
# x, y = load_iris(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, shuffle=True, train_size= 0.8, random_state= 337, stratify=y
# )

# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer(n_quantiles=1000)  # 디폴트 / 분위수 조절
# scaler = QuantileTransformer(n_quantiles=10)
# scaler = PowerTransformer()
# scaler = PowerTransformer(method='yeo-johnson') # 디폴트
# scaler = PowerTransformer(method='box-cox') # 양수만 사용가능


# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델
# model = RandomForestRegressor()

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# print("결과 : ", round(model.score(x_test, y_test)), 4)

###############################################################

data_list = [load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype]
data_name_list = ['아이리스', '캔서', '디지트', '와인', '패치콥타입']

scaler_list = [MinMaxScaler(), StandardScaler(), RobustScaler(), MaxAbsScaler(), QuantileTransformer(), PowerTransformer()]
scaler_name_list = ['민맥스', '스탠다드', '로버스트', '맥스앱스', '퀀틸', '파워']              

for i, v in enumerate(data_list):
    x, y = v(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size= 0.8, random_state= 337, shuffle=True
    )
    
    for j, v2 in enumerate(scaler_list):
        scaler = v2
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        model = RandomForestClassifier()
        
        model.fit(x_train, y_train)
        
        print('===============',data_name_list[i],'================')
        print(scaler_name_list[j], "결과 : ", round(model.score(x_test, y_test), 4))


# =============== 아이리스 ================
# 민맥스 결과 :  0.9667
# =============== 아이리스 ================
# 스탠다드 결과 :  0.9667
# =============== 아이리스 ================
# 로버스트 결과 :  0.9667
# =============== 아이리스 ================
# 맥스앱스 결과 :  0.9667
# =============== 아이리스 ================
# 퀀틸 결과 :  0.9667
# =============== 아이리스 ================
# 파워 결과 :  0.9667
# =============== 캔서 ================
# 민맥스 결과 :  0.9649
# =============== 캔서 ================
# 스탠다드 결과 :  0.9649
# =============== 캔서 ================
# 로버스트 결과 :  0.9649
# =============== 캔서 ================
# 맥스앱스 결과 :  0.9649
# =============== 캔서 ================
# 퀀틸 결과 :  0.9649
# =============== 캔서 ================
# 파워 결과 :  0.9649
# =============== 디지트 ================
# 민맥스 결과 :  0.9639
# =============== 디지트 ================
# 스탠다드 결과 :  0.9694
# =============== 디지트 ================
# 로버스트 결과 :  0.9583
# =============== 디지트 ================
# 맥스앱스 결과 :  0.9667
# =============== 디지트 ================
# 퀀틸 결과 :  0.9722
# =============== 디지트 ================
# 파워 결과 :  0.9667
# =============== 와인 ================
# 민맥스 결과 :  0.9722
# =============== 와인 ================
# 스탠다드 결과 :  0.9722
# =============== 와인 ================
# 로버스트 결과 :  0.9722
# =============== 와인 ================
# 맥스앱스 결과 :  0.9444
# =============== 와인 ================
# 퀀틸 결과 :  0.9722
# =============== 와인 ================
# 파워 결과 :  0.9722
# =============== 패치콥타입 ================
# 민맥스 결과 :  0.9558
# =============== 패치콥타입 ================
# 스탠다드 결과 :  0.9557
# =============== 패치콥타입 ================
# 로버스트 결과 :  0.9558
# =============== 패치콥타입 ================
# 맥스앱스 결과 :  0.9547
# =============== 패치콥타입 ================
# 퀀틸 결과 :  0.9561
# =============== 패치콥타입 ================
# 파워 결과 :  0.9553