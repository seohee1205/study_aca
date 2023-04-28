#[실습]Dancon_wine : ML활용 acc올리기
# -RandomForestClassifier모델 
#결측치/ 원핫인코딩, 데이터분리, 스케일링/ 함수형,dropout
#다중분류 - softmax, categorical

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgbm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE

#1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size= 0.2, 
    stratify=y
)

print(pd.Series(y).value_counts().sort_index())
# 0    212
# 1    357

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

print("============== SMOTE 적용 후 =============")
smote = SMOTE(random_state= 321, k_neighbors= 5) # 디폴트 5
x_train, y_train = smote.fit_resample(x_train.copy(), y_train.copy())


#2. 모델구성 
# model = XGBClassifier()
model = RandomForestClassifier(random_state=337)

#3. 컴파일, 훈련 
model.fit(x_train, y_train)  

  
#4. 평가예측 
results = model.score(x_test, y_test)
print("최종점수 :", results)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)
print("acc 는", acc)


# ============== SMOTE 적용 후 =============
# 최종점수 : 0.9385964912280702
# acc 는 0.9385964912280702