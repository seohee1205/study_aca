# 함수형 모델

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']

print(type(x))      # <class 'numpy.ndarray'>
print(x)


# 보스턴 임포트 못 하는 사람 (1.2 부터 안 되니까 1.1 버전 설치)
# pip uninstall scikit-learn     # 사이킷런 삭제
# pip install scikit- learn==1.1.0      # 1.1 버전 설치

# print(np.min(x), np.max(x)) # 0.0 711.0
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)     # 변환
# print(np.min(x), np.max(x)) # 0.0 1.0

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 333
)

# 전처리 (정규화)는 데이터를 나눈 후 한다
# scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0
 
# 정규화: 0 ~ 1 사이로 만들어주는 것,  y 제외 x의 훈련 데이터만
# => 정규화 했을 때 무조건 성능 좋아지는 거 X, 좋아질 수도 있고 안 좋아질 수 있음
# 정규화 비율 구하는 공식
# X / Max	->	X - min / max - min
# ex)	10~100 일 때, X - 10 / 100 - 10

#2. 모델 
model = Sequential()
model.add(Dense(1, input_dim = 8))
model.add(Dropout(0.3))
model.add(Dense(5, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(1))

#(함수형) 모델 구성
# input1 = Input(shape=(8,)) # 인풋명시, 
# dense1 = Dense(1)(input1)  # Dense 모델 구성 후, 이 모델은 어디에서 시작해서 어디에서 끝나는지 연결
# dense2 = Dense(5, activation = 'relu')(dense1)
# dense3 = Dense(7, activation = 'relu')(dense2)
# dense4 = Dense(12, activation = 'relu')(dense3)
# dense5 = Dense(4, activation = 'relu')(dense4)
# dense6 = Dense(3, activation = 'relu')(dense5)
# output1 = Dense(1)(dense6)
# model = Model(inputs = input1, outputs = output1)


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

import datetime
date = datetime.datetime.now()
print(date)  # 2023-03-14 11:11:30.046663
date = date.strftime("%m%d_%H%M") # 시간을 문자로 (월, 일, 시간, 분)
print(date)  # 0314_1116

filepath = './_save/MCP/keras28/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5'


# 정의하기

es = EarlyStopping(monitor= 'val_loss', patience = 20, mode= 'min',     # if mode = auto: 자동으로 min 또는 max로 맞춰줌 
              verbose= 1,    
              restore_best_weights= True) 

mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
        verbose = 1, 
        save_best_only= True,
        filepath="".join([filepath, 'k27_', date, '_', filename]))

hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 200,
                 validation_split = 0.2, 
                 verbose = 1,
                 callbacks= [es]    # es 호출
                 )


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)              # RMSE 함수 사용
print("RMSE : ", rmse)



'''
loss :  0.8745238780975342
r2 스코어 :  -6.03147888495655
RMSE :  0.9351598279750362

'''

