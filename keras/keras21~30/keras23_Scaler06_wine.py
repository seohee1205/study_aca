# 사이킷런 load_wine

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping


#1. 데이터
datasets = load_wine()
print(datasets.DESCR)   # 판다스에서 describe() 와 동일
print(datasets.feature_names) # 판다스에서 clolumns 와 동일
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (178, 13) (178,)
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y))    # y의 라벨값 :  [0 1 2]

################# 이 지점에서 원핫을 해줘야 함 #######################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
# print(y)
print(y.shape)      # (178, 3)


## y를 (178, ) -> (150, 3)
# 판다스에 겟더미, 사이킷런에 원핫인코더
#####################################################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state= 333, 
    train_size = 0.8, 
    stratify = y
)
print(y_train)
print(np.unique(y_train, return_counts= True))

# 전처리 (정규화)는 데이터를 나눈 후 한다
# scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0

#2. 모델 구성
model = Sequential()
model.add(Dense(50, activation = 'relu', input_dim = 13))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax')) # 3개의 확률값은 1
# 다중분류 문제는 마지막 레이어의 activation = softmax, 주의: 모델 마지막 부분(output)은 y의 라벨값의 개수만큼

#3. 컴파일. 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics= ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'val_loss', patience = 55, mode = 'min',
                   verbose = 1,
                    restore_best_weights = True)

model.fit(x_train, y_train, epochs = 50, batch_size = 10,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])


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
# scaler = MinMaxScaler() 
loss :  [0.048989348113536835, 0.9722222089767456]
r2 스코어 :  0.9537032573414098
RMSE :  0.09766591
 
# scaler = StandardScaler() 
loss :  [0.09472034126520157, 0.9722222089767456]
r2 스코어 :  0.91675021702868
RMSE :  0.1324856

# scaler = MaxAbsScaler()
loss :  [0.05352781340479851, 1.0]
r2 스코어 :  0.9612235784240332
RMSE :  0.088002205

# scaler = RobustScaler()
loss :  [0.10054199397563934, 0.9722222089767456]
r2 스코어 :  0.9180810109954051
RMSE :  0.1320566

'''
