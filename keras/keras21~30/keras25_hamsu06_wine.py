# 사이킷런 load_wine

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
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

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(50, activation = 'relu', input_dim = 13))
# model.add(Dense(40, activation = 'relu'))
# model.add(Dense(40, activation = 'relu'))
# model.add(Dense(10, activation = 'relu'))
# model.add(Dense(3, activation = 'softmax')) # 3개의 확률값은 1
# # 다중분류 문제는 마지막 레이어의 activation = softmax, 주의: 모델 마지막 부분(output)은 y의 라벨값의 개수만큼


#(함수형) 모델 구성
input1 = Input(shape=(13,)) # 인풋명시, 
dense1 = Dense(50)(input1)  # Dense 모델 구성 후, 이 모델은 어디에서 시작해서 어디에서 끝나는지 연결
dense2 = Dense(40, activation = 'relu')(dense1)
dense3 = Dense(40, activation = 'relu')(dense2)
dense4 = Dense(10, activation = 'relu')(dense3)
output1 = Dense(3, activation = 'softmax')(dense4)
model = Model(inputs = input1, outputs = output1)


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
loss :  [10.297672271728516, 0.3888888955116272]
r2 스코어 :  -160.97555853066154
RMSE :  0.6338467
 
# scaler = StandardScaler() 
loss :  [6.268148422241211, 0.3333333432674408]
r2 스코어 :  -14.379368516180909
RMSE :  1.1132692

# scaler = MaxAbsScaler()
loss :  [5.3726983070373535, 0.3888888955116272]
r2 스코어 :  -105.08713152734788
RMSE :  0.8404218

# scaler = RobustScaler()
loss :  [1.0201411247253418, 0.0]
r2 스코어 :  -5.1421363828687445
RMSE :  0.9536042

'''
