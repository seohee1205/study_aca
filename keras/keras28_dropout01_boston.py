# 데이터를 많이 
# 로드의 일부를 뺀다 (drop out)
# How 성능 향상

# 저장할 때 평가결과값, 훈련시간 등을 파일에 넣어줘

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model, load_model
from tensorflow.python.keras.layers import Dense,Input, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x))      # <class 'numpy.ndarray'>
print(x)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 650
)

# 전처리 (정규화)는 데이터를 나눈 후 한다
scaler = MinMaxScaler()   # 하나로 모아줄 때  

x_train = scaler.fit_transform(x_train) # 위 두 줄과 같음
x_test = scaler.transform(x_test)

print(np.min(x_test), np.max(x_test)) # 0.0 1.0


#2. 함수형모델 구성
# input1 = Input(shape=(13,)) # 인풋명시, 
# dense1 = Dense(10)(input1)  # Dense 모델 구성 후, 이 모델은 어디에서 시작해서 어디에서 끝나는지 연결
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(5)(drop1)
# drop2 = Dropout(0.2)(dense2)
# dense3 = Dense(7)(drop2)
# drop3 = Dropout(0.5)(dense3)
# dense4 = Dense(2)(drop3)
# dense5 = Dense(4)(dense4)
# output1 = Dense(1)(dense5)
# model = Model(inputs = input1, outputs = output1)


# model.save('./_save/keras26_1_save_model.h5')

model = Sequential()
model.add(Dense(10, input_shape=(13,)))
model.add(Dropout(0.3))
model.add(Dense(5))
model.add(Dropout(0.2))
model.add(Dense(7))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

import datetime
date = datetime.datetime.now()
print(date)  # 2023-03-14 11:11:30.046663
date = date.strftime("%m%d_%H%M") # 시간을 문자로 (월, 일, 시간, 분)
print(date)  # 0314_1116

filepath = './_save/MCP/keras27_4/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5'



es = EarlyStopping(monitor= 'val_loss', patience= 10, mode = 'min',
                   verbose = 1, 
                   restore_best_weights= True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
        verbose = 1, 
        save_best_only= True,
        filepath="".join([filepath, 'k27_', date, '_', filename]))

# Shift + Tab = 왼쪽으로 이동 

model.fit(x_train, y_train, epochs = 10000,
          callbacks=[es, mcp],
          validation_split= 0.2)


# model = load_model('./_save/MCP/keras27_ModelCheckPoint1.hdf5')



#4. 평가, 예측
from sklearn.metrics import r2_score

print("====================== 1. 기본 출력 ========================")
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# print("====================== 3. MCP 출력 ========================")
# model3 = load_model('./_save/MCP/keras27_3_MCP.hdf5')
# loss = model3.evaluate(x_test, y_test, verbose=0)
# print('loss : ', loss)


'''
loss :  29.424924850463867
r2 스코어 :  0.5832944719187578
'''