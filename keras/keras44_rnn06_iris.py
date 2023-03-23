from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, GRU, Input, Dropout, MaxPooling2D
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import pandas as pd
import time

#1. 데이터
datasets = load_iris()

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (150, 4) (150,)

#1-1. x, y 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 777
)

print(x_train.shape, y_train.shape)     # (120, 4) (120,)
print(x_test.shape, y_test.shape)       # (30, 4) (30,)

# 2차원으로 만들어주기(Scaler가 2차원에서만 되기 때문)
x_train = x_train.reshape(120, 4*1*1)
x_test = x_test.reshape(30, 4*1*1)

# scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)

#2. 함수형모델 구성
input1 = Input(shape=(4, 1))
GRU1 = GRU(54, activation='relu',  return_sequences = True)(input1)
GRU2 = GRU(34, activation='relu',  return_sequences = True)(GRU1)
GRU3 = GRU(24)(GRU2)
dense1 = Dense(16,activation='relu')(GRU3)
dense2 = Dense(12,activation='relu')(dense1)
output1 = Dense(1)(dense2)
model = Model(inputs=input1, outputs=output1)

# model.summary()

#3. 컴파일, 훈련
start = time.time()
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'loss', patience = 60, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 1000, batch_size = 500,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

end = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)
print('걸린 시간 : ', np.round(end-start, 2))


# result :  0.03998696804046631
# r2 스코어 :  0.9458822969777595
# 걸린 시간 :  48.93