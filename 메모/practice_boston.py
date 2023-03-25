import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv1D, Flatten, LSTM
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(x.shape, y.shape)     # (506, 13) (506,)

#1-1 x, y 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 777
)

print(x_train.shape, y_train.shape)     # (404, 13) (404,)
print(x_test.shape, y_test.shape)       # (102, 13) (102,)

# 1-2 Scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#1-3. 모델에서 LSTM사용하기 위해서는 2차원을 3차원으로 바꿔줘야함
x_train = x_train.reshape(404, 13, 1)
x_test = x_test.reshape(102, 13, 1)


#2. 모델 구성
input1 = Input(shape=(13, 1))
LSTM1 = LSTM(50, activation = 'relu')(input1)
dense1 = Dense(28, activation = 'relu')(LSTM1)
dense2 = Dense(26, activation = 'relu')(dense1)
dense3 = Dense(10, activation = 'relu')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs = input1, outputs = output1)


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto',
                   verbose = 1, restore_best_weights=True)

mcp = ModelCheckpoint(monitor = 'val_loss', mode = 'auto',
                      verbose = 1, save_best_only= True,
                      filepath="".join([filepath, 'k27_', data, '_', filename]))

model.fit(x_train, y_train, epochs = 1000, batch_size = 500,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

