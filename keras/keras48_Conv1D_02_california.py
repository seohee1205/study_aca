from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, Input, Flatten
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets['target']


#1-2. x, y 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 333
)

print(x_train.shape, y_train.shape) # (16512, 8) (16512,)
print(x_test.shape, y_test.shape)   # (4128, 8) (4128,)


#1-3 scaler
scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(16512, 8, 1)
x_test = x_test.reshape(4128, 8, 1)

#2. 모델
input1 = Input(shape=(8, 1))
Conv1 = Conv1D(64,3, activation='linear')(input1)
Conv2 = Conv1D(26, 3, activation='relu')(Conv1)
Flat1 = Flatten()(Conv2)
dense1 = Dense(26, activation='relu')(Flat1)
dense2 = Dense(16, activation='relu')(dense1)
dense3 = Dense(12, activation='relu')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

# model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'loss', patience = 50, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 1000, batch_size = 120,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)


# result :  0.3110845685005188
# r2 스코어 :  0.7558355227011729