# https://www.kaggle.com/datasets/mnassrib/jena-climate

# loss = mse / metrics = mae    데이터 7:2:1  train(,validation) / test / predict(RMSE, r2)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Conv1D, Input, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

#1. 데이터
path = './_data/kaggle_jena/'

datasets = pd.read_csv(path + 'jena_climate_2009_2016.csv', index_col = 0)
# print(datasets)     # [420551 rows x 14 columns]

# print(datasets.columns)   
# Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
#        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')

# print(datasets.info())      # 결측치 없음
print(datasets.describe())

print(datasets['T (degC)'])
print(datasets['T (degC)'].values)      # 판다스를 넘파이로
print(datasets['T (degC)'].to_numpy())  # 판다스를 넘파이로

# import matplotlib.pyplot as plt
# plt.plot(datasets['T (degC)'].values)
# plt.show()

dataset = np.array(range(1, 420551))
timesteps = 6

# 입력 데이터와 출력 데이터 생성
input_data, output_data = [], []
values = datasets.values
for i in range(len(values) - timesteps):
    input_data.append(values[i:i + timesteps, 1:])
    output_data.append(values[i + timesteps, 1:])

# 입력 데이터와 출력 데이터를 numpy array로 변환
input_data = np.array(input_data)
output_data = np.array(output_data)

# 입력 데이터와 출력 데이터 shape 출력
print('Input Data Shape:', input_data.shape)    # (420407, 144, 13)
print('Output Data Shape:', output_data.shape)  # (420407, 13)

# x, y 분리
x = datasets.drop(['T (degC)'], axis = 1)
y = datasets['T (degC)']

print(x.shape, y.shape)         # (420551, 13) (420551,)

# train, test, predict 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=False, train_size=0.7, random_state=777)
x_test, x_predict, y_test, y_predict = train_test_split(
    x_test, y_test, shuffle=False, train_size=0.1, random_state=777)

print(x_train.shape, y_train.shape)    # (294385, 13), (294385,)
print(x_test.shape, y_test.shape)      # (42056, 13), (42056,)
print(x_predict.shape, y_predict.shape)# (126110, 13), (126110,)


#2. 모델 구성
input1 = Input(shape=(13, 1))
Conv1 = Conv1D(64, 2, activation='linear')(input1)
Conv2 = Conv1D(26, 2, activation='relu')(Conv1)
Flat1 = Flatten()(Conv2)
dense2 = Dense(16, activation='relu')(Flat1)
dense3 = Dense(12, activation='relu')(dense2)
output1 = Dense(10, activation = 'relu')(dense3)
model = Model(inputs=input1, outputs=output1)
# model.summary()


#3. 컴파일, 훈련
start = time.time()
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

es = EarlyStopping(monitor = 'val_loss', patience = 5, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 2, batch_size = 64,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

end = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_predict)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)              # RMSE 함수 사용
print("RMSE : ", rmse)

print('걸린 시간 : ', np.round(end-start, 2))

