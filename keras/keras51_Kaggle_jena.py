# https://www.kaggle.com/datasets/mnassrib/jena-climate

# loss = mse / metrics = mae    데이터 7:2:1  train(,validation) / test / predict(RMSE, r2)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
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

 
# x, y 분리
x = datasets.drop(['T (degC)'], axis = 1)
y = datasets['T (degC)']

print(x.shape, y.shape)         # (420551, 13) (420551,)



# train, test, predict 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=False, train_size=0.7, random_state=777)
x_test, x_predict, y_test, y_predict = train_test_split(
    x_test, y_test, shuffle=False, train_size=0.67, random_state=777)


print(x_train.shape, y_train.shape)     # (294385, 13), (294385,)
print(x_test.shape, y_test.shape)       # 84531, 13) (84531,)
print(x_predict.shape, y_predict.shape) # (41635, 13) (41635,)


dataset = np.array(range(1, 420551))
timesteps = 6


def split_x(dataset, timesteps):
    aaa = []    # aaa라는 빈 리스트를 만들어라
    for i in range(len(dataset) - timesteps):        
        subset = dataset[i : (i + timesteps)]               
        aaa.append(subset)                                  
    return np.array(aaa)    

x_train = split_x(x_train, timesteps)
x_test = split_x(x_test, timesteps)
x_predict = split_x(x_predict, timesteps)

y_train = y_train[timesteps:]
y_test = y_test[timesteps:]
y_predict = y_predict[timesteps:]


print(x_train.shape, y_train.shape)         # (294379, 6, 13) (294379,)
print(x_test.shape, y_test.shape)           # (84525, 6, 13) (84525,)
print(x_predict.shape, y_predict.shape)     # (41629, 6, 13) (41629,)



#2. 모델 구성
input1 = Input(shape=(6, 13))
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

es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 2000, batch_size = 777,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)              # RMSE 함수 사용
print("RMSE : ", rmse)

print('걸린 시간 : ', np.round(end-start, 2))



