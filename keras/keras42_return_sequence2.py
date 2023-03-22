# return_sequences: LSTM에서 다음층으로 2차원이 아닌 3차원으로 던져준다 -> 연속된 층에 LSTM 사용 가능해짐
# return_sequencse 사용 ->LSTM, GRU 연속해서 사용할 때 씀

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping
import time

#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
             [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
             [9, 10, 11], [10, 11, 12],
             [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x_predict = np.array([50, 60, 70])    # I want 80
print(x.shape)  # (13, 31 ,1)

x = x.reshape(13, 3, 1)


#2. 모델
model = Sequential()                  #  [batch(행의 크기 / 데이터의 개수), timesteps(열 / 몇 개씩 자를 건지), feature(몇 개씩 훈련할 건지)]
model.add(LSTM(10, input_shape = (3, 1), return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(GRU(10,  return_sequences = True))
model.add(GRU(18, ))
model.add(Dense(10))
model.add(Dense(1))

# LSTM은 Input 3차원, output 2차원 / GRU는 input 3차원, output 2차원 
# model.summary()

# RNN에서 받는 데이터는 시계열 데이터
# 시계열의 연속된 데이터에 대해서 만든 모델임
# 다만 리턴 시퀀스를 통해 다음 RNN 레이어에 던져주더라도
# 그것이 명확한 시계열 데이터라는 확증이 없기 때문에 조심해야 함

#3. 컴파일, 훈련
start = time.time()
model.compile(loss = 'mse', optimizer = 'adam')


es = EarlyStopping(monitor = 'loss', patience = 24, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

# mcp = ModelCheckpoint(monitor='loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath= './_save/MCP/keras40_LSTM6_scale.hdf5')   # 가중치만
                          

model.fit(x, y, epochs = 1000, callbacks = [es,]) #mcp])

end = time.time()

#4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([50, 60, 70]).reshape(1, 3, 1)

result = model.predict(x_predict)
print('loss : ', loss)
print('[50, 60, 70]의 결과 : ', result)
print('걸린 시간 : ', np.round(end-start, 2))


# # loss :  0.0762687474489212
# [50, 60, 70]의 결과 :  [[71.16838]]