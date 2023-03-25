# return_sequences: LSTM에서 다음층으로 2차원이 아닌 3차원으로 던져준다 -> 연속된 층에 LSTM 사용 가능해짐
# return_sequencse 사용 ->LSTM, GRU 연속해서 사용할 때 씀

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping


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
model.add(LSTM(10, input_shape = (3, 1), return_sequences= True))
model.add(LSTM(11), return_sequences = True)
model.add(GRU(12))
model.add(Dense(1))

model.summary()


# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  lstm (LSTM)                 (None, 3, 10)             480

#  lstm_1 (LSTM)               (None, 11)                968

#  dense (Dense)               (None, 1)                 12

# =================================================================