import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping


#2. 모델
model = Sequential()                  #  [batch(행의 크기 / 데이터의 개수), timesteps(열 / 몇 개씩 자를 건지), feature(몇 개씩 훈련할 건지)]
model.add(GRU(10, input_shape = (5, 1)))      # input_shape = 행 빼고 나머지

# units * (feature + bias + units) = params       # 전체 param의 개수   10 * (1 + 1 + 10)

model.add(Dense(7, activation = 'relu'))
model.add(Dense(1))
model.summary()



# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #                        # 3 * (n^2 + nm + n)
# =================================================================                     # 3 * (다음 노드 수^2 + 다음 노드 수 * Shape의 feature + 다음 노드 수)
#  gru (GRU)                   (None, 10)                390                            # 3 * (10^2 + 1 * 10 + 10) = 360

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================




