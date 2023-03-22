import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping


#1. 데이터


#2. 모델
model = Sequential()                  #  [batch(행의 크기 / 데이터의 개수), timesteps(열 / 몇 개씩 자를 건지), feature(몇 개씩 훈련할 건지)]
                                      #  [batch, input_length, input_dim] 
model.add(LSTM(10, input_shape = (5, 1)))      # input_shape = 행 빼고 나머지
# model.add(LSTM(10, input_length = 5, input_dim = 1))    # input_shape가 먹히지 않는 상황이 있다. 그럴 때 인풋 렝스와 딤을 사용해야 할 때가 있다.
# model.add(LSTM(10, input_dim = 1, input_length = 5))    # 약간 가독성 떨어짐 (비추)


# units * (feature + bias + units) = params       # 전체 param의 개수   10 * (1 + 1 + 10)

model.add(Dense(7, activation = 'relu'))
model.add(Dense(1))
model.summary()



