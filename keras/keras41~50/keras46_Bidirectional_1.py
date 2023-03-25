import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.layers import Bidirectional

#1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = ?

x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7],[6, 7, 8], [7, 8, 9]])
y = np.array([4, 5, 6, 7, 8, 9, 10])
x = x.reshape(7, 3, 1)  # [[[1], [2], [3]], [[2], [3], [4], .......]]

# Bidirectional은 혼자서 작동이 안 됨
# 지정한 함수를 양방향으로 사용하게 해주는 함수
# Bidirectional은 RNN을 함께 해줘야 함
#2. 모델구성
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape = (3, 1)))
model.add(LSTM(10, return_sequences = True))
model.add(Bidirectional(GRU(10)))

model.summary()
