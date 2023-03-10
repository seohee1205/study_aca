# 슬라이싱

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

# [실습] 넘파이 리스트의 슬라이싱 7:3으로 잘라라
x_train = x[0:7]    # =  x_train = x[:7]
print(x_train)      # [1 2 3 4 5 6 7]
x_test = x[7:10]    # = x_test = x[7:]
print(x_test)       # [ 8 9 10]
y_train = y[:7]     # = y_tran = y[0:7]
y_test = y[7:]      # = y_test = y[7:10]


print(x_train.shape, x_test.shape)  # (7,) (3,)
print(y_train.shape, y_test.shape)   # (7,) (3,)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([11])
print('[11]의 예측값 : ', result)
