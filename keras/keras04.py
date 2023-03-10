import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 5, 4])

# [실습] [6.00]을 예측한다.

#2. 모델 구축
model = Sequential()
model.add(Dense(8, input_dim = 1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))


# 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')     # 손실 0? , loss를 최적화한다.
model.fit(x, y, epochs = 50)


#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([6])
print("[6]의 예측값 : ", result)

