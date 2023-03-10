# x는 1개
# y는 3개

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)])      #0부터 9까지 / 21부터 30까지 / 201부터 210까지
print(x.shape)  #(1, 10)
x = x.T     # (10, 1)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])     # (3, 10)
y = y.T     # (10, 3)
print(y.shape)

# [실습]
# 예측 : [[9]] -> 예상 y값 [[10, 1,9, 0]]

#2. 모델구성
model = Sequential()
model.add(Dense(13, input_dim = 1))     # 열이 1개니까 (열 = 피처 = 컬럼 = 특성)
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(11))
model.add(Dense(12))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 200, batch_size = 3)        # 1 epochs당 훈련 3번 총 200번 돌아감 

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[9]])                 # w1*x1 + w2*x2 = 20
print('[[9]]의 예측값 : ', result)

#(1) 13, 12, 8, 10, 12, 10, 8, 4, 3 / eopchs = 200 -> loss: 1.8477e-12 = 0.0000000000018477
#(2) 13, 12, 8, 11, 12, 10, 8, 4, 3 / eopchs = 200 -> loss: 8.3669e-13 = 0.00000000000083669
