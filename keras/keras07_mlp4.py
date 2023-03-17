import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10), range(21, 31), range(201, 211)])      #0부터 9까지 / 21부터 30까지 / 201부터 210까지

# print(x)
print(x.shape)  #(3, 10)
x = x.T     # (10, 3)

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])     # (1, 10)
y = y.T     # (10, 1)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 3))     # 열이 3개니까 (열 = 피처 = 컬럼 = 특성)
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))         # 열과 output layer은 동일해야 함

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 50, batch_size = 3)        # 1 epochs당 훈련 10번 총 500번 돌아감 

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[9, 30, 210]])                 # w1*x1 + w2*x2 = 20
print('[[9, 30, 210]]의 예측값 : ', result)
