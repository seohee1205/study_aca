import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM


#1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = ?

x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
              [5, 6, 7],[6, 7, 8], [7, 8, 9]])
# y값을 지정해줘야 하기 때문에 10은 뺀다

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape)     #  (7, 3) (7,)

# RNN은 통상 3차원 데이터로 훈련, 
# [1, 2, 3] 훈련을 한다면 1 한 번, 2 한 번, 3 한 번 훈련한다

# x의 shape = (행, 열, 몇개씩 훈련하는지)   # 3차원
x = x.reshape(7, 3, 1)  # [[[1], [2], [3]], [[2], [3], [4], .......]]
print(x.shape)      # (7, 3, 1)
# RNN을 shape에 맞춰주기 위해 바꿈

#2. 모델
model = Sequential()
# model.add(SimpleRNN(40, input_shape = (3, 1)))
model.add(LSTM(32,  input_shape = (3, 1)))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 3000)

#4 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([8, 9, 10]).reshape(1, 3, 1)   # [[[8], [9], [10]]]
# print(x_predict.shape)      # (1, 3, 1)

result = model.predict(x_predict)
print('loss : ', loss)
print('[8, 9, 10]의 결과 : ', result)

# loss :  7.3340925155207515e-06
# [8, 9, 10]의 결과 :  [[10.627253]]