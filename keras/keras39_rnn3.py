import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = ?

x = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8],
              [5, 6, 7, 8, 9]])
# y값을 지정해줘야 하기 때문에 10은 뺀다

y = np.array([6, 7, 8, 9, 10])

print(x.shape, y.shape)     #  (6, 5) (5,)

# RNN은 통상 3차원 데이터로 훈련, 
# [1, 2, 3] 훈련을 한다면 1 한 번, 2 한 번, 3 한 번 훈련한다

# x의 shape = (행, 열, 몇개씩 훈련하는지)   # 3차원
x = x.reshape(5, 5, 1)  # [[[1], [2], [3]], [[2], [3], [4], .......]]
print(x.shape)      # (5, 5, 1)
# RNN을 shape에 맞춰주기 위해 바꿈

#2. 모델
model = Sequential()
model.add(SimpleRNN(64, input_shape = (5, 1)))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
# es = EarlyStopping(monitor = 'loss', patience = 45, mode = 'auto',
#                    verbose = 1, restore_best_weights= True)

import time
start = time.time()
model.fit(x, y, epochs = 10000) #callbacks = [es])
end = time.time()

#4 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([6, 7, 8, 9, 10]).reshape(1, 5, 1)   # [[[7], [8], [9], [10]]]
# print(x_predict.shape)      # (1, 5, 1)

result = model.predict(x_predict)
print('loss : ', loss)
print('[6, 7, 8, 9, 10]의 결과 : ', result)
print('걸린시간 :', round(end - start, 2))

# [6, 7, 8, 9, 10]의 결과 :  [[10.64599]]
# [6, 7, 8, 9, 10]의 결과 :  [[10.76299]]

# loss :  1.70823659573216e-05
# [6, 7, 8, 9, 10]의 결과 :  [[10.326052]]
# 걸린시간 : 41.65