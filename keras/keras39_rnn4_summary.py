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
model = Sequential()                  #  [batch(행의 크기 / 데이터의 개수), timesteps(열 / 몇 개씩 자를 건지), feature(몇 개씩 훈련할 건지)]
model.add(SimpleRNN(10, input_shape = (5, 1)))      # input_shape = 행 빼고 나머지

# units * (feature + bias + units) = params       # 전체 param의 개수   10 * (1 + 1 + 10)

model.add(Dense(7, activation = 'relu'))
model.add(Dense(1))
model.summary()

# Model: "sequential"
# _____________________________________________________________            
#  Layer (type)                Output Shape              Param #                       ( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1(bias) * unit 개수 )
# =================================================================     
#  simple_rnn (SimpleRNN)      (None, 10)                120                            (10 * 10) + (1 * 10) + (1 * 10)     =  120

#  dense (Dense)               (None, 7)                 77   
             
#  dense_1 (Dense)             (None, 1)                 8

# =================================================================



'''
#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor = 'loss', patience = 20, mode = 'auto',
                   verbose = 1, restore_best_weights= True)
model.fit(x, y, epochs = 10000, callbacks = [es])

#4 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([6, 7, 8, 9, 10]).reshape(1, 5, 1)   # [[[7], [8], [9], [10]]]
# print(x_predict.shape)      # (1, 5, 1)

result = model.predict(x_predict)
print('loss : ', loss)
print('[6, 7, 8, 9, 10]의 결과 : ', result)

# [6, 7, 8, 9, 10]의 결과 :  [[10.64599]]
# [6, 7, 8, 9, 10]의 결과 :  [[10.76299]]

'''