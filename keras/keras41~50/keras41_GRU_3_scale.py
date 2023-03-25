import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
             [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
             [9, 10, 11], [10, 11, 12],
             [20, 30, 40], [30, 40, 50], [40, 50, 60]])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x_predict = np.array([50, 60, 70])    # I want 80
print(x.shape)  # (13, 3)

x = x.reshape(13, 3, 1)


# [실습]

#2. 모델 구성
model = Sequential()
model.add(GRU(80, input_length = 3, input_dim = 1))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(52, activation = 'relu'))
model.add(Dense(42, activation = 'relu'))
model.add(Dense(24, activation = 'relu'))
model.add(Dense(26, activation = 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')


es = EarlyStopping(monitor = 'loss', patience = 22, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

# mcp = ModelCheckpoint(monitor='loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath= './_save/MCP/keras40_LSTM6_scale.hdf5')   # 가중치만
                          

model.fit(x, y, epochs = 2000, callbacks = [es,]) #mcp])



#4. 평가, 예측
loss = model.evaluate(x, y)
x_predict = np.array([50, 60, 70]).reshape(1, 3, 1)

result = model.predict(x_predict)
print('loss : ', loss)
print('[50, 60, 70]의 결과 : ', result)


# loss :  4.831842670682818e-05
# [50, 60, 70]의 결과 :  [[78.89703]]