import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.python.keras.callbacks import EarlyStopping
import time


dataset = np.array(range(1, 101))    # 1부터 100까지
timesteps = 5           # 5개씩 잘라라
x_predict = np.array(range(96, 106))     # 100~106 예상값
# 96, 97, 98, 99
# 97, 98, 99, 100
# 98, 99, 100, 101
# ...
# 102, 103, 104, 105


def split_x(dataset, timesteps):
    aaa = []    # aaa라는 빈 리스트를 만들어라
    for i in range(len(dataset) - timesteps + 1):           # 반복 횟수 / 행의 개수 / 100 - 5 + 1 = 96 
        subset = dataset[i : (i + timesteps)]               # 반복할 내용 0부터 0+5 즉 0부터 5까지의 데이터셋을 섭셋에 저장
        aaa.append(subset)                                  # 섭셋을 aaa 리스트에 추가해라
    return np.array(aaa)                    

# aaa라는 리스트 공간을 만들고
# for 반복할 거야, (6번을) , 

bbb = split_x(dataset, timesteps)
print(bbb)
print(bbb.shape)    # (96, 5)

# x = bbb[:, :3]
x = bbb[:, :-1]
y = bbb[:, -1]

print(x.shape)      # (96, 4)
print(y.shape)      # (96,)

x_predict = split_x(x_predict, 4)
print(x_predict.shape)

x = x.reshape(96, 4, 1)  # (7, 4)
x_predict = x_predict.reshape(7, 4, 1)

#2. 모델
model = Sequential()
model.add(LSTM(10, input_shape = (4, 1), activation = 'linear'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
start = time.time()
model.compile(loss = 'mse', optimizer = 'adam')


es = EarlyStopping(monitor = 'loss', patience = 45, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

# mcp = ModelCheckpoint(monitor='loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath= './_save/MCP/keras40_LSTM6_scale.hdf5')   # 가중치만
                          

model.fit(x, y, epochs = 1000, callbacks = [es,]) #mcp])

end = time.time()

#4. 평가, 예측
loss = model.evaluate(x, y)

result = model.predict(x_predict)
print('loss : ', loss)
print('x_preidct의 결과 : ', result)
print('걸린 시간 : ', np.round(end-start, 2))


# x_preidct의 결과 :  [[ 99.99047 ]
#  [100.98782 ]
#  [101.984985]
#  [102.98192 ]
#  [103.978615]
#  [104.97509 ]
#  [105.97137 ]]
# 걸린 시간 :  44.41

