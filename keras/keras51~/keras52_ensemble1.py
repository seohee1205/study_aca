#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)])       # 삼성, 아모레
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)])
# 온도, 습도, 강수량
print(x1_datasets.shape)    # (2, 100)
print(x2_datasets.shape)    # (3, 100)

#1-1. 행, 열 바꾸기
x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
print(x1.shape)    # (100, 2)
print(x2.shape)    # (100, 3)

y = np.array(range(2001, 2101))  # 환율

#1-2. train, test 분리
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1, x2, y, train_size = 0.7, random_state= 333 
)
# y_train, y_test = train_test_split(
#     y, train_size = 0.7, random_state= 333 
# )

print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)    # (70, 3) (30, 3)
print(y_train.shape, y_test.shape)      # (70,) (30,)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape = (2,))
dense1 = Dense(35, activation = 'relu', name = 'stock1')(input1)
dense2 = Dense(24, activation = 'relu', name = 'stock2')(dense1)
dense3 = Dense(12, activation = 'relu', name = 'stock3')(dense2)
output1 = Dense(11, activation = 'relu', name = 'output1')(dense3)

#2-2. 모델2
input2 = Input(shape = (3,))
dense11 = Dense(10, name = 'weather1')(input2)
dense12 = Dense(16, name = 'weather2')(dense11)
dense13 = Dense(12, name = 'weather3')(dense12)
dense14 = Dense(22, name = 'weather4')(dense13)
output2 = Dense(11, name = 'output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate     # 사슬처럼 잇다 / # 소문자: 함수, 대문자: class
merge1 = concatenate([output1, output2], name = 'mg1')    # 리스트 형태로 받아들임
merge2 = Dense(12, activation= 'relu', name = 'mg2')(merge1)
merge3 = Dense(3, activation= 'relu', name = 'mg3')(merge2)
last_output = Dense(1, name = 'last')(merge3)

model = Model(inputs = [input1, input2], outputs = last_output)

# 큰 모델로 봤을 때, input과 output만 맞게 하면 됨 (중간 모델의 아웃풋은 노상관~)
# model.summary()

# Model: "model"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_2 (InputLayer)           [(None, 3)]          0           []

#  input_1 (InputLayer)           [(None, 2)]          0           []

#  weather1 (Dense)               (None, 10)           40          ['input_2[0][0]']

#  stock1 (Dense)                 (None, 35)           105         ['input_1[0][0]']

#  weather2 (Dense)               (None, 16)           176         ['weather1[0][0]']

#  stock2 (Dense)                 (None, 24)           864         ['stock1[0][0]']

#  weather3 (Dense)               (None, 12)           204         ['weather2[0][0]']

#  stock3 (Dense)                 (None, 12)           300         ['stock2[0][0]']

#  weather4 (Dense)               (None, 22)           286         ['weather3[0][0]']

#  output1 (Dense)                (None, 11)           143         ['stock3[0][0]']

#  output2 (Dense)                (None, 11)           253         ['weather4[0][0]']

#  mg1 (Concatenate)              (None, 22)           0           ['output1[0][0]',
#                                                                   'output2[0][0]']

#  mg2 (Dense)                    (None, 12)           276         ['mg1[0][0]']

#  mg3 (Dense)                    (None, 3)            39          ['mg2[0][0]']

#  last (Dense)                   (None, 1)            4           ['mg3[0][0]']

# ==================================================================================================

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
import time
start = time.time()

model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

model.fit([x1_train, x2_train], y_train, epochs = 2000,
          batch_size = 6, validation_split = 0.2, callbacks = [es])

end = time.time()

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_squared_error

result = model.evaluate([x1_test, x2_test],
                      y_test)
print('result : ', result)

y_predict = model.predict([x1_test, x2_test])

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)              # RMSE 함수 사용
print("RMSE : ", rmse)

print('걸린 시간 : ', np.round(end-start, 2))


# r2 스코어 :  0.9999942644775116
# RMSE :  0.058180782117433105
# 걸린 시간 :  48.58