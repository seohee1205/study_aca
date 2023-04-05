#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301, 401)])       # 삼성, 아모레
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)])
x3_datasets = np.array([range(201, 301), range(511, 611), range(1300, 1400)])

# 온도, 습도, 강수량
print(x1_datasets.shape)    # (2, 100)
print(x2_datasets.shape)    # (3, 100)
print(x3_datasets.shape)    # (3, 100)

#1-1. 행, 열 바꾸기

x1 = np.transpose(x1_datasets)
x2 = x2_datasets.T
x3 = x3_datasets.T
print(x1.shape)    # (100, 2)
print(x2.shape)    # (100, 3)
print(x3.shape)    # (100, 3)

y = np.array(range(2001, 2101))  # 환율

#1-2. train, test 분리
from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(
    x1, x2, x3, y, train_size = 0.7, random_state= 333 
)
# y_train, y_test = train_test_split(
#     y, train_size = 0.7, random_state= 333 
# )

print(x1_train.shape, x1_test.shape)    # (70, 2) (30, 2)
print(x2_train.shape, x2_test.shape)    # (70, 3) (30, 3)
print(x3_train.shape, x3_test.shape)    # (70, 3) (30, 3)
print(y_train.shape, y_test.shape)      # (70,) (30,)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape = (2,))
dense1 = Dense(35, activation = 'swish', name = 'stock1')(input1)
dense2 = Dense(24, activation = 'swish', name = 'stock2')(dense1)
dense3 = Dense(12, activation = 'swish', name = 'stock3')(dense2)
output1 = Dense(11, activation = 'swish', name = 'output1')(dense3)

#2-2. 모델2
input2 = Input(shape = (3,))
dense11 = Dense(30, name = 'weather1')(input2)
dense12 = Dense(16, activation = 'swish', name = 'weather2')(dense11)
dense13 = Dense(52, activation = 'swish', name = 'weather3')(dense12)
dense14 = Dense(32, name = 'weather4')(dense13)
output2 = Dense(11, name = 'output2')(dense14)

#2-3. 모델3
input3 = Input(shape = (3,))
dense21 = Dense(20, activation = 'swish', name = 'weather11')(input3)
dense22 = Dense(24, activation = 'swish', name = 'weather22')(dense21)
dense23 = Dense(10, activation = 'swish', name = 'weather33')(dense22)
dense24 = Dense(8,  activation = 'swish', name = 'weather44')(dense23)
output3 = Dense(11, name = 'output3')(dense24)

from tensorflow.keras.layers import concatenate, Concatenate     # 사슬처럼 잇다 / # 소문자: 함수, 대문자: class
merge1 = concatenate([output1, output2, output3], name = 'mg1')    # 리스트 형태로 받아들임
merge2 = Dense(12, activation= 'swish', name = 'mg2')(merge1)
merge3 = Dense(3, activation= 'swish', name = 'mg3')(merge2)
last_output = Dense(1, name = 'last')(merge3)

model = Model(inputs = [input1, input2, input3], outputs = last_output)

# 큰 모델로 봤을 때, input과 output만 맞게 하면 됨 (중간 모델의 아웃풋은 노상관~)
# model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
import time
start = time.time()

model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

model.fit([x1_train, x2_train, x3_train], y_train, epochs = 2000,
          batch_size = 6, validation_split = 0.2, callbacks = [es])

end = time.time()

#4. 평가, 예측
from sklearn.metrics import r2_score, mean_squared_error

result = model.evaluate([x1_test, x2_test, x3_test],
                      y_test)
print('result : ', result)

y_predict = model.predict([x1_test, x2_test, x3_test])

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)              # RMSE 함수 사용
print("RMSE : ", rmse)

print('걸린 시간 : ', np.round(end-start, 2))


# r2 스코어 :  0.99998851286253
# RMSE :  0.08233774540896802
# 걸린 시간 :  16.55
