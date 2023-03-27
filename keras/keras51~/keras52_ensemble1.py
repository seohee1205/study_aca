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
dense1 = Dense(10, activation = 'relu', name = 'stock1')(input1)
dense2 = Dense(20, activation = 'relu', name = 'stock2')(dense1)
dense3 = Dense(30, activation = 'relu', name = 'stock3')(dense2)
output1 = Dense(1, activation = 'relu', name = 'output1')(dense3)

#2-2. 모델2
input2 = Input(shape = (3,))
dense11 = Dense(10, name = 'weather1')(input2)
dense12 = Dense(10, name = 'weather2')(dense11)
dense13 = Dense(10, name = 'weather3')(dense12)
dense14 = Dense(10, name = 'weather4')(dense13)
output2 = Dense(1, name = 'output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate     # 사슬처럼 잇다 / # 소문자: 함수, 대문자: class
merge1 = concatenate([output1, output2], name = 'mg1')    # 리스트 형태로 받아들임
merge2 = Dense(2, activation= 'relu', name = 'mg2')(merge1)
merge3 = Dense(3, activation= 'relu', name = 'mg3')(merge2)
last_output = Dense(1, name = 'last')(merge3)

model = Model(inputs = [input1, input2], outputs = last_output)

model.summary()

