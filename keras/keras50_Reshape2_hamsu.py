import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv2D, Flatten, Conv1D, MaxPooling2D, Reshape, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score, mean_squared_error
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)/225.
x_test = x_test.reshape(10000, 28, 28, 1)/225.

print(x_train.shape, y_train.shape)     # (60000, 28, 28, 1) (60000,)


#2. 모델
# model = Sequential()
# model.add(Conv2D(filters = 64, kernel_size = (3,3),
#                  padding = 'same', input_shape = (28, 28, 1)))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (3, 3)))
# model.add(Conv2D(10, 3))
# model.add(MaxPooling2D())
# model.add(Flatten())       # (N, 250)
# model.add(Reshape(target_shape = (25, 10)))
# model.add(Conv1D(10, 3, padding = 'same'))
# model.add(LSTM(784))
# model.add(Reshape(target_shape = (28, 28, 1)))
# model.add(Conv2D(32, (3, 3), padding = 'same'))
# model.add(Flatten())  
# model.add(Dense(10, activation = 'softmax'))

# model.summary()

#2. 함수형 모델
input1 = Input(shape = (28, 28, 1))
conv1 = Conv2D(64, (3, 3),
               padding= 'same',
               activation= 'relu')(input1)
mp1 = MaxPooling2D()(conv1)
conv2 = Conv2D(32, (3, 3))(mp1)
conv3 = Conv2D(10, 3)(conv2)
mp2 = MaxPooling2D()(conv3)
flat1 = Flatten()(mp2)
reshape1 = Reshape(target_shape= (25, 10))(flat1)
conv4 = Conv1D(10, 3, padding = 'same')(reshape1)
lstm = LSTM(784)(conv4)
reshape2 = Reshape(target_shape= (28, 28, 1))(lstm)
conv5 = Conv2D(32, (3, 3), padding = 'same')(reshape2)
flat2 = Flatten()(conv5)
output1 = Dense(10, activation= 'softmax')(flat2)
model = Model(inputs = input1, outputs = output1)

model.summary()

