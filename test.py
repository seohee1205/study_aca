import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
print(y_train.shape, y_test.shape)  # (50000, 1) (10000, 1)

#1-2. 원핫
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

#1-3. 스케일
Scale = MinMaxScaler()
x_train = Scale.fit_transform(x_train)
x_test = Scale.transform(x_test)

print(x_train.shape, x_test.shape)  # (50000, 3072) (10000, 3072)
print(y_train.shape, y_test.shape)  # (50000, 10) (10000, 10)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

#2. 모델
model = Sequential()
model.add(Conv2D(16, (2, 2), padding = 'same', strides = (2, 2), input_shape = (32, 32, 3)))
model.add(Conv2D(8, (2, 2)))
model.add(Flatten())
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일 및 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1, batch_size = 16)

#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis = 1)
print(y_predict)

y_test_acc = np.argmax(y_test, axis = 1)
print(y_test_acc)

acc = accuracy_score(y_test_acc, y_predict)

print(loss, acc)