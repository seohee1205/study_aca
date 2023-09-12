import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
# print(y_train.shape, y_test.shape)  # (60000,) (10000,)

print(np.unique(y_train))   # [0 1 2 3 4 5 6 7 8 9]    # y 클래스 개수 10개
print(y_train)  # [5 0 4 ... 5 6 8]

# 원핫
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train)
# [[0. 0. 0. ... 0. 0. 0.]
#  [1. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 1. 0.]]

# scale 전 reshape
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

# scale
scale = MinMaxScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

# 다시 reshape
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#2. 모델
model = Sequential()
model.add(Conv2D(16, (2, 2), strides= (2, 2), input_shape = (28, 28, 1)))
model.add(Conv2D(10, (2, 2)))
model.add(Flatten())
model.add(Dense(8))
model.add(Dense(10, activation= 'softmax'))

#3. 컴파일 및 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 2, batch_size = 16)

#4. 평가 및 예측
result = model.evaluate(x_test, y_test)

y_predict = np.argmax(model.predict(x_test), axis = 1)
y_acc = np.argmax(y_test, axis = 1)

acc = accuracy_score(y_acc, y_predict)

print('acc : ', acc)