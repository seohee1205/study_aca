from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (20640, 8) (20640,)

# [실습]
# R2 = 0.55 ~ 0.6 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size = 0.8, shuffle = True, random_state= 475)
# x_test -> x_train , y_test -> y_train 순

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 860,
          validation_split = 0.2)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)            # x_test: 훈련 안 시킨 데이터들

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


#  loss :  1.4018065929412842
# r2스코어 :  -0.036320431830559885
