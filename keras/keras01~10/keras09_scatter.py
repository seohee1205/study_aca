from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])

# print(x.shape)          # (20,)
# print(y.shape)          # (20,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size = 0.7, shuffle = True, random_state= 1234)
# x_test -> x_train , y_test -> y_train 순

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 2000, batch_size = 3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)    # x 전체데이터를 넣은 모델의 예측을 y_pred로 정함

import matplotlib.pyplot as plt

# 시각화
plt.scatter(x, y)
# plt.scatter(x, y_predict)
plt.plot(x, y_predict, color = 'red')
plt.show()
