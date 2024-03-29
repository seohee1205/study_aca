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
        train_size = 0.8, shuffle = True, random_state= 1234)
# x_test -> x_train , y_test -> y_train 순

#2. 모델구성
model = Sequential()
model.add(Dense(24, input_dim = 1))
model.add(Dense(18))
model.add(Dense(5))
model.add(Dense(9))
model.add(Dense(6))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 200, batch_size = 3)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)            # x_test: 훈련 안 시킨 데이터들

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# r2 스코어 올리기 (0.99 정도)
# train_size = 0.8 / 24, 18, 5, 9, 6, 2, 1 / epochs = 200, batch_size = 3 -> 6/6 [==============================] - 0s 798us/step - loss: 15.0972
# 1/1 [==============================] - 0s 101ms/step - loss: 0.7492
# loss :  0.749151349067688
# 1/1 [==============================] - 0s 62ms/step
# r2스코어 :  0.981123746143059
