from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)     # (442, 10) (442,)

# [실습]
# R2 = 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size = 0.9, shuffle = True, random_state= 123)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 10))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 3000, batch_size = 100)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)         

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#4/4 [==============================] - 0s 1ms/step - loss: 2929.5447
# 2/2 [==============================] - 0s 15ms/step - loss: 2387.2019
# loss :  2387.201904296875
# 2/2 [==============================] - 0s 998us/step
# r2스코어 :  0.6420389358966216
