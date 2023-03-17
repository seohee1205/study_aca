#1. R2를 음수가 아닌 0.5 이하로 만들 것
#2. 데이터는 건들지 말 것
#3. 레이어는 인풋과 아웃풋 포함해서 7개 이상
#4. batch_size = 1 고정
#5. 히든레이어의 노드는 10개 이상 100개 이하
#6. train 사이즈 75% 고정
#7. epoch 100번 이상
#8. loss지표는 mse, mae
# [실습]


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
        train_size = 0.75, shuffle = True, random_state= 50)
# x_test -> x_train , y_test -> y_train 순

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)            # x_test: 훈련 안 시킨 데이터들

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# r2 스코어 올리기 (0.99 정도)
# train_size = 0.8 / 2, 18, 5, 9, 6, 2, 1 / epochs = 200, batch_size = 3 -> 6/6 [==============================] - 0s 798us/step - loss: 15.0972
# 1/1 [==============================] - 0s 101ms/step - loss: 0.7492
# loss :  0.749151349067688
# 1/1 [==============================] - 0s 62ms/step
# r2스코어 :  0.981123746143059
