from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1, 17))    # 스칼라 10개짜리 벡터 1개
y = np.array(range(1, 17))
# x_val = np.array([14, 15, 16])
# y_val = np.array([14, 15, 16])
# x_test = np.array([11, 12, 13])
# y_test = np.array([11, 12, 13])

### 실습 : 잘라보자 ###
# train_test_split 로만 잘라라
# 10:3:3 (train, val, test)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.625, random_state = 1)
# 먼저 10:6으로 나눈 후 6을 3:3을 나눈다.

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, train_size = 0.5, random_state = 1)

print(x_train, x_val, x_test)
# [ 5  2 15  1 16 10  9 13 12  6] [ 4  3 11] [ 8 14  7]


#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim = 1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs = 20, batch_size= 2,
          validation_data = (x_val, y_val)          # 훈련하고 검증하고 훈련하고 검증하고 ...(반복)
)
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

result = model.predict([17])
print('17의 예측값 : ', result)
