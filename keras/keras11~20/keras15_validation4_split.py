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
# 10:3:3

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size= 0.2, random_state = 123)



#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim = 1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs = 20, batch_size= 2,
        validation_split = 0.2)           # 훈련하고 검증하고 훈련하고 검증하고 ...(반복)
# ctrl + 스페이스바 = 예약어 볼 수 있음


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

result = model.predict([17])
print('17의 예측값 : ', result)

