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

x_val = x[13:16]    # [14, 15, 16]
print(x_val)
x_test = x[10:13]   # [11 12 13]
print(x_test)

y_val = y[13:16]    # [14 15 16]
print(y_val)
y_test = y[10:13]    # [11 12 13]
print(y_test)


'''
#2. 모델구성
model = Sequential()
model.add(Dense(5, activation='linear', input_dim = 1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer= 'adam')
model.fit(x_train, y_train, epochs = 20, batch_size= 2,
          validation_data= (x_val, y_val))          # 훈련하고 검증하고 훈련하고 검증하고 ...(반복)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

result = model.predict([17])
print('17의 예측값 : ', result)

'''