#1. 데이터
import numpy as np
x = np.array([1, 2, 3]) # 배열하는 숫자
y = np.array([1, 2, 3])

#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential     # 텐서플로 케라스의 모델을 순차적으로 가져다 쓰겠다.
from tensorflow.keras.layers import Dense         # 텐서플로 케라스에 있는, 댄스모델을 가져오겠다. => y=wx + b

model = Sequential()               # 이 시퀀셜을 가져다 모델을 쓰겠다.
model.add(Dense(3, input_dim = 1))      # 댄스 레이어로 쌓고있다. 
model.add(Dense(6))     # 아웃풋, 인풋 / 상위에 있는 애가 인풋이기 때문에 명시하지 않아도 됨.
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')     # 손실 0? , loss를 최적화한다.
model.fit(x, y, epochs = 100)

#4. 평가, 예측
loss = model.evaluate(x, y)     # evaluate: 평가
print('loss : ', loss)

result = model.predict([4])
print("[4]의 예측값 : ", result)

