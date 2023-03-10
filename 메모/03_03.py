import numpy as np
from tensorflow.keras.models import Sequntial
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([])

y = np.array([])

#2. 모델구성
model = Sequential()
model.add(Dense())


#3. 컴파일, 훈련
model.compile(loss = '', otimizer = 'adam')
model.fit(x, y, epochs = , batch_size)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[]])
print('[]의 예측값 : ', result)