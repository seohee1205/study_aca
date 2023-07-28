import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import tensorflow as tf
tf.random.set_seed(337)


#1. 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

#2. 모델
model = Sequential()
model.add(Dense(3, input_dim=1))
# [[-1.175985  , -0.20179522, -1.1501358 ]]           # dense/kernel에서 kernel = weight  (layer에서 kernel = weight)
model.add(Dense(2))
model.add(Dense(1))

model.summary()

###########################################
model.trainable = False     # ★★★        # False: 가중치가 저장되지 않음, 미분이 안 돼있음
###########################################

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

model.fit(x, y, batch_size = 1, epochs= 50)

y_predict = model.predict(x)
print(y_predict)
# [[1.3275863]
#  [2.168451 ]
#  [3.0093157]
#  [3.8501809]
#  [4.6910458]]


