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
model.add(Dense(2, input_dim=1))
model.add(Dense(1))

model.summary()

print(model.weights)
###########################################
model.trainable = False     # ★★★        # False: 가중치가 저장되지 않음, 미분이 안 돼있음

# 이 기능 쓰는 이유는 시드가 고정돼도 틀어지는 경우가 발생하기 때문
# 전이학습, 사전학습
# 남이 만든 모델을 재학습할 필요가 없을 때 
# 사전 훈련된 모델을 사용하는 경우()
# 입출력 shape만 다시 하면 됨
# 중간부분, Flatten 이후 layer => Non-trainable

###########################################

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam')

model.fit(x, y, batch_size = 1, epochs= 10)

y_predict = model.predict(x)
print(y_predict)
# [[-0.7544218]
#  [-1.5088435]
#  [-2.2632656]
#  [-3.017687 ]
#  [-3.7721088]]


