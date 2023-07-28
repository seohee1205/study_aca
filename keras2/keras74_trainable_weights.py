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
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 3)                 6
# _________________________________________________________________
# dense_1 (Dense)              (None, 2)                 8
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 3
# =================================================================
# Total params: 17
# Trainable params: 17
# Non-trainable params: 0
# _________________________________________________________________

print(model.weights)
print("================================")
print(model.trainable_weights)
print("================================")

print(len(model.weights))             # 6   layer 개수 * 2
print(len(model.trainable_weights))   # 6

###########################################
model.trainable = False     # ★★★
###########################################

print(len(model.weights))             # 6
print(len(model.trainable_weights))   # 0

print("================================")
print(model.weights) 
print("================================")
print(model.trainable_weights)   # []
print("================================")

model.summary()


