import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


#1. 데이터
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #                y = wx + b
# =================================================================             param = w와 b까지 합쳐준 것
# dense (Dense)                (None, 5)                 10                   (뒤에 1개가 더 있다고 생각하여 계산)
# _________________________________________________________________         
# dense_1 (Dense)              (None, 4)                 24                 
# _________________________________________________________________
# dense_2 (Dense)              (None, 3)                 15                
# _________________________________________________________________
# dense_3 (Dense)              (None, 2)                 8                   
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 3
# =================================================================


