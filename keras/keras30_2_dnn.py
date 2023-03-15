from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten 

model = Sequential()                        # (N, 3)
model.add(Dense(10, input_shape=(3,)))      # input_shape = (batch_size, input_dim)
model.add(Dense(units=15))                # output layer = units,  # 출력 (batch_size, units)
model.summary()