#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(8, (2, 2), input_shape= (100, 100, 1), activation= 'relu'))
model.add(Conv2D(8, (3, 3), activation= 'relu'))
model.add(Flatten())
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))
model.summary()