import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

def custom_activation(x):
    return K.clip(x, -0.5, 3.5)

custom_activation.name = 'custom_activation'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20

xy_train = train_datagen.flow_from_directory(
'c:/study/_data/sh/train/',
target_size=(100, 100),
batch_size=batch_size,
class_mode='sparse',
color_mode='grayscale',
shuffle=True
)


xy_test = test_datagen.flow_from_directory(
'c:/study/_data/sh/test/',
target_size=(100, 100),
batch_size=batch_size,
class_mode='sparse',
color_mode='grayscale',
shuffle=True
)

model = Sequential()
model.add(Conv2D(8, (2, 2), input_shape=(100, 100, 1), activation='relu'))
model.add(Conv2D(8, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation=lambda x: custom_activation(x)))

model.compile(loss='mse', optimizer='adam')
model.summary()

model.fit_generator(xy_train, epochs=10, steps_per_epoch=10, verbose=1,)
