import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image


#1. 함수 정의
train_datagen = ImageDataGenerator(
    rescale = 1./255
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/project/train/',
    target_size= (100, 100),
    batch_size = 20,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle = True          # shuffle = false면 앵그리만 가져오게 됨
)

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/project/test/',
    target_size= (100, 100),
    batch_size = 20,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle = True 
)


# 0 ~ 0.75 이하 : angry
# 0.75 초과 ~ 1.5 이하: sad
# 1.5 초과 ~ 2.25 이하: default
# 2.25 초과 ~ 3 이하: happy


# sigmoid는 0부터 1까지인데 감정을 0부터 3까지 정의했으니까 사용자 정의 함수를 사용해야함
# import keras.backend as K
# def custom_activation(x):
#     return K.clip(x, 0, 3)

# custom_activation.__name__ = 'custom_activation' # Add a __name__ attribute

# print(xy_train[0][1])     # [1. 1. 3. 2. 3. 3. 2. 3. 2. 3. 0. 2. 0. 3. 1. 0. 3. 2. 3. 0.]


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


model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(xy_train, epochs = 10,  steps_per_epoch = 1, verbose = 1 )



