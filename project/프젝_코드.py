import numpy as np
import tensorflow as tf
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
'd:/study_data/_data/project/train/',
target_size=(100, 100),
batch_size=batch_size,
class_mode='sparse',
color_mode='grayscale',
shuffle=True
)


xy_test = test_datagen.flow_from_directory(
'd:/study_data/_data/project/test/',
target_size=(100, 100),
batch_size=batch_size,
class_mode='sparse',
color_mode='grayscale',
shuffle=True
)
