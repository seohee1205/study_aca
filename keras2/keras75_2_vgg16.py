import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

# model = VGG16()   # include_top True, input_shape = (224, 224, 3)     # 디폴트 상태
vgg16 = VGG16(weights= 'imagenet', include_top=False,
              input_shape=(32, 32, 3)
              )

vgg16.trainable = False     # 가중치 동결

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation = 'softmax'))

# model.trainable = True

model.summary()

print(len(model.weights))               
print(len(model.trainable_weights))     

# Trainable: True // model False //  VGG False
#            30   //   30     //    30
#            30   //   0     //     4

