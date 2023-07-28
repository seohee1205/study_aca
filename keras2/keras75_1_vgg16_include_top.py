import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

# model = VGG16()   # include_top True, input_shape = (224, 224, 3)     # 디폴트 상태
model = VGG16(weights= 'imagenet', include_top=False,
              input_shape=(32, 32, 3)
              )

model.summary()

print(len(model.weights))   # 32 -> 26
print(len(model.trainable_weights))   # 32  -> 26

###################### include_top = True #########################
#1. FC layer 원래거 쓴다
#2. inpit_shape = (224, 224, 3) 고정값, - 바꿀 수 없다

# input_1 (InputLayer)        [(None, 224, 224, 3)]     0
# block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
# ......
# flatten (Flatten)           (None, 25088)             0
# fc1 (Dense)                 (None, 4096)              102764544
# fc2 (Dense)                 (None, 4096)              16781312
# predictions (Dense)         (None, 1000)              4097000\
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

###################### include_top = False #########################
#1. FC layer 원래거 삭제 -> 나는야 커스터마이징
#2. inpit_shape = (32, 32, 3) 고정값, - 바꿀 수 있다 -> 나는야 커스터마이징

# input_1 (InputLayer)        [(None, 32, 32, 3)]       0
# block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792
# ......
# Flatten 하단부분 (풀리커넥티드 레이어부분) 아디오스!!
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0