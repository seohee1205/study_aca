# cifar100으로 바꿔서 Gap이 이기면 끝
import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
tf.random.set_seed(337)


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

# print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
                                                 # dtype=int64))

# 2차원으로 만들어주기(Scaler가 2차원에서만 되기 때문)
x_train = x_train.reshape(50000, 32*32*3 )
x_test = x_test.reshape(10000, 32*32*3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train)  # [5 0 4 ... 5 6 8]
# print(y_train.shape)    # (60000, 10)
# print(y_test)   # [7 2 1 ... 4 5 6]
# print(y_test.shape)     # (10000, 10)

Scaler = MinMaxScaler()     # 2차원에서만 됨
Scaler.fit(x_train) 
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)

# 이미지는 4차원이니까 다시 4차원으로 바꿔주기
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# Scaler랑 똑같음
# x_train = x_train/255.    # 실수형이니까 . 붙여주기
# x_test = x_test/255.


#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2, 2), padding = 'same', input_shape = (32, 32, 3)))
model.add(MaxPooling2D())   # 중첩되지 않은 부분 중 가장 큰 것
model.add(Conv2D(filters = 64, kernel_size = (2, 2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(32, 2))
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100, activation = 'softmax'))

# model.summary()

# ######################## Flatten의 연산량 #########################
#  conv2d_2 (Conv2D)           (None, 14, 14, 32)        8224

#  flatten (Flatten)           (None, 6272)              0

#  dense (Dense)               (None, 16)                100368

#  dense_1 (Dense)             (None, 8)                 136

#  dense_2 (Dense)             (None, 10)                90

# =================================================================
# Total params: 126,098
####################################################################
################# GlobalAveragePooling2D 연산량 ####################
#  conv2d_2 (Conv2D)           (None, 14, 14, 32)        8224

#  global_average_pooling2d (G  (None, 32)               0
#  lobalAveragePooling2D)

#  dense (Dense)               (None, 16)                528

#  dense_1 (Dense)             (None, 8)                 136

#  dense_2 (Dense)             (None, 10)                90

# =================================================================
# Total params: 26,258


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

import time
start = time.time()
model.fit(x_train, y_train, epochs=30, batch_size=128,
          validation_split= 0.2)
end = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])
print('걸린시간 : ', end - start)


# Flatten   epochs 30
# oss : 10.759369850158691
# acc : 0.2840000092983246
# 걸린시간 :  124.37365245819092

# Global    epochs 30
# loss : 3.044166088104248
# acc : 0.2531999945640564
# 걸린시간 :  126.33325242996216
