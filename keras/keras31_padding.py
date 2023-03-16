from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten   #  Flatten : 평판화

#  컬러면 3, 흑백이면 1 (input_shape 마지막이)

model = Sequential()
model.add(Conv2D(7, (2, 2), 
                padding = 'same',       # shape 유지시켜주고 싶으면 same
                input_shape=(8, 8, 1)))   # 7장으로 늘린다 / (2, 2) = 자르는 크기 / input_dim = 이미지의 형태
                # 특성을 좁히기 위해 조각조각 (2,2)로 자를 거여 -> 출력 : (N, 8, 8, 7)이 됨     summary에서 확인 할 수 있음
                             # input_shape = batch_size, rows, columns, channels
            # (7 = 출력 채널, 2, 2 = 필터 크기, 1 = 입력 채널 rgb )
model.add(Conv2D(filters = 4, 
                kernel_size = (3, 3), 
                padding = 'valid',      # # shape 유지시켜주고 싶지 않으면 valid
                activation= 'relu'))   # 출력 : (N, 6, 6, 4)        4차원의 데이터를 받음, 출력도 4차원 

model.add(Conv2D(10, (2, 2)))            # 출력 : (N, 5, 5, 10)
model.add(Flatten())                    # 출력 : (N, 5*5*10) -> (N, 250)
model.add(Dense(32, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(3, activation= 'softmax'))


model.summary()   # 출력값(Output shape) 알 수 있음

'''
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 8, 8, 7)           35
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 6, 6, 4)           256
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 5, 5, 10)          170
_________________________________________________________________
flatten (Flatten)            (None, 250)               0
_________________________________________________________________
dense (Dense)                (None, 32)                8032
_________________________________________________________________
dense_1 (Dense)              (None, 10)                330
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 33
=================================================================
Total params: 8,856
Trainable params: 8,856
Non-trainable params: 0
_________________________________________________________________
'''