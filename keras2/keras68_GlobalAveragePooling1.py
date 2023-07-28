# GlobalAveragePooling2D는 일반적으로 CNN의 마지막 층으로 사용되며, 
# 이후에는 완전 연결 계층(Dense layer) 등을 추가하여 최종적인 출력을 생성gka
# 이를 통해 CNN은 이미지 분류나 객체 감지와 같은 작업에 적합한 특징 추출기로 사용될 수 있음

import numpy as np
from tensorflow.keras.datasets import mnist
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
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

# print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
                                                 # dtype=int64))

# 2차원으로 만들어주기(Scaler가 2차원에서만 되기 때문)
x_train = x_train.reshape(60000, 28*28 )
x_test = x_test.reshape(10000, 28*28)

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
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# Scaler랑 똑같음
# x_train = x_train/255.    # 실수형이니까 . 붙여주기
# x_test = x_test/255.


#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2, 2), padding = 'same', input_shape = (28, 28, 1)))
model.add(MaxPooling2D())   # 중첩되지 않은 부분 중 가장 큰 것
model.add(Conv2D(filters = 64, kernel_size = (2, 2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(33, 2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())     # 앞 필터의 개수만큼 노드로 만들어 줌
model.add(Dense(10, activation = 'softmax'))

model.summary()

# ######################## Flatten의 연산량 #########################
#  conv2d_2 (Conv2D)           (None, 12, 12, 33)        8481

#  flatten (Flatten)           (None, 4752)              0

#  dense (Dense)               (None, 10)                47530

# =================================================================
# Total params: 72,779
####################################################################
################# GlobalAveragePooling2D 연산량 ####################
#  conv2d_2 (Conv2D)           (None, 12, 12, 33)        8481

#  global_average_pooling2d (G  (None, 33)               0
#  lobalAveragePooling2D)

#  dense (Dense)               (None, 10)                340

# =================================================================
# Total params: 25,589


'''
#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

import time
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=128,
          validation_split= 0.2)
end = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('loss :', results[0])
print('acc :', results[1])
print('걸린시간 : ', end - start)

# Flatten   epochs 20
# loss : 0.062358539551496506
# acc : 0.9853000044822693
# 걸린시간 :  99.03180837631226

# Global    epochs 20
# loss : 0.3067144751548767
# acc : 0.9003999829292297
# 걸린시간 :  101.4152820110321

# Flatten epochs 30
# loss : 0.07573443651199341
# acc : 0.9873999953269958
# 걸린시간 :  145.07344913482666

# Global    epochs 50
loss : 0.1975974291563034
acc : 0.9381999969482422
걸린시간 :  207.33216500282288
'''