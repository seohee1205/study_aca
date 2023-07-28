# 자동제거, x를 x로 훈련시킨다. 준지도 학습.

import numpy as np
from tensorflow.keras.datasets import mnist
#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data() #x로 훈련, 결과를 내기 위해

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.


#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
# encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(32, activation='relu')(input_img) #줄이는거
# encoded = Dense(1024, activation='relu')(input_img) #늘리는거
encoded = Dense(1, activation='relu')(input_img) # 아예줄이기

# decoded = Dense(784, activation='linear')(encoded)
decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='relu')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)
autoencoder = Model(input_img, decoded)

autoencoder.summary()
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 784)]             0

#  dense (Dense)               (None, 64)                50240

#  dense_1 (Dense)             (None, 784)               50960

# =================================================================
# Total params: 101,200
# Trainable params: 101,200
# Non-trainable params: 0
# _________________________________________________________________

#오토인코더의 고질적인 문제 : 사진이 뿌얘질수도 있음. -> 압축 -> 풀기를 하기때문에.
#학습자체가 문제있는 사진, 문제없는 사진 이렇게 학습.

# autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc'])
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

#3. 컴파일, 훈련
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train,x_train, epochs =30, batch_size= 128,
                validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n =10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


# 인코더 4개의 히든과
# 디코더 4개의 activation
# 그리고 2개의 컴파일-로스 부분의 경우의 수
# #총 32가지의 로스를 정의비교하고,
# 눈으로 결과치를 비교해볼것/ 뭐가 제일 좋은지?

'''
1024 - linear : 매우잘나옴.
64 - linear : 약간 흐림
1 - linear : 아예 안보임,

relu 1024 - relu : 잘나옴

'''