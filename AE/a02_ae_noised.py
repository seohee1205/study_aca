# 자동제거, x를 x로 훈련시킨다. 준지도 학습.

import numpy as np
from tensorflow.keras.datasets import mnist
#1. 데이터
(x_train, _), (x_test, _) = mnist.load_data() #x로 훈련, 결과를 내기 위해

x_train = x_train.reshape(60000, 784).astype('float32')/255.
x_test = x_test.reshape(10000, 784).astype('float32')/255.

x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape)

# print(x_train.shape,x_test.shape) #(60000, 784) (10000, 784)

# print(np.max(x_train_noised), np.min(x_train_noised)) # 1.5007721999914518 -0.5842199637104563
# print(np.max(x_test_noised), np.min(x_test_noised))   # 1.4629771680998809 -0.5212620595820425
#음수가 나오는 이유
# 평균이 0이고 표준 편차가 0.1인 정규 분포에서 크기가 x_train.shape인 샘플을 생성하는 함수이기때문에
#random을 쓰지말고 uniform으로 써야됨.

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) #0보다 작으면 0, 1보다 크면 1 즉 0~1사이로 제한한다.
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
 
# print(np.max(x_train_noised), np.min(x_train_noised)) #1.0 0.0
# print(np.max(x_test_noised), np.min(x_test_noised))   #1.0 0.0

                ##### 아까 만든거 가지고 확인!! #####\
#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
# encoded = Dense(204, activation='relu')(input_img)
encoded = Dense(32, activation='relu')(input_img) #줄이는거
# encoded = Dense(1024, activation='relu')(input_img) #늘리는거
# encoded = Dense(1, activation='relu')(input_img) # 아예줄이기

# decoded = Dense(784, activation='linear')(encoded)
# decoded = Dense(784, activation='sigmoid')(encoded)
decoded = Dense(784, activation='relu')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)
autoencoder = Model(input_img, decoded)

#3. 컴파일, 훈련
autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noised,x_train, epochs =30, batch_size= 128,
                validation_split=0.2)

#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt
n =10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n, i+1)
    plt.imshow(x_test_noised[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()