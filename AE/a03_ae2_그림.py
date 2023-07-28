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

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) #0보다 작으면 0, 1보다 크면 1
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
 
# print(np.max(x_train_noised), np.min(x_train_noised)) #1.0 0.0
# print(np.max(x_test_noised), np.min(x_test_noised))   #1.0 0.0

#2. 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape = (784,)))
    model.add(Dense(784, activation='sigmoid'))
    
    return model

model = autoencoder(hidden_layer_size=154) #pca 95% 성능
# model = autoencoder(hidden_layer_size=331) #pca 99% 성능
# model = autoencoder(hidden_layer_size=486) #pca 99.9% 성능
# model = autoencoder(hidden_layer_size=713) #pca 100% 성능
 
model.compile(optimizer= 'adam', loss = 'mse')

model.fit(x_train_noised, x_train, epochs = 30, batch_size= 128)

#4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

###################################################################################

import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
        plt.subplots(3, 5, figsize=(20,7))

#이미지 다섯 개를 무작위로 고른다.
random_image = random.sample(range(decoded_imgs.shape[0]), 5)

#원본(입력) 이미지를 맨 위에 그린다.
for i,ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_image[i]].reshape(28,28),cmap='gray')
    if i == 0:
        ax.set_ylabel("Input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 노이즈를 넣은 이미지
for i,ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_image[i]].reshape(28,28),cmap='gray')
    if i == 0:
        ax.set_ylabel("Noise", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 오토인코더가 출력한 이미지를 아래에 그린다.
for i,ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_image[i]].reshape(28,28),cmap='gray')
    if i == 0:
        ax.set_ylabel("Output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()