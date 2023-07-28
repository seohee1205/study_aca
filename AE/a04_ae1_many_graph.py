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

model_01 = autoencoder(hidden_layer_size= 1) #pca 100% 성능
model_08 = autoencoder(hidden_layer_size= 8) #pca 100% 성능
model_32 = autoencoder(hidden_layer_size= 32) #pca 100% 성능
model_64 = autoencoder(hidden_layer_size= 64) #pca 100% 성능
model_154 = autoencoder(hidden_layer_size= 154) #pca 100% 성능
model_331 = autoencoder(hidden_layer_size=331) #pca 100% 성능
model_486 = autoencoder(hidden_layer_size=486) #pca 100% 성능
model_713 = autoencoder(hidden_layer_size=713) #pca 100% 성능
 
 
#3. 컴파일, 훈련
print("===================node 1개 시작 =========================")
model_01.compile(optimizer= 'adam', loss = 'mse')
model_01.fit(x_train_noised, x_train, epochs = 10, batch_size= 128)

print("===================node 8개 시작 =========================")
model_08.compile(optimizer= 'adam', loss = 'mse')
model_08.fit(x_train_noised, x_train, epochs = 10, batch_size= 128)

print("===================node 32개 시작 =========================")
model_32.compile(optimizer= 'adam', loss = 'mse')
model_32.fit(x_train_noised, x_train, epochs = 10, batch_size= 128)

print("===================node 64개 시작 =========================")
model_64.compile(optimizer= 'adam', loss = 'mse')
model_64.fit(x_train_noised, x_train, epochs = 10, batch_size= 128)

print("===================node 154개 시작 =========================")
model_154.compile(optimizer= 'adam', loss = 'mse')
model_154.fit(x_train_noised, x_train, epochs = 10, batch_size= 128)

print("===================node 331개 시작 =========================")
model_331.compile(optimizer= 'adam', loss = 'mse')
model_331.fit(x_train_noised, x_train, epochs = 10, batch_size= 128)

print("===================node 486개 시작 =========================")
model_486.compile(optimizer= 'adam', loss = 'mse')
model_486.fit(x_train_noised, x_train, epochs = 10, batch_size= 128)

print("===================node 713개 시작 =========================")
model_713.compile(optimizer= 'adam', loss = 'mse')
model_713.fit(x_train_noised, x_train, epochs = 10, batch_size= 128)

#4. 평가, 예측
decoded_imgs_01 = model_01.predict(x_test_noised)
decoded_imgs_08 = model_08.predict(x_test_noised)
decoded_imgs_32 = model_32.predict(x_test_noised)
decoded_imgs_64 = model_64.predict(x_test_noised)
decoded_imgs_154 = model_154.predict(x_test_noised)
decoded_imgs_331 = model_331.predict(x_test_noised)
decoded_imgs_486 = model_486.predict(x_test_noised)
decoded_imgs_713 = model_713.predict(x_test_noised)


###################################################################################

import matplotlib.pyplot as plt
import random

fig, axes = plt.subplots(9, 5, figsize=(15, 15))

random_image = random.sample(range(decoded_imgs_01.shape[0]), 5)
outputs = [x_test, decoded_imgs_01,decoded_imgs_08,
           decoded_imgs_154,decoded_imgs_331,decoded_imgs_32,
           decoded_imgs_64, decoded_imgs_713,decoded_imgs_486]


for row_num, row in enumerate(axes):
    for col_num,ax in enumerate(row):
        ax.imshow(outputs[row_num][random_image[col_num]].reshape(28,28),cmap='gray')
        
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()