# [실습] keras56_4 남자 여자에 noise를 넣는다.
# predict : 기미, 주근깨 제거
# 5개 사진 출력/ 원본, 노이즈, 아웃풋

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing import image

path = 'c:/study/sok.jpg'
save_path = 'c:/study/_save/men_women/'

x_train = np.load(save_path + 'keras56_7_x_train.npy')
x_test = np.load(save_path + 'keras56_7_x_test.npy')
y_train = np.load(save_path + 'keras56_7_y_train.npy')
y_test = np.load(save_path + 'keras56_7_y_test.npy')
img = np.array(image.load_img(path, target_size=(150,150)))

# print(x_train.shape, y_train.shape) #(718, 150, 150, 3) (718,)
x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape)
img_noised = img + np.random.normal(0, 0.1,size=img.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) #0보다 작으면 0, 1보다 크면 1
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
img_noised = np.clip(img_noised,a_min=0, a_max=1).reshape(1, 150, 150, 3)

#2. 모델 

def autoencoder():
    model = Sequential()
    #인코더
    model.add(Conv2D(64, (3,3), activation='relu',
                    padding='same', input_shape = (150, 150, 3)))
    model.add(MaxPooling2D())    #맥스풀링은 반토막! -> 디폴트는 (2,2)
    # model.add(Conv2D(32, (3,3), activation='relu',
    #                 padding='same'))
    # model.add(MaxPooling2D())    #(None, 7, 7, 8) 
    model.add(Conv2D(32, (3,3), activation='relu',
                     padding='same'))
    #디코더
    # model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D(size=(3,3)))  
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D()) #(None, 28, 28, 16)
    model.add(Conv2D(32, (3,3), activation='relu',
                     padding='same'))
    model.add(Conv2D(3, (3,3), activation='sigmoid', padding='same')) #(None, 28, 28, 1)
    model.summary()
    return model

model = autoencoder()
model.compile(optimizer= 'adam', loss = 'mse')

model.fit(x_train_noised, x_train, epochs = 1, batch_size= 128)

#4. 평가, 예측
decoded_imgs = model.predict(x_test_noised)

decoded_sok = model.predict(img_noised)
import matplotlib.pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5, ax16), (ax6, ax7, ax8, ax9, ax10, ax17),
      (ax11, ax12, ax13, ax14, ax15, ax18)) = \
        plt.subplots(3, 6, figsize=(20,7))

#이미지 다섯 개를 무작위로 고른다.
random_image = random.sample(range(decoded_imgs.shape[0]), 5)

#원본(입력) 이미지를 맨 위에 그린다.
for i,ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_image[i]])

    if i == 0:
        ax.set_ylabel("Input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 노이즈를 넣은 이미지
for i,ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_image[i]])
    if i == 0:
        ax.set_ylabel("Noise", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
# 오토인코더가 출력한 이미지를 아래에 그린다.
for i,ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_image[i]])
    if i == 0:
        ax.set_ylabel("Output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

#내사진
for i,ax in enumerate([ax16]):
    ax.imshow(img.reshape(150,150,3))
    if i == 0:
        ax.set_ylabel("input", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
for i,ax in enumerate([ax17]):
    ax.imshow(img_noised.reshape(150,150,3))
    if i == 0:
        ax.set_ylabel("noise", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
for i,ax in enumerate([ax18]):
    ax.imshow(decoded_sok.reshape(150,150,3))
    if i == 0:
        ax.set_ylabel("Output", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    

plt.tight_layout()
plt.show()