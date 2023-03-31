# 수치형으로 제공된 데이터를 증폭

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale= 1./255,
    horizontal_flip= True,
    vertical_flip= True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    rotation_range= 5,
    zoom_range= 0.1,
    shear_range= 0.7,
    fill_mode= 'nearest'
)

augment_size = 100

print(x_train.shape)       # (60000, 28, 28)
print(x_train[0].shape)    # (28, 28)
print(x_train[1].shape)    # (28, 28) 
print(x_train[0][0].shape)  # (28,)

print(np.tile(x_train[0].reshape(28*28),    # 수치를 증폭시키기 위해 shape 바꿔줌
              augment_size).reshape(-1, 28, 28, 1).shape)     # (100, 28, 28, 1)
# x_train의 0번째 데이터를 augment_size 만큼 증폭해서 (-1, 28, 28, 1) 모양으로 리쉐이프 해라
# np.tile(데이터, 증폭시킬 개수)

print(np.zeros(augment_size))   # zeros = 0을 출력해줌 -> 100개에 0을 출력해줌
print(np.zeros(augment_size).shape)     # (100,)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),    # x 데이터
    np.zeros(augment_size),  # y 데이터: 그림만 그릴 거임 -> 필요없으니까 0만 넣어줌
    batch_size = augment_size,  # 100개 데이터를 100개의 배치사이즈로 넣으니까 x데이터의 첫 번째 행에 모든 데이터가 들어감
    shuffle= True
)
print(x_data)
# <keras.preprocessing.image.NumpyArrayIterator object at 0x000001D88A8A5DF0>
print(x_data[0])        # x와 y가 모두 포함
print(x_data[0][0].shape)   # (100, 28, 28, 1)
print(x_data[0][1].shape)   # (100,)

import matplotlib.pyplot as plt
plt.figure(figsize = (7, 7))
for i in range(49):
    plt.subplot(7, 7, i+1)  # 7 바이 7의 서브플롯을 만든다.
    plt.axis('off')
    plt.imshow(x_data[0][0][i], cmap = 'gray')
