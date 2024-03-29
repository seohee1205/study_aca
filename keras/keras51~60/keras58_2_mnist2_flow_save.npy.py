# 수치형으로 제공된 데이터를 증폭

from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
np.random.seed(0)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

augment_size = 40000

# randidx = np.random.randint(60000, size = 40000)
randidx = np.random.randint(x_train.shape[0], size = augment_size)
print(randidx)          # [28678 44651 21849 ... 29255 51144 18206]
print(randidx.shape)    # (40000,)
print(np.min(randidx), np.max(randidx)) # 3 59999

x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy()      
print(x_augmented)
print(x_augmented.shape, y_augmented.shape)  # (40000, 28, 28) (40000,)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1], 
                        x_test.shape[2], 1)
x_augmented = x_augmented.reshape(x_augmented.shape[0], 
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)

# (2) = (1)     # 증폭하는 부분
x_augmented = train_datagen.flow(
    x_augmented, y_augmented, batch_size= augment_size, shuffle= False
).next()[0]

print(x_augmented) 
print(x_augmented.shape)  # (40000, 28, 28, 1)

print(np.max(x_train), np.min(x_train))     # 255.0 0.0
print(np.max(x_augmented), np.min(x_augmented)) # 1.0 0.0

x_train = np.concatenate((x_train/255., x_augmented), axis = 0)
y_train = np.concatenate((y_train, y_augmented), axis = 0)
x_test = x_test/255.

print(x_train.shape, y_train.shape)    # (100000, 28, 28, 1) (100000,)

# 원핫
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 

save_path = 'd:/study_data/_save/mnist/'

np.save(save_path + 'keras58_2_mnist_x_train.npy', arr = x_train)
np.save(save_path + 'keras58_2_mnist_x_test.npy', arr = x_test)
np.save(save_path + 'keras58_2_mnist_y_train.npy', arr = y_train)
np.save(save_path + 'keras58_2_mnist_y_test.npy', arr = y_test)

