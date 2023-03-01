# 수치형으로 제공된 데이터를 증폭

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
np.random.seed(0)
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

# (1)
# x_augmented = train_datagen.flow(
#     x_augmented, y_augmented, batch_size= augment_size, shuffle= False
# )
# print(x_augmented)  # <keras.preprocessing.image.NumpyArrayIterator object at 0x0000013F3E81CB20>
# print(x_augmented[0][0].shape)  # (40000, 28, 28, 1)

# (2) = (1)
x_augmented = train_datagen.flow(
    x_augmented, y_augmented, batch_size= augment_size, shuffle= False
).next()[0]

print(x_augmented) 
print(x_augmented.shape)  # (40000, 28, 28, 1)

# x_train = x_train + x_augmented
# print(x_train.shape)
# 오류남

print(np.max(x_train), np.min(x_train))     # 255.0 0.0
print(np.max(x_augmented), np.min(x_augmented)) # 1.0 0.0

x_train = np.concatenate((x_train/255., x_augmented), axis = 0)
y_train = np.concatenate((y_train, y_augmented), axis = 0)
x_test = x_test/255.

print(x_train.shape, y_train.shape)    # (100000, 28, 28, 1) (100000,)


# [실습]
# x_augmented 10개와 x_train 10개를 비교하는 이미지 출력할 것   (2, 10)

# 원핫
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 

#2. 모델구성
model = Sequential()
model.add(Conv2D(256, (2,2), input_shape= (28, 28, 1), activation= 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), activation= LeakyReLU(0.8)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), activation= LeakyReLU(0.8)))
model.add(Flatten())
model.add(Dense(32, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['acc'])

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

hist = model.fit(x_train, y_train, epochs = 100,   # x데이터, y데이터, batch
                    validation_split = 0.2,
                    batch_size = 130,
                    callbacks = [es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_acc, y_predict)

print('acc : ', acc)


import matplotlib.pyplot as plt
plt.figure(figsize=(7, 7))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[i+60000], cmap='gray')
    plt.subplot(2, 10, i+11)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')
plt.show()