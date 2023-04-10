import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping


# sigmoid는 0부터 1까지인데 감정을 0부터 3까지 정의했으니까 사용자 정의 함수를 사용해야함
def custom_activation(x):
    return K.clip(x, 0, 3)


custom_activation.name = 'custom_activation'

# 데이터 함수 정의
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# 폴더별로 라벨값 부여
batch_size = 30

xy_train = train_datagen.flow_from_directory(
 'd:/study_data/_data/project/train/',
target_size=(100, 100),
batch_size= batch_size,
class_mode='sparse',        # 1차원 정수 넘파이 배열
color_mode='grayscale',
shuffle=True
)
# Found 20153 images belonging to 4 classes.

xy_test = test_datagen.flow_from_directory(
 'd:/study_data/_data/project/test/',
target_size=(100, 100),
batch_size=batch_size,
class_mode='sparse',
color_mode='grayscale',
shuffle=True
)
# Found 5085 images belonging to 4 classes.

# 모델 구성
model = Sequential()
model.add(Conv2D(20, (2, 2), input_shape=(100, 100, 1), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation=lambda x: custom_activation(x)))
model.summary()

# 컴파일 및 훈련
model.compile(loss= 'mse', optimizer= 'adam', metrics= ['acc'])

es = EarlyStopping(monitor = 'acc', patience = 60, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

model.fit_generator(xy_train, epochs= 100,
                    steps_per_epoch= 10,
                    # validation_data = xy_test,
                    # validation_steps = 5085/batch_size,
                    callbacks = [es])


# def food(y):
#     for i in range(len(y)):
#         if 0<=y[i]<3/8:
#             print('emotion : angry, food recommendation : a')
#         elif 3/8<=y[i]<3/4:
#             print('emotion : angry, food recommendation : b')
#         elif 3/4<=y[i]<9/8:
#             print('emotion : sad, food recommendation : c')
#         elif 9/8<=y[i]<12/8:
#             print('emotion : sad, food recommendation : d')
#         elif 12/8<=y[i]<15/8:
#             print('emotion : default, food recommendation : e')
#         elif 15/8<=y[i]<18/8:
#             print('emotion : default, food recommendation : f')
#         elif 18/8<=y[i]<21/8:
#             print('emotion : happy, food recommendation : g')
#         elif 21/8<=y[i]<=3:
#             print('emotion : happy, food recommendation : h')


# import random
# random_index_1 = np.random.randint(0, 19)
# random_index_2 = np.random.randint(0, 19)
# x_pred = xy_test[random_index_1][0][random_index_2].reshape(-1, 100, 100, 1)
# food(model.predict(x_pred))

# 평가, 예측
acc = model.evaluate(xy_test)[1]
print('acc : ', acc)