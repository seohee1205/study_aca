import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping



savepath = 'd:/study_data/_save/project/',
# mcpname = '{epoch:04d}-{val_loss:.2f}.hdf5'

# 데이터 함수 정의
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


# 폴더별로 라벨값 부여
batch_size = 16

xy_train = train_datagen.flow_from_directory(
 'd:/study_data/_data/project/train/',
target_size=(48, 48),
batch_size= batch_size,
class_mode='sparse',        # 1차원 정수 넘파이 배열
color_mode='grayscale',
shuffle=True
)
# Found 20153 images belonging to 4 classes.

xy_test = test_datagen.flow_from_directory(
 'd:/study_data/_data/project/test/',
target_size=(48, 48),
batch_size= batch_size,
class_mode='sparse',
color_mode='grayscale',
shuffle= False
)
# Found 5085 images belonging to 4 classes.


# sigmoid는 0부터 1까지인데 감정을 0부터 3까지 정의했으니까 사용자 정의 함수를 사용해야함
def custom_activation(x):
    return K.clip(x, 0, 3,)      # x를 0부터 3까지 사이의 값을 반환하겠다


custom_activation.name = 'custom_activation'


# 모델 구성
model = Sequential()
model.add(Conv2D(128, (2, 2), padding='same', input_shape=(48, 48, 1), activation= 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(80, (2, 2), padding='same', activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(60, (2, 2), padding='same', activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(100, (2, 2), padding='same', activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(120, (2, 2), padding='same', activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(90, (3, 3), padding='same', activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation=lambda x: custom_activation(x)))
model.summary()



# 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')

es = EarlyStopping(monitor = 'val_loss', patience = 70, mode = 'min',
                   verbose = 1, restore_best_weights= True)

model.fit(xy_train, epochs= 200,
                    validation_data= xy_test,
                    shuffle = True,
                    # steps_per_epoch= 10,
                    # validation_data = xy_test,
                    # validation_steps = 5085/batch_size,
                    callbacks = [es]
                    )


def food(y):
    for i in range(len(y)):
        if y[i]<1/8:
            print('emotion : angry, food recommendation : 신길동 매운짬뽕')
        elif 1/8<=y[i]<2/8:   
            print('emotion : angry, food recommendation : 디진다 돈까스')
        elif 2/8<=y[i]<3/8:
            print('emotion : angry, food recommendation : 송주불냉면')
        elif 3/8<=y[i]<4/8:
            print('emotion : angry, food recommendation : 치킨플러스 - 핵매운치킨') 
        elif 4/8<=y[i]<5/8:
            print('emotion : angry, food recommendation : 염라대왕라면') 
        elif 5/8<=y[i]<6/8:
            print('emotion : angry, food recommendation : 엽기떡볶이')    
            
            
        elif 3/4<=y[i]<7/8:
            print('emotion : sad, food recommendation : 닭발')
        elif 7/8<=y[i]<8/8:
            print('emotion : sad, food recommendation : 낙지김치죽')    
        elif 8/8<=y[i]<9/8:
            print('emotion : sad, food recommendation : 짬뽕')    
        elif 9/8<=y[i]<10/8:
            print('emotion : sad, food recommendation : 매운돼지갈비찜')    
        elif 11/8<=y[i]<12/8:
            print('emotion : sad, food recommendation : 제육덮밥')    
        elif 12/8<=y[i]<13/8:
            print('emotion : sad, food recommendation : 안성탕면')    
            
            
            
        elif 13/8<=y[i]<14/8:
            print('emotion : default, food recommendation : 비빔냉면')
        elif 14/8<=y[i]<15/8:
            print('emotion : default, food recommendation : 카레')    
        elif 16/8<=y[i]<17/8:
            print('emotion : default, food recommendation : 국밥')    
        elif 18/8<=y[i]<19/8:
            print('emotion : default, food recommendation : 피자')    
        elif 20/8<=y[i]<21/8:
            print('emotion : default, food recommendation : 짜장면')    
        elif 22/8<=y[i]<23/8:
            print('emotion : default, food recommendation : 간장치킨')    
            
            
 
        elif 23/8<=y[i]<24/8:
            print('emotion : happy, food recommendation : 소고기')
        elif 24/8<=y[i]<25/8:
            print('emotion : happy, food recommendation : 육회비빔밥')    
        elif 25/8<=y[i]<26/8:
            print('emotion : happy, food recommendation : 돈까스')    
        elif 27/8<=y[i]<28/8:
            print('emotion : happy, food recommendation : 쌀국수')    
        elif 28/8<=y[i]<29/8:
            print('emotion : happy, food recommendation : 김밥')    
        elif 29/8<=y[i]:
            print('emotion : happy, food recommendation : 죽')


import random
random_index_1 = np.random.randint(0, 19)
random_index_2 = np.random.randint(0, 19)
x_pred = xy_test[random_index_1][0][random_index_2].reshape(-1, 48, 48, 1)
food(model.predict(x_pred))


# 평가, 예측
loss = model.evaluate(xy_test)
print('loss : ', loss)


model.save('d:/study_data/_save/project/loss_1.h5')

