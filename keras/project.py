import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score



savepath= 'd:/study_data/_save/project/',
# mcpname = '{epoch:04d}-{val_loss:.2f}.hdf5'

# 데이터 함수 정의
datagen= ImageDataGenerator(rescale=1./255)


# 폴더별로 라벨값 부여
batch_size= 20

xy_train= datagen.flow_from_directory(
 'd:/study_data/_data/project/train/',
target_size=(48, 48),
batch_size= batch_size,
class_mode='sparse',        # 1차원 정수 넘파이 배열
color_mode='grayscale',
shuffle=True
)
# Found 20153 images belonging to 4 classes.

xy_test= datagen.flow_from_directory(
 'd:/study_data/_data/project/test/',
target_size=(48, 48),
batch_size= batch_size,
class_mode='sparse',
color_mode='grayscale',
shuffle= True
)
# Found 5085 images belonging to 4 classes.

x_predict= datagen.flow_from_directory(
    'd:/study_data/y_predict/',
    target_size= (48, 48),
    batch_size= batch_size,
    class_mode= None,           # 왜냐? 훈련시킨 데이터는 y라벨값이 있음, pred이미지는 감정이 뭔지 모르니까
    color_mode= 'grayscale',
    shuffle= False
)
x_pred = x_predict[0][0].reshape(1, 48, 48, 1)
print(x_pred)
print(x_pred.shape)

# sigmoid는 0부터 1까지인데 감정을 0부터 3까지 정의했으니까 사용자 정의 함수를 사용해야함
def custom_activation(x):
    return K.clip(x, 0, 3,)    # x를 0부터 3까지 사이의 값을 반환하겠다

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
# model.summary()

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 48, 48, 128)       640
#  batch_normalization (BatchN  (None, 48, 48, 128)      512
#  ormalization)
#  conv2d_1 (Conv2D)           (None, 48, 48, 80)        41040
#  batch_normalization_1 (Batc  (None, 48, 48, 80)       320
#  hNormalization)
#  max_pooling2d (MaxPooling2D  (None, 24, 24, 80)       0
#  )
#  dropout (Dropout)           (None, 24, 24, 80)        0
#  conv2d_2 (Conv2D)           (None, 24, 24, 60)        19260
#  max_pooling2d_1 (MaxPooling  (None, 12, 12, 60)       0
#  2D)
#  conv2d_3 (Conv2D)           (None, 12, 12, 100)       24100
#  max_pooling2d_2 (MaxPooling  (None, 6, 6, 100)        0
#  2D)
#  conv2d_4 (Conv2D)           (None, 6, 6, 120)         48120
#  max_pooling2d_3 (MaxPooling  (None, 3, 3, 120)        0
#  2D)
#  dropout_1 (Dropout)         (None, 3, 3, 120)         0
#  conv2d_5 (Conv2D)           (None, 3, 3, 90)          97290
#  max_pooling2d_4 (MaxPooling  (None, 1, 1, 90)         0
#  2D)

#  flatten (Flatten)           (None, 90)                0
#  dense (Dense)               (None, 64)                5824
#  dense_1 (Dense)             (None, 32)                2080
#  dropout_2 (Dropout)         (None, 32)                0
#  dense_2 (Dense)             (None, 64)                2112
#  dense_3 (Dense)             (None, 50)                3250      
#  dropout_3 (Dropout)         (None, 50)                0
#  dense_4 (Dense)             (None, 1)                 51
# =================================================================
# Total params: 244,599
# Trainable params: 244,183
# Non-trainable params: 416
# _________________________________________________________________


# 컴파일, 훈련
model.compile(loss= 'mse', optimizer= 'adam')

es = EarlyStopping(monitor = 'val_loss', patience = 70, mode = 'min',
                   verbose = 1, restore_best_weights= True)

model.fit(xy_train, epochs= 10,
                    validation_data= xy_test,
                    shuffle = True,
                    # steps_per_epoch= 10,
                    # validation_data = xy_test,
                    # validation_steps = 5085/batch_size,
                    callbacks = [es]
                    )


# 음식값 정의

# food_frequency_angry= {
#     "Spicy/Hot food: 26% ",
#     "Chicken: 5.3% ",
#     "Snacks: 3.2% ",
#     "Meat dishes: 2.75% ",
#     "Noodles: 2.1% ",
#     "BiBimbab: 2.1% "
# }

# food_frequency_sad= {
#     "Spicy/Hot food: 8.1% ",
#     "Stew $ Soup: 5.7% ",
#     "Noodles: 5.35% ",
#     "Rice porridge: 3.7% ",
#     "Meat dishes: 2.1% ",
#     "Snacks: 1.6% "
# }

# food_frequency_default= {
#     "샐러드"
#     "라면"
#     "돈까스"
#     "육회비빔밥"
#     "제육덮밥"
#     "순두부찌개"
# }
    
# food_frequency_happy = {
#     "Meat dishes: 14.75% ",
#     "Noodles:14% ",
#     "Pizza&Spagetti: 8.85% ",
#     "cake: 5.1% ",
#     "Stew $ Soup: 4.7% ",
#     "Rawfish&sushi: 4.05% "
# }


def food(y):
    for i in range(len(y)):
        if y[i]<1/8:
            print('emotion : angry, food recommendation : 낚지볶음')
        elif 1/8<=y[i]<2/8:   
            print('emotion : angry, food recommendation : 치킨')
        elif 2/8<=y[i]<3/8:
            print('emotion : angry, food recommendation : 과자')
        elif 3/8<=y[i]<4/8:
            print('emotion : angry, food recommendation : 소갈비') 
        elif 4/8<=y[i]<5/8:
            print('emotion : angry, food recommendation : 칼국수') 
        elif 5/8<=y[i]<6/8:
            print('emotion : angry, food recommendation : 비빔밥')    

        elif 3/4<=y[i]<7/8:
            print('emotion : sad, food recommendation : 비빔국수')
        elif 7/8<=y[i]<8/8:
            print('emotion : sad, food recommendation : 된장찌개')    
        elif 8/8<=y[i]<9/8:
            print('emotion : sad, food recommendation : 울면')    
        elif 9/8<=y[i]<10/8:
            print('emotion : sad, food recommendation : 죽')    
        elif 10/8<=y[i]<11/8:
            print('emotion : sad, food recommendation : 돼지갈비')    
        elif 11/8<=y[i]<12/8:
            print('emotion : sad, food recommendation : 과자')    
            
        elif 12/8<=y[i]<13/8:
            print('emotion : default, food recommendation : 샐러드')
        elif 13/8<=y[i]<14/8:
            print('emotion : default, food recommendation : 라면')    
        elif 14/8<=y[i]<15/8:
            print('emotion : default, food recommendation : 돈까스')    
        elif 15/8<=y[i]<16/8:
            print('emotion : default, food recommendation : 육회비빔밥')    
        elif 16/8<=y[i]<17/8:
            print('emotion : default, food recommendation : 제육덮밥')    
        elif 17/8<=y[i]<18/8:
            print('emotion : default, food recommendation : 순두부찌개')    
            
        elif 18/8<=y[i]<19/8:
            print('emotion : happy, food recommendation : 소곱창')
        elif 19/8<=y[i]<20/8:
            print('emotion : happy, food recommendation : 냉면')    
        elif 20/8<=y[i]<21/8:
            print('emotion : happy, food recommendation : 피자&스파게티')    
        elif 21/8<=y[i]<22/8:
            print('emotion : happy, food recommendation : 케이크')    
        elif 22/8<=y[i]<23/8:
            print('emotion : happy, food recommendation : 김치찌개')    
        elif 23/8<=y[i]:
            print('emotion : happy, food recommendation : 회&초밥')


# import random
# random_index_1 = np.random.randint(0, 19)
# random_index_2 = np.random.randint(0, 19)
# x_pred = xy_test[random_index_1][0][random_index_2].reshape(-1, 48, 48, 1)
# food(model.predict(x_pred))


model.save('d:/study_data/_save/project/loss_2.h5')

# 평가, 예측

path = 'd:/study_data/y_predict/'
# y_predict = model.predict(xy_test)

loss = model.evaluate(xy_test)
print('loss : ', loss)


name = "광수"  # 원하는 이름 추가
name_food = model.predict(x_pred)
print(food(name_food))
print(name + '에게 추천하는 음식은? : ', name_food)


