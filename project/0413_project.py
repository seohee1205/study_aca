import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# model = load_model('./_save/samsung/keras53_samsung2_ysh.h5')

savepath= 'd:/study_data/_save/project/',
# mcpname = '{epoch:04d}-{val_loss:.2f}.hdf5'

# 데이터 함수 정의
datagen= ImageDataGenerator(rescale=1./255)


# 폴더별로 라벨값 부여
batch_size= 10

xy_train= datagen.flow_from_directory(
 'd:/study_data/_data/project/train/',
target_size=(48, 48),
batch_size= batch_size,
class_mode='categorical',    
color_mode='grayscale',
shuffle=True
)
# Found 20153 images belonging to 4 classes.

xy_test= datagen.flow_from_directory(
 'd:/study_data/_data/project/test/',
target_size=(48, 48),
batch_size= batch_size,
class_mode='categorical',
color_mode='grayscale',
shuffle= True
)
# Found 5085 images belonging to 4 classes.

x_predict= datagen.flow_from_directory(
    'd:/study_data/y_predict/',
    target_size= (48, 48),
    batch_size= batch_size,
    class_mode= 'categorical',          # class_mode를 categorical로 하면서 원핫인코딩도 같이 해줌
    color_mode= 'grayscale',
    shuffle= False
)
x_pred = x_predict[0][0].reshape(1, 48, 48, 1)
# print(x_pred)
# print(x_pred.shape)     # (1, 48, 48, 1)


# 모델 구성
model = Sequential()
model.add(Conv2D(128, (2, 2), padding='same', input_shape=(48, 48, 1), activation= 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(80, (2, 2), padding='same', activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(60, (2, 2), padding='same', activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(100, (2, 2), padding='same', activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(120, (2, 2), padding='same', activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(90, (3, 3), padding='same', activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(50, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation= 'softmax'))
model.summary()

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
#  batch_normalization_2 (Batc  (None, 12, 12, 60)       240
#  hNormalization)
#  conv2d_3 (Conv2D)           (None, 12, 12, 100)       24100
#  batch_normalization_3 (Batc  (None, 12, 12, 100)      400
#  hNormalization)
#  max_pooling2d_2 (MaxPooling  (None, 6, 6, 100)        0
#  2D)
#  conv2d_4 (Conv2D)           (None, 6, 6, 120)         48120
#  batch_normalization_4 (Batc  (None, 6, 6, 120)        480
#  hNormalization)
#  max_pooling2d_3 (MaxPooling  (None, 3, 3, 120)        0
#  2D)
#  dropout_1 (Dropout)         (None, 3, 3, 120)         0
#  conv2d_5 (Conv2D)           (None, 3, 3, 90)          97290
#  batch_normalization_5 (Batc  (None, 3, 3, 90)         360
#  hNormalization)
#  max_pooling2d_4 (MaxPooling  (None, 1, 1, 90)         0
#  2D)
#  flatten (Flatten)           (None, 90)                0
#  dense (Dense)               (None, 64)                5824
#  dense_1 (Dense)             (None, 32)                2080
#  dropout_2 (Dropout)         (None, 32)                0
#  dense_2 (Dense)             (None, 64)                2112
#  dense_3 (Dense)             (None, 50)                3250
#  dropout_3 (Dropout)         (None, 50)                0
#  dense_4 (Dense)             (None, 4)                 204
# =================================================================
# Total params: 246,232
# Trainable params: 245,076
# Non-trainable params: 1,156
# _________________________________________________________________

# 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer= 'adam')

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min',
                   verbose = 1, restore_best_weights= True)

model.fit(xy_train, epochs= 200,
                    validation_data= xy_test,
                    shuffle = True,
                    callbacks = [es])

# print(model.predict(xy_train).shape)    # (20153, 4)

# 음식값 정의
import random
import numpy as np
# angry
def angry():
    dice = random.randint(1,10000)/10000
    prob =[0.334,  0.209,  0.133, 0.101, 0.068, 0.041,  0.035, 0.027, 0.027, 0.024, 0.022, 0.019]
    food =["낚지볶음", "소주", "초콜릿", "주스", "치킨", "과자", "소갈비", "칼국수", "비빔밥", "아이스크림", "케이크", "짬뽕" ]
    prob_sum = 0
    for i in range(len(prob)):
        prob_sum += prob[i]
        if dice < prob_sum:
            return food[i]
    return angry()

# sad
def sad():
    dice = random.randint(1,10000)/10000
    prob =[0.3176, 0.2207, 0.1093, 0.0856, 0.0769, 0.0722, 0.0499, 0.0283, 0.0216, 0.0216, 0.0169]
    food =["막걸리", "초콜릿", "비빔국수", "녹차", "된장찌개", "울면", "죽", "돼지갈비", "과자", "푸딩", "치킨"]
    prob_sum = 0
    for i in range(len(prob)):
        prob_sum += prob[i]
        if dice < prob_sum:
            return food[i]

# default
def default():
    dice = random.randint(1,10000)/10000
    prob =[0.189, 0.1625, 0.1395, 0.1277, 0.1214, 0.067, 0.0642, 0.0518, 0.0366, 0.0332, 0.0255, 0.0228]
    food =["치킨", "맥주", "삼겹살", "피자&스파게티", "아이스크림", "커피", "초콜릿", "케이크", "회&초밥", "와플", "과자", "순두부찌개"]
    prob_sum = 0
    for i in range(len(prob)):
        prob_sum += prob[i]
        if dice < prob_sum:
            return food[i]
    return default()

# happy
def happy():
    dice = random.randint(1,10000)/10000  # 0부터 1 사이에 랜덤으로 위치 정해줌
    prob =[0.2045, 0.1943, 0.1256, 0.1227, 0.0706, 0.0706, 0.0653, 0.0562, 0.0521, 0.0305, 0.0243]
    food =["소곱창", "치킨", "와인", "피자&스파게티", "케이크", "초콜릿", "김치찌개", "회&초밥", "스테이크", "코코아", "샐러드"]
    prob = np.array(prob)/100
    prob_sum = 0
    for i in range(len(prob)):
        prob_sum += prob[i]
        if dice < prob_sum:
            return food[i]
    return happy()

emotion= model.predict(x_pred)
emotion_list=('angry','sad','default','happy')
emotion=emotion_list[np.argmax(emotion)]
food=str()
if emotion=='angry':
    food=angry()
elif emotion=='sad':
    food=sad()
elif emotion=='default':
    food=default()
elif emotion=='happy':
    food=happy()


# 평가, 예측

# model.save("d:/study_data/y_predict/project_7.h5")

loss = model.evaluate(xy_test)
print('loss : ', loss)

print(f'감정: {emotion} \n추천 메뉴 : {food}')

