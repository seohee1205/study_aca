import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import keras.backend as K
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping



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
class_mode='categorical',        # 1차원 정수 넘파이 배열
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
    class_mode= 'categorical',           # 왜냐? 훈련시킨 데이터는 y라벨값이 있음, pred이미지는 감정이 뭔지 모르니까
    color_mode= 'grayscale',
    shuffle= False
)
x_pred = x_predict[0][0].reshape(1, 48, 48, 1)
# print(x_pred)
# print(x_pred.shape)     # (1, 48, 48, 1)

#######################################################
model = load_model("d:/study_data/y_predict/project_6.h5")

# 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer= 'adam')

# 음식값 정의
import random
import numpy as np
# angry
def angry():
    dice = random.randint(1,100000)/100000
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
    dice = random.randint(1,100000)/100000
    prob =[0.3176, 0.2207, 0.1093, 0.0856, 0.0769, 0.0722, 0.0499, 0.0283, 0.0216, 0.0216, 0.0169]
    food =["막걸리", "초콜릿", "비빔국수", "녹차", "된장찌개", "울면", "죽", "돼지갈비", "과자", "푸딩", "치킨"]
    prob_sum = 0
    for i in range(len(prob)):
        prob_sum += prob[i]
        if dice < prob_sum:
            return food[i]

# default
def default():
    dice = random.randint(1,100000)/100000
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
    dice = random.randint(1,100000)/100000  # 0부터 1 사이에 랜덤으로 위치 정해줌
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

# model.save("d:/study_data/y_predict/project_6.h5")

loss = model.evaluate(xy_test)
print('loss : ', loss)

print(f'감정: {emotion} \n추천 메뉴 : {food}')

# print(f'감정: {emotion}')
# print(f'추천 메뉴 : {food}')