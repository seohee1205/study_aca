import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터
# 이미지 전처리
train_datagen = ImageDataGenerator(
    rescale = 1./255,           # MimMax 스케일링(정규화) 하겠다는 의미, .을 붙인 이유는 부동소수점으로 연산하라 라는 뜻
    # horizontal_flip= True,      # 상하반전 (수평방향 뒤집기)
    # vertical_flip= True,        # 좌우반전 (수직방향 뒤집기)
    # width_shift_range = 0.1,    # 10%만큼 좌우로 이동 가능
    # height_shift_range= 0.1,    # 데이터 증폭: 상하로 10% 이동
    # rotation_range= 5,          # 지정된 각도 범위 내에서 돌릴 수 있는 범위
    # zoom_range= 1.2,            # 20%까지 확대
    # shear_range= 0.7,           # 찌그러트릴 수 있는 범위
    # fill_mode = 'nearest',      # 이미지를 움직일 때, 움직여서 없어진 범위에 근접값을 입력해주는 기능
                                # 숫자 6과 9 같은 반전하면 데이터가 꼬이는 경우도 있음, 이럴 경우 옵션 조절해야함
)

test_datagen = ImageDataGenerator(
    rescale = 1./255,
# 테스트 데이터는 평가하는 데이터이기 때문에 데이터를 증폭한다는 건 결과를 조작하는 것, 때문에 스케일을 제외한 옵션은 삭제
# 통상적으로 테스트 데이터는 증폭(전처리)하지 않음
)



xy_train = train_datagen.flow_from_directory(   # 이미지제너레이터는 폴더별로 라벨값 부여
    
    'd:/study_data/_data/brain/train/',     # 분류된 폴더의 상위폴더까지 지정  # directory= 폴더
    target_size=(100, 100),           # 수집한 데이터마다 이미지 사진크기 다르므로 이미지크기 동일하게 고정
    batch_size = 5,  # 전체 데이터 쓰려면 160(전체 데이터 개수) 이상 넣기 / # 5장씩 잘라라
    class_mode = 'binary',           # 0,1로 구별(nomal,ad) / 0,1,2(가위,바위,보)// # 원핫사용한 경우 => 'categorical'
    color_mode = 'grayscale',
    # color_mode = 'rgb',
    shuffle= True,
)       # Found 160 images belonging to 2 classes.      0과 1의 클래스로 분로되었다.    # x = 160, 200. 200. 1로 변환됨, y= 160,

xy_test = test_datagen.flow_from_directory(
    
    'd:/study_data/_data/brain/test/',
    target_size=(100, 100),       
    batch_size = 5,                
    class_mode = 'binary',            
    color_mode = 'grayscale',
    shuffle= True,
)       # Found 120 images belonging to 2 classes.      0과 1의 클래스로 분로되었다.    # 160, 200, 200, 1로 변환됨, y = 120,

print(xy_train)     # <keras.preprocessing.image.DirectoryIterator object at 0x000001BDB18D8490>

print(xy_train[0])
# print(xy_train.shape)     # error
print(len(xy_train))        # 32
print(len(xy_train[0]))     # 2
print(xy_train[0][0])       # x 5개 들어가있다. 
print(xy_train[0][1])       # [1. 0. 0. 1. 0.]

print(xy_train[0][0].shape) # (5, 200, 200, 1)
print(xy_train[0][1].shape) # (5,)


print("=======================================================")
print(type(xy_train))   # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

# 현재 (5, 200, 200, 1) 짜리 데이터가 32덩어리

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape = (100, 100, 1), activation= 'relu'))
model.add(Conv2D(64, (3, 3), activation= 'relu'))
model.add(Flatten())
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

# model.fit(xy_train[:][0], xy_train[:][1], epochs = 10)  # 에러
# model.fit(xy_train[0][0], xy_train[0][1], epochs = 10)  # 통배치 넣으면 가능
hist = model.fit_generator(xy_train, epochs = 30,   # x데이터, y데이터, batch
                    steps_per_epoch = 32,   # 전체데이터크기/batch 사이즈 = 160/5 = 32
                    validation_data= xy_test,
                    validation_steps= 24,   # 발리데이터/batch = 120/5 = 24
)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

# print(acc)
print('loss : ', loss[-1])  
print('val_loss : ', val_loss[-1])  
print('acc : ', acc[-1])  
print('val_acc : ', val_acc[-1])  


#1. 그림그리기 
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(range(len(hist.history['loss'])),hist.history['loss'],label='loss')
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label='val_loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(hist.history['acc'])),hist.history['acc'],label='acc')
plt.plot(range(len(hist.history['val_acc'])),hist.history['val_acc'],label='val_acc')
plt.legend()
plt.show()



#2. 튜닝 0.95 이상

# val_acc: 0.9667
# loss :  0.20503728091716766
# val_loss :  0.255054771900177
# acc :  1.0
# val_acc :  0.9666666388511658


