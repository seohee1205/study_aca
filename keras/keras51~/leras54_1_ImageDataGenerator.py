import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,           # MimMax 스케일링(정규화) 하겠다는 의미, .을 붙인 이유는 부동소수점으로 연산하라 라는 뜻
    horizontal_flip= True,      # 가로 뒤집기 (수평방향 뒤집기)
    vertical_flip= True,        # 세로 뒤집기 (수직방향 뒤집기)
    width_shift_range = 0.1,    # 10%만큼
    height_shift_range= 0.1,    # 상하로 10% 움직일 수 있다는 뜻
    rotation_range= 5,          # 돌릴 수 있는 범위
    zoom_range= 1.2,            # 20%까지 확대
    shear_range= 0.7,           # 찌그러트릴 수 있는 범위
    fill_mode = 'nearest',      # 이미지를 움직일 때, 움직여서 없어진 범위에 근접값을 입력해주는 기능
                                # 숫자 6과 9 같은 반전하면 데이터가 꼬이는 경우도 있음, 이럴 경우 옵션 조절해야함
)

test_datagen = ImageDataGenerator(
    rescale = 1./255,
                # 테스트 데이터는 평가하는 데이터이기 때문에 데이터를 증폭한다는 건 결과를 조작하는 것, 때문에 스케일을 제외한 옵션은 삭제
                # 통상적으로 테스트 데이터는 증폭하지 않음
)

xy_train = train_datagen.flow_from_directory(
    
    'd:/study_data/_data/brain/train/',
    target_size=(200, 200),          # 이미지 데이터를 200 * 200으로 확대 혹은 축소해라. 사이즈를 동일하게 만들어줌
    batch_size = 5,                  # 5장씩 잘라라
    class_mode = 'binary',           # 0과 1을 찾는 mode, int형 수치화해서 만들어줌
    color_mode = 'grayscale',
    shuffle= True,
)       # Found 160 images belonging to 2 classes.      0과 1의 클래스로 분로되었다.    # x = 160, 200. 200. 1로 변환됨, y= 160,

xy_test = test_datagen.flow_from_directory(
    
    'd:/study_data/_data/brain/test/',
    target_size=(200, 200),       
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

