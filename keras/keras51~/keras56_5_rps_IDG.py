# 150 * 150 * 3

# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

# 넘파이까지 저장
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time

path = 'd:/study_data/_data/rps/'


# np.save(path + '파일명', arr = ???)
#1. time.time()으로 이미지 수치화하는 시간 체크할 것
#2. time.time()으로 넘파이로 변경하는 시간 체크할 것


#1. 데이터
start = time.time()
# 이미지 전처리
train_datagen = ImageDataGenerator(
    rescale = 1./255,           # MimMax 스케일링(정규화) 하겠다는 의미, .을 붙인 이유는 부동소수점으로 연산하라 라는 뜻
#     horizontal_flip= True,      # 상하반전 (수평방향 뒤집기)
#     vertical_flip= True,        # 좌우반전 (수직방향 뒤집기)
#     width_shift_range = 0.1,    # 10%만큼 좌우로 이동 가능
#     height_shift_range= 0.1,    # 데이터 증폭: 상하로 10% 이동
#     rotation_range= 5,          # 지정된 각도 범위 내에서 돌릴 수 있는 범위
#     zoom_range= 1.2,            # 20%까지 확대
#     shear_range= 0.7,           # 찌그러트릴 수 있는 범위
#     fill_mode = 'nearest',      # 이미지를 움직일 때, 움직여서 없어진 범위에 근접값을 입력해주는 기능
#                                 # 숫자 6과 9 같은 반전하면 데이터가 꼬이는 경우도 있음, 이럴 경우 옵션 조절해야함
 )

test_datagen = ImageDataGenerator(
    rescale = 1./255,
# 테스트 데이터는 평가하는 데이터이기 때문에 데이터를 증폭한다는 건 결과를 조작하는 것, 때문에 스케일을 제외한 옵션은 삭제
# 통상적으로 테스트 데이터는 증폭(전처리)하지 않음
)

xy_train = train_datagen.flow_from_directory(   # 이미지제너레이터는 폴더별로 라벨값 부여
    
    'd:/study_data/_data/rps/',     # 분류된 폴더의 상위폴더까지 지정  # directory= 폴더
    target_size=(150, 150),           # 수집한 데이터마다 이미지 사진크기 다르므로 이미지크기 동일하게 고정
    batch_size = 15000,  # 전체 데이터 쓰려면 160(전체 데이터 개수) 이상 넣기 / # 5장씩 잘라라
    class_mode = 'categorical',           # 0,1로 구별(nomal,ad) / 0,1,2(가위,바위,보)// # 원핫사용한 경우 => 'categorical'
    # color_mode = 'grayscale',
    color_mode = 'rgb',
    shuffle= True,
)       # Found 25000 images belonging to 2 classes.      0과 1의 클래스로 분로되었다.    # x = 160, 200. 200. 1로 변환됨, y= 160,

# xy_test = test_datagen.flow_from_directory(
#     'd:/study_data/_data/cat_dog/Petimages/',
#     target_size=(30, 30),       
#     batch_size = 120,                
#     class_mode = 'binary',          # y의 클래스 , binary: 수치로 빼라 / categorical: 원핫인코딩 됨
#     # color_mode = 'grayscale',
#     color_mode = 'rgb',
#     shuffle= True,
# )       # Found 25000 images belonging to 2 classes.       0과 1의 클래스로 분로되었다.    # 160, 200, 200, 1로 변환됨, y = 120,

# print(xy_train)     # <keras.preprocessing.image.DirectoryIterator object at 0x000001BDB18D8490>

print(xy_train[0])
# print(xy_train.shape)     # error
print(len(xy_train))        # 1
print(len(xy_train[0]))     # 2
print(xy_train[0][0])       # x 5개 들어가있다. 
print(xy_train[0][1])       # [1. 0. 0. 1. 0.]

print(xy_train[0][0].shape)     # (2520, 150, 150, 3)
# print(xy_test[0][0].shape)      # (120, 30, 30, 1)
print(xy_train[0][1].shape)     # (2520, 3)
# print(xy_test[0][1].shape)      # (120,)

end1 = time.time()
print('이미지 수치화 소요 시간 : ', np.round(end1-start, 2))

save_path = 'd:/study_data/_save/rps/'
np.save(save_path + 'keras56_x_train.npy', arr=xy_train[0][0])
# np.save(save_path + 'keras56_x_test.npy', arr=xy_test[0][0])
np.save(save_path + 'keras56_y_train.npy', arr=xy_train[0][1])
# np.save(save_path + 'keras56_y_test.npy', arr=xy_test[0][1])

end2 = time.time()

print('넘파이 변경 소요 시간 : ', np.round(end2-start, 2))


# 이미지 수치화 소요 시간 :  31.86
# 넘파이 변경 소요 시간 :  47.43




