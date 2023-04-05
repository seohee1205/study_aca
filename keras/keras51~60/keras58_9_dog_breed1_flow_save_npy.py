# 수치형으로 제공된 데이터를 증폭

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
np.random.seed(0)


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

test_datagen = ImageDataGenerator(
    rescale = 1./255,
# 테스트 데이터는 평가하는 데이터이기 때문에 데이터를 증폭한다는 건 결과를 조작하는 것, 때문에 스케일을 제외한 옵션은 삭제
# 통상적으로 테스트 데이터는 증폭(전처리)하지 않음
)

xy_train = train_datagen.flow_from_directory(   # 이미지제너레이터는 폴더별로 라벨값 부여
    
   'd:/study_data/_data/dogs_breed/',     # 분류된 폴더의 상위폴더까지 지정  # directory= 폴더
    target_size=(100, 100),           # 수집한 데이터마다 이미지 사진크기 다르므로 이미지크기 동일하게 고정
    batch_size = 5,  # 전체 데이터 쓰려면 160(전체 데이터 개수) 이상 넣기 / # 5장씩 잘라라
    class_mode = 'categorical',           # 0,1로 구별(nomal,ad) / 0,1,2(가위,바위,보)// # 원핫사용한 경우 => 'categorical'
    # color_mode = 'grayscale',
    color_mode = 'rgba',
    shuffle= True,
)       # Found 160 images belonging to 2 classes.      0과 1의 클래스로 분로되었다.    # x = 160, 200. 200. 1로 변환됨, y= 160,


x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_train,x_test, y_train,y_test = train_test_split(x_train, y_train, train_size=0.7, shuffle=True, random_state=123)


augment_size = 1000

randidx = np.random.randint(x_train.shape[0], size = augment_size)
print(randidx)          # [28678 44651 21849 ... 29255 51144 18206]
print(randidx.shape)    # (40000,)
print(np.min(randidx), np.max(randidx)) # 3 59999

x_augmented = x_train[randidx].copy() 
y_augmented = y_train[randidx].copy()      

# (1)     # 증폭하는 부분
x_augmented = train_datagen.flow(
    x_augmented, y_augmented, batch_size= augment_size, shuffle= False
).next()[0]

x_train = np.concatenate((x_train/255., x_augmented), axis = 0)
y_train = np.concatenate((y_train, y_augmented), axis = 0)
x_test = x_test/255.

print(x_train.shape, y_train.shape)    # (100000, 28, 28, 1) (100000,)


save_path = path = 'd:/study_data/_save/dogs_breed/'

np.save(save_path + 'keras58_9_dogs_breed_x_train.npy', arr = x_train)
np.save(save_path + 'keras58_9_dogs_breed_x_test.npy', arr = x_test)
np.save(save_path + 'keras58_9_dogs_breed_y_train.npy', arr = y_train)
np.save(save_path + 'keras58_9_dogs_breed_y_test.npy', arr = y_test)

