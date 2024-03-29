import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. 데이터
train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

xy_train = train_datagen.flow_from_directory(
'C:/study_aca/brain/train/',
target_size= (100, 100),
batch_size = 16,
class_mode= 'binary',
color_mode= 'grayscale',
shuffle=True
)   # Found 133 images belonging to 2 classes.


xy_test = test_datagen.flow_from_directory(
'C:/study_aca/brain/test/',
target_size= (100, 100),
batch_size = 16,
class_mode= 'binary',
color_mode= 'grayscale',
shuffle=True
)   # Found 133 images belonging to 2 classes.

# print(len(xy_train))    # 
# print(len(xy_train[0])) 

# print(len(xy_test))     # 


#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model=Sequential()
model.add(Conv2D(32, (2, 2), input_shape = (100, 100, 1)))
model.add(Flatten())
model.add(Dense(16))
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# model.fit(xy_train, epochs = 1, batch_size = 16)


#4. 평가 및 예측
results = model.evaluate(xy_test)

print('loss : ' , results[0])
print('acc : ' , results[1])


