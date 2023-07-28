
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from tensorflow.keras.applications import ResNet101V2
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, GlobalAvgPool2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import time

train_datagen = ImageDataGenerator(
                rescale=1./255, 
                validation_split=0.2)


test_datagen = ImageDataGenerator(rescale=1./255)

xy = train_datagen.flow_from_directory(
    'd:/study_data/_data/cat_dog/Petimages/',
    target_size = (50, 50),
    batch_size = 25000,
    class_mode = 'binary',
    shuffle = True)

x = xy[0][0]
y = xy[0][1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle= True, random_state= 337)

#2. 모델
res101 = ResNet101V2(weights='imagenet', include_top=False,
              input_shape=(50, 50, 3))
res101.trainable = True

model = Sequential()
model.add(res101)
model.add(GlobalAvgPool2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 
learning_rate = 1e-4
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss = "binary_crossentropy", optimizer=optimizer, metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='amin', verbose=1, factor=0.2)

start = time.time()
hist = model.fit(x_train, y_train, epochs=10, steps_per_epoch=50, validation_split=0.2 , validation_steps=50, callbacks=[es, reduce_lr]) 
end = time.time() - start

#4. 예측
loss = model.evaluate(x_test, y_test)
print("걸린 시간 : ", round(end, 2))
y_pred = np.round(model.predict(x_test))
y_test = np.round(y_test)
acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

path = 'D:\study_data\_data\나.jpg'

img = image.load_img(path, target_size=(50, 50))

x = image.img_to_array(img)/255.

x = x.reshape(1, *x.shape)
print(x)
# x = preprocess_input(x)
x_pred = model.predict(x)
print(x_pred.shape)
print(x_pred)

if x_pred > 0.5 :
    print('당신은 개 입니다')
    
else :
    print('당신은 고양이 입니다')
    
# (1, 1)
# [[0.12569068]]
# 당신은 고양이 입니다

