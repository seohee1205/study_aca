# 150 * 150 * 3

# https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data

# 넘파이까지 저장
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import time



# np.save(path + '파일명', arr = ???)
#1. time.time()으로 이미지 수치화하는 시간 체크할 것
#2. time.time()으로 넘파이로 변경하는 시간 체크할 것


#1. 데이터
start = time.time()
path = 'd:/study_data/_save/dogs_breed/'

start2 = time.time()
x = np.load(path + 'keras56_x_train500.npy')
y = np.load(path + 'keras56_y_train500.npy')

end = time.time()

# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8,
                                                    random_state = 123, shuffle = True)

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape= (150, 150, 4), activation= 'relu'))
model.add(Conv2D(64, (3,3), activation= 'relu'))
model.add(Flatten())
model.add(Dense(16, activation= 'relu'))
model.add(Dense(5, activation= 'softmax'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam', metrics = ['acc'])

# model.fit(xy_train[:][0], xy_train[:][1], epochs = 10)  # 에러

# hist = model.fit(xy_train[0][0], xy_train[0][1], epochs = 10,   # 통배치 넣으면 가능
#           batch_size = 16, 
#           validation_data= (xy_test[0][0], xy_test[0][1]))
  
# hist = model.fit_generator(xy_train, epochs = 30,   # x데이터, y데이터, batch
#                     steps_per_epoch = 32,   # 전체데이터크기/batch 사이즈 = 160/5 = 32
#                     validation_data= xy_test,
#                     validation_steps= 24,   # 발리데이터/batch = 120/5 = 24
# )

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

hist = model.fit(x_train, y_train, epochs = 1000,   # x데이터, y데이터, batch
                    # steps_per_epoch = 32,   # 전체데이터크기/batch 사이즈 = 160/5 = 32
                    validation_split = 0.2,
                    # validation_steps= 24,   # 발리데이터/batch = 120/5 = 24
                    callbacks = [es]
)


# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# acc = hist.history['acc']
# val_acc = hist.history['val_acc']

# print(acc)
# print('loss : ', loss[-1])  
# print('val_loss : ', val_loss[-1])  
# print('acc : ', acc[-1]) 
# print('val_acc : ', val_acc[-1]) 

end2 = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_pred = model.predict(x_test)
print('acc : ', accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))


# loss : [1.5436173677444458, 0.3106796145439148]
# acc :  0.3106796116504854

