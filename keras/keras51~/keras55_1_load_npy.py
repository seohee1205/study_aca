import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터

path = 'd:/study_data/_save/_npy/'
# np.save(path + 'keras55_1_x_train.npy', arr=xy_train[0][0])
# np.save(path + 'keras55_1_x_test.npy', arr=xy_train[0][0])
# np.save(path + 'keras55_1_y_train.npy', arr=xy_train[0][1])
# np.save(path + 'keras55_1_y_test.npy', arr=xy_train[0][1])


x_train = np.load(path + 'keras55_1_x_train.npy')
x_test = np.load(path + 'keras55_1_x_test.npy') 
y_train = np.load(path +'keras55_1_y_train.npy')
y_test = np.load(path + 'keras55_1_y_test.npy')

# print
print(x_train.shape, x_test.shape)   # (160, 100, 100, 1) (120, 100, 100, 1) /  minmax로 전처리까지 돼있음
print(y_train.shape, y_test.shape)   # (160,) (120,)




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
model.compile(loss = 'binary_crossentropy',
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

es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

hist = model.fit(x_train, y_train, epochs = 30,   # x데이터, y데이터, batch
                    # steps_per_epoch = 32,   # 전체데이터크기/batch 사이즈 = 160/5 = 32
                    validation_data = (x_test, y_test),
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
    
    
    

# #1. 그림그리기 
# import matplotlib.pyplot as plt
# plt.subplot(1,2,1)
# plt.plot(range(len(hist.history['loss'])),hist.history['loss'],label='loss')
# plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label='val_loss')
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(range(len(hist.history['acc'])),hist.history['acc'],label='acc')
# plt.plot(range(len(hist.history['val_acc'])),hist.history['val_acc'],label='val_acc')
# plt.legend()
# plt.show()


