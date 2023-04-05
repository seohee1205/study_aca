
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical


save_path = path = 'd:/study_data/_save/cat_dog/'

x_train = np.load(save_path + 'keras58_8_cat_dog1_x_train.npy')
x_test = np.load(save_path + 'keras58_8_cat_dog1_x_test.npy')
y_train = np.load(save_path + 'keras58_8_cat_dog1_y_train.npy')
y_test = np.load(save_path + 'keras58_8_cat_dog1_y_test.npy')


#2. 모델구성
model = Sequential()
model.add(Conv2D(128, 2, input_shape=(100, 100, 3)))
model.add(Conv2D(64, 2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy',
              optimizer = 'adam', metrics = ['acc'])

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

model.fit(x_train, y_train, epochs = 10,   # x데이터, y데이터, batch
                    validation_split = 0.2,
                    batch_size = 130,
                    callbacks = [es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


y_predict = model.predict(x_test)
y_predict = np.round(y_predict)
acc = accuracy_score(y_test, y_predict)

print('acc : ', acc)

# loss :  [91.12712097167969, 0.5]
# acc :  0.5
