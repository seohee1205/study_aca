from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical


save_path = 'd:/study_data/_save/mnist/'

x_train = np.load(save_path + 'keras58_2_mnist_x_train.npy')
x_test = np.load(save_path + 'keras58_2_mnist_x_test.npy')
y_train = np.load(save_path + 'keras58_2_mnist_y_train.npy')
y_test = np.load(save_path + 'keras58_2_mnist_y_test.npy')


#2. 모델구성
model = Sequential()
model.add(Conv2D(256, (2,2), input_shape= (28, 28, 1), activation= 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (2,2), activation= LeakyReLU(0.8)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), activation= LeakyReLU(0.8)))
model.add(Flatten())
model.add(Dense(32, activation= 'relu'))
model.add(Dense(64, activation= 'relu'))
model.add(Dense(32, activation= 'relu'))
model.add(Dense(10, activation= 'softmax'))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy',
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
y_predict = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_acc, y_predict)

print('acc : ', acc)

# loss :  [0.05335493013262749, 0.9857000112533569]
# acc :  0.9857