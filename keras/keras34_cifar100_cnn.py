from tensorflow.keras.datasets import cifar100
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 1)


x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)


# 원핫
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(np.unique(y_train, return_counts=True))   # (array([0., 1.], dtype=float32), array([4950000, 50000], dtype=int64))

print(y_train)
print(y_train.shape)    # (50000, 100)
print(y_test) 
print(y_test.shape)     # (10000, 100)


Scaler = MinMaxScaler()     # 2차원에서만 됨
Scaler.fit(x_train) 
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)

# 이미지는 4차원이니까 다시 4차원으로 바꿔주기
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# Scaler랑 똑같음
# x_train = x_train/255.
# x_test = x_test/255.


#2. 모델 구성
model = Sequential()
model.add(Conv2D(30, (2, 2), padding = 'same', input_shape = (32, 32, 3)))
model.add(Dropout(0.5))
model.add(MaxPooling2D())   # 중첩되지 않은 부분 중 가장 큰 것
model.add(Conv2D(filters = 32, kernel_size = (2, 2), padding = 'valid', activation = 'relu'))
model.add(Conv2D(12, 2))
model.add(Flatten())
model.add(Dense(8, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

#3. 컴파일, 훈련
start_time = time.time()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'loss', patience = 60, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

model.fit(x_train, y_train, epochs = 50, batch_size = 5000,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

end_time = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_acc, y_predict)
print('acc : ', acc)
print('time', round(end_time-start_time,2))