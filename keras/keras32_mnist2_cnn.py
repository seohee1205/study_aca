from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

##### 실습 #####

print(np.unique(y_train, return_counts=True))    # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
                                                 #  dtype=int64))


print(y_test)   # [7 2 1 ... 4 5 6]

# 2차원으로 만들어주기(Scaler가 2차원에서만 되기 때문)
x_train = x_train.reshape(60000, 28*28 )
x_test = x_test.reshape(10000, 28*28)


# y 원핫 
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 
    
print(y_train)  # [5 0 4 ... 5 6 8]
print(y_train.shape)    # (60000, 10)
print(y_test)   # [7 2 1 ... 4 5 6]
print(y_test.shape)     # (10000, 10)

Scaler = MinMaxScaler()     # 2차원에서만 됨
Scaler.fit(x_train) 
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)

# Scaler랑 똑같음
# x_train = x_train/255.
# x_test = x_test/255.

#2. 모델 구성

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# reshape는 구조만 바뀌고, 순서와 내용(값)은 바뀌지 않는다 / 곱해서 합치면 동일 


print(np.max(x_train), np.min(x_train))     # (1.0 0.0)


model = Sequential()
model.add(Conv2D(10, (2, 2), padding = 'same', input_shape = (28, 28, 1)))
model.add(Dropout(0.5))
model.add(Conv2D(filters = 5, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(filters = 6, kernel_size = (4,4), padding = 'valid', activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))


#3. 컴파일, 훈련
start_time = time.time()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'loss', patience = 15, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

model.fit(x_train, y_train, epochs = 100, batch_size = 5000,
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
