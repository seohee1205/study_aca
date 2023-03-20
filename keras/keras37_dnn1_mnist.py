from tensorflow.keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score


# [실습]
# 목표: cnn성능보다 좋게 만들기

#1. 데이터
# reshape 해주기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28 )
x_test = x_test.reshape(10000, 28*28)

print(np.unique(y_train, return_counts = True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
                                                # dtype=int64))

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 

print(y_train.shape)    # (60000, 10)
print(y_test.shape)     # (10000, 10)

# scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델구성
model = Sequential()
model.add(Dense(50, input_shape = (784,)))
# = model.add(Dense(64, input_shape = (28*28,)))
model.add(Dropout(0.3))
model.add(Dense(42, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'loss', patience = 55, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))


model.fit(x_train, y_train, epochs = 2000, batch_size = 512,
          validation_split= 0.2,
          verbose = 1,
          callbacks = [es])

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_acc, y_predict)
print('acc : ', acc)

# 시간 측정
import datetime
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')


# result :  [0.20836767554283142, 0.9531000256538391]
# acc :  0.9531

# result :  [0.2052716761827469, 0.9539999961853027]
# acc :  0.954