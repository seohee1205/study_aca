from tensorflow.keras.datasets import cifar100
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

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

print(np.unique(y_train, return_counts = True))

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 
print(y_train.shape)
print(y_test.shape) 

# scaler
scaler = MinMaxScaler()     # 2차원에서만 됨
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_shape = (32*32*3,)))
# = model.add(Dense(64, input_shape = (28*28,)))
model.add(Dropout(0.4))
model.add(Dense(42, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(12, activation = 'relu'))
model.add(Dense(100, activation = 'softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'loss', patience = 60, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

model.fit(x_train, y_train, epochs = 2000, batch_size = 333,
          validation_split = 0.2,
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


# result :  [3.6092426776885986, 0.15209999680519104]
# acc :  0.1521

# result :  [3.542661428451538, 0.16439999639987946]
# acc :  0.1644