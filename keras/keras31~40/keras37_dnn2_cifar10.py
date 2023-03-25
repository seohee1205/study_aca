from tensorflow.keras.datasets import cifar10
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
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# 2차원으로 만들어주기(Scaler가 2차원에서만 되기 때문)
x_train = x_train.reshape(50000, 32*32*3 )
x_test = x_test.reshape(10000, 32*32*3)

print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
                                              # dtype=int64))
                                              
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 
print(y_train.shape)        # (50000, 10)
print(y_test.shape)         # (10000, 10)

# scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델 구성
model = Sequential()
model.add(Dense(50, input_shape = (3072,)))
model.add(Dropout(0.5))
model.add(Dense(42, activation = 'relu'))
model.add(Dropout(0.3))
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


model.fit(x_train, y_train, epochs = 2000, batch_size = 1500,
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


# result :  [1.5625195503234863, 0.43540000915527344]
# acc :  0.4354
