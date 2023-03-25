from tensorflow.keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Input, GRU, Dropout
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time


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

# 이미지는 4차원이니까 다시 4차원으로 바꿔주기
x_train = x_train.reshape(50000, 32*32, 3)
x_test = x_test.reshape(10000, 32*32, 3)


#2. 모델 구성
input1 = Input(shape=(32*32, 3))
GRU1 = GRU(54, activation='relu',  return_sequences = True)(input1)
GRU2 = GRU(34, activation='relu',  return_sequences = True)(GRU1)
GRU3 = GRU(24)(GRU2)
dense1 = Dense(16,activation='relu')(GRU3)
dense2 = Dense(12,activation='relu')(dense1)
output1 = Dense(10, activation = 'softmax')(dense2)
model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

# 정의하기
start = time.time()
es = EarlyStopping(monitor = 'loss', patience = 55, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))


model.fit(x_train, y_train, epochs = 100, batch_size = 1500,
          validation_split= 0.2,
          verbose = 1,
          callbacks = [es])

end = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_acc, y_predict)
print('acc : ', acc)
print('걸린 시간 : ', np.round(end-start, 2))






