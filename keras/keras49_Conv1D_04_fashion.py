from tensorflow.keras.datasets import cifar100, fashion_mnist
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input, Conv1D, Flatten
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)   # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)     # (10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
                                              # dtype=int64))

# 2차원으로 만들어주기(Scaler가 2차원에서만 되기 때문)
x_train = x_train.reshape(60000, 28*28 )
x_test = x_test.reshape(10000, 28*28)


# y 원핫
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 
    
# print(y_train)  # [5 0 4 ... 5 6 8]
# print(y_train.shape)    # (60000, 10)
# print(y_test)   # [7 2 1 ... 4 5 6]
# print(y_test.shape)     # (10000, 10)

Scaler = MinMaxScaler()     # 2차원에서만 됨
Scaler.fit(x_train) 
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)

# 이미지는 4차원이니까 다시 4차원으로 바꿔주기
x_train = x_train.reshape(60000, 28,28)
x_test = x_test.reshape(10000, 28,28)

#2. 모델 구성
input1 = Input(shape=(28, 28))
Conv1 = Conv1D(64, 2, activation='linear')(input1)
Conv2 = Conv1D(26, 2, activation='relu')(Conv1)
Flat1 = Flatten()(Conv2)
dense2 = Dense(16, activation='relu')(Flat1)
dense3 = Dense(12, activation='relu')(dense2)
output1 = Dense(10, activation = 'relu')(dense3)
model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
start_time = time.time()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'loss', patience = 100, mode = 'auto',
                   verbose = 1, restore_best_weights = True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath= './_save/MCP/keras27_3_MCP.hdf5')

model.fit(x_train, y_train, epochs = 100, batch_size = 1000,
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

# 걸린 시간
print('time', round(end_time-start_time,2))


# result :  [nan, 0.10000000149011612]
# acc :  0.1
# time 46.95