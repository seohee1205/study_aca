from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)    # (60000, 28, 28)
# print(x_test.shape)     # (10000, 28, 28)
# print(np.unique(y_train, return_counts=True)) # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
                                                 # dtype=int64))

# 2차원으로 만들어주기(Scaler가 2차원에서만 되기 때문)
x_train = x_train.reshape(60000, 28*28 )
x_test = x_test.reshape(10000, 28*28)


Scaler = MinMaxScaler()     # 2차원에서만 됨
Scaler.fit(x_train) 
x_train = Scaler.transform(x_train)
x_test = Scaler.transform(x_test)

# 이미지는 4차원이니까 다시 4차원으로 바꿔주기
x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. (함수형) 모델 구성
input1 = Input(shape=(28, 28))
Conv1 = Conv1D(64, 2, activation='linear')(input1)
Conv2 = Conv1D(26, 2, activation='relu')(Conv1)
Flat1 = Flatten()(Conv2)
dense2 = Dense(16, activation='relu')(Flat1)
dense3 = Dense(12, activation='relu')(dense2)
output1 = Dense(10)(dense3)
model = Model(inputs=input1, outputs=output1)
 
#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])


# 정의하기
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

start = time.time()
es = EarlyStopping(monitor= 'val_loss', patience = 20, mode= 'min',     # if mode = auto: 자동으로 min 또는 max로 맞춰줌 
              verbose= 1,    
              restore_best_weights= True)  

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath= './_save/MCP/keras27_3_MCP.hdf5')

hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 500,
                 validation_split = 0.2, 
                 verbose = 1,
                 callbacks= [es]    # es 호출
                 )

end = time.time()

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
print('result : ', results)
print('loss :', results[0])
print('acc :', results[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test_acc, y_predict)
print('걸린 시간 : ', np.round(end-start, 2))

# loss : 9.699861526489258
# acc : 0.10339999943971634
# acc :  0.1034
# 걸린 시간 :  22.35
