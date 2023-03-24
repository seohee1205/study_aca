from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, Input, Flatten
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets['target']

#1-2. x, y 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 777
)

# print(x_train.shape, y_train.shape)     # (353, 10) (353,)
# print(x_test.shape, y_test.shape)       # (89, 10) (89,)

#1-3.  2차원으로 만들어주기(Scaler가 2차원에서만 되기 때문)
x_train = x_train.reshape(353, 10*1*1)
x_test = x_test.reshape(89, 10*1*1)

#1-3 scaler
scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(353, 10, 1)
x_test = x_test.reshape(89, 10, 1)

#2. 모델
input1 = Input(shape = (10, 1))
Conv1 = Conv1D(20, 3)(input1)
Conv2 = Conv1D(16, 3)(Conv1)
Flat1 = Flatten()(Conv2)
dense1 = Dense(10, activation = 'relu')(Flat1)
dense2 = Dense(8, activation = 'relu')(dense1)
dense3 = Dense(8, activation = 'relu')(dense2)
dense4 = Dense(4, activation = 'relu')(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs = input1, outputs=output1)
# model.summary()

#3. 컴파일, 훈련
start = time.time()

model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'loss', patience = 60, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 500, batch_size = 32,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

end = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)
print('걸린 시간 : ', np.round(end-start, 2))


# result :  2450.552490234375
# r2 스코어 :  0.5527838677410536
# 걸린 시간 :  54.2

