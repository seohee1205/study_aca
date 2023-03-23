from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, SimpleRNN, LSTM, Input, Dropout, MaxPooling2D
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical



#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x))      # <class 'numpy.ndarray'>
print(x.shape, y.shape)     # (506, 13) (506,)

#1-1. x, y 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 650
)
print(x_train.shape, y_train.shape)     # (404, 13) (404,)
print(x_test.shape, y_test.shape)       # (102, 13) (102,)


# 2차원으로 만들어주기(Scaler가 2차원에서만 되기 때문)
x_train = x_train.reshape(404, 13*1*1 )
x_test = x_test.reshape(102, 13*1*1)


# scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(404, 13, 1)
x_test = x_test.reshape(102, 13, 1)


#2. 함수형모델 구성
input1 = Input(shape=(13, 1))
LSTM1 = LSTM(64, activation='linear')(input1)
dense1 = Dense(26, activation='relu')(LSTM1)
dense2 = Dense(16, activation='relu')(dense1)
dense3 = Dense(12, activation='relu')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

# model.summary()


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'loss', patience = 60, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 1000, batch_size = 500,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])  # mcp

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)


# result :  25.23102569580078
# r2 스코어 :  0.6426869875979253
