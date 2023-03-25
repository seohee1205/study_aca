# save_model과 비교

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model, load_model
from tensorflow.python.keras.layers import Dense,Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets['target']

print(type(x))      # <class 'numpy.ndarray'>
print(x)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 650
)

# 전처리 (정규화)는 데이터를 나눈 후 한다
scaler = MinMaxScaler()   # 하나로 모아줄 때  

x_train = scaler.fit_transform(x_train) # 위 두 줄과 같음
x_test = scaler.transform(x_test)

print(np.min(x_test), np.max(x_test)) # 0.0 1.0


#2. 함수형모델 구성
input1 = Input(shape=(13,)) # 인풋명시, 
dense1 = Dense(10, activation = 'relu')(input1)  # Dense 모델 구성 후, 이 모델은 어디에서 시작해서 어디에서 끝나는지 연결
dense2 = Dense(5, activation = 'relu')(dense1)
dense3 = Dense(7, activation = 'relu')(dense2)
dense4 = Dense(2, activation = 'relu')(dense3)
dense5 = Dense(4, activation = 'relu')(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs = input1, outputs = output1)

# model.save('./_save/keras26_1_save_model.h5')

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor= 'val_loss', patience= 10, mode = 'min',
                   verbose = 1, 
                   restore_best_weights= True
                   )
mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
        verbose = 1, 
        save_best_only= True,
        filepath= './_save/MCP/keras27_3_MCP.hdf5'
                      )
# Shift + Tab = 왼쪽으로 이동 

model.fit(x_train, y_train, epochs = 10000,
          callbacks=[es, mcp],
          validation_split= 0.2)


# model = load_model('./_save/MCP/keras27_ModelCheckPoint1.hdf5')

model.save('./_save/MCP/keras27_3_save_model.h5')


#4. 평가, 예측
from sklearn.metrics import r2_score

print("====================== 1. 기본 출력 ========================")
loss = model.evaluate(x_test, y_test, verbose=0)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

print("====================== 2. load_model 출력 ========================")
model2 = load_model('./_save/MCP/keras27_3_save_model.h5')
loss = model2.evaluate(x_test, y_test, verbose=0)
print('loss : ', loss)

y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

print("====================== 3. MCP 출력 ========================")
model3 = load_model('./_save/MCP/keras27_3_MCP.hdf5')
loss = model3.evaluate(x_test, y_test, verbose=0)
print('loss : ', loss)

y_predict = model3.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

# restore_best_weights= True (주석 처리 했을 때 = 1,2번은 같고 3번은 다름)
# loss :  70.62925720214844
# r2 스코어 :  -0.0002269744222462844
# ====================== 2. load_model 출력 ========================
# loss :  70.62925720214844
# r2 스코어 :  -0.0002269744222462844
# ====================== 3. MCP 출력 ========================
# loss :  70.6220703125
# r2 스코어 :  -0.00012513465354513365


# restore_best_weights= True (주석 처리 안 했을 때 = 1, 2, 3 다 똑같음) 
# ====================== 1. 기본 출력 ========================
# loss :  17.162548065185547
# r2 스코어 :  0.7569499541957669
# ====================== 2. load_model 출력 ========================
# loss :  17.162548065185547
# r2 스코어 :  0.7569499541957669
# ====================== 3. MCP 출력 ========================
# loss :  17.162548065185547
# r2 스코어 :  0.7569499541957669