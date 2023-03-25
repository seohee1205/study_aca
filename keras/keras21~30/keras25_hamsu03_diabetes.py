#  전처리 (정규화)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets['target']

print(type(x))      # <class 'numpy.ndarray'>
print(x)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 650874
)

# 전처리 (정규화)는 데이터를 나눈 후 한다
# scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0
 

# #2. 모델 
# model = Sequential()
# model.add(Dense(15, input_dim = 10))
# model.add(Dense(7, activation = 'relu'))
# model.add(Dense(10, activation = 'relu'))
# model.add(Dense(13, activation = 'relu'))
# model.add(Dense(5, activation = 'relu'))
# model.add(Dense(1))

#(함수형) 모델 구성
input1 = Input(shape=(10,)) # 인풋명시, 
dense1 = Dense(15)(input1)  # Dense 모델 구성 후, 이 모델은 어디에서 시작해서 어디에서 끝나는지 연결
dense2 = Dense(7, activation = 'relu')(dense1)
dense3 = Dense(10, activation = 'relu')(dense2)
dense4 = Dense(13, activation = 'relu')(dense3)
dense5 = Dense(5, activation = 'relu')(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs = input1, outputs = output1)


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
# 정의하기

es = EarlyStopping(monitor= 'val_loss', patience = 20, mode= 'min',     # if mode = auto: 자동으로 min 또는 max로 맞춰줌 
              verbose= 1,   
              restore_best_weights= True)  


hist = model.fit(x_train, y_train, epochs = 1000, batch_size =25,
                 validation_split = 0.2, 
                 verbose = 1,
                 callbacks= [es]    # es 호출
                 )


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)              # RMSE 함수 사용
print("RMSE : ", rmse)



'''
# scaler = MinMaxScaler() 
loss :  2517.78125
r2 스코어 :  0.2354107790820431
RMSE :  50.17749790510487
 
# scaler = StandardScaler() 
loss :  2620.727294921875
r2 스코어 :  0.2540607562499685
RMSE :  51.19303831525312

# scaler = MaxAbsScaler()
loss :  2486.43505859375
r2 스코어 :  0.2991667980703582
RMSE :  49.86416573807331

# scaler = RobustScaler()
loss :  2567.7939453125
r2 스코어 :  0.3032844675416295
RMSE :  50.67340435037842

'''
