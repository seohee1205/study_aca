# 함수형 모델

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib

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
# scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0




# 모델
# model = Sequential()
# model.add(Dense(10, input_dim = 13, name= 'S1' ))
# model.add(Dense(5, name= 'S2'))
# model.add(Dense(7, name= 'S3' ))
# model.add(Dense(2, name= 'S4'))
# model.add(Dense(4, name= 'S5' ))
# model.add(Dense(1))
# model.summary()

#2. 함수형 모델 구성
input1 = Input(shape=(13,), name = 'h1') # 인풋명시, 
dense1 = Dense(10, name = 'h2', activation = 'relu')(input1)  # Dense 모델 구성 후, 이 모델은 어디에서 시작해서 어디에서 끝나는지 연결
dense2 = Dense(5, name = 'h3', activation = 'relu')(dense1)
dense3 = Dense(7, name = 'h4', activation = 'relu')(dense2)
dense4 = Dense(2, name = 'h5', activation = 'relu')(dense3)
dense5 = Dense(4, name = 'h6', activation = 'relu')(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs = input1, outputs = output1)



#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

# 정의하기

es = EarlyStopping(monitor= 'val_loss', patience = 25, mode= 'min',     # if mode = auto: 자동으로 min 또는 max로 맞춰줌 
              verbose= 1,    # val-loss를 기준으로 할 것이고, 5번 참을 것이다. / val-loss의 최솟값을 찾아라
              restore_best_weights= True)  # 최적(최소 loss)의 w 값을 반환한다.


hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 30,
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
loss :  202.77447509765625
r2 스코어 :  -17610223.053002313
RMSE :  14.239890052307342
 
# scaler = StandardScaler() 
loss :  10.598248481750488
r2 스코어 :  0.8342501706494181
RMSE :  3.2554951577071503

# scaler = MaxAbsScaler()
loss :  12.886128425598145
r2 스코어 :  0.7933013240365123
RMSE :  3.5897253871998074

# scaler = RobustScaler()
loss :  14.537967681884766
r2 스코어 :  0.7840139166781994
RMSE :  3.812868646025028

'''



