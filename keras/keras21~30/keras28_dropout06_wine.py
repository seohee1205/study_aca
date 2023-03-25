import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from tensorflow.keras.utils import to_categorical   # (케라스에 원핫)

#1. 데이터
datasets = load_wine()
print(datasets.DESCR)   # 판다스에서 describe() 와 동일
print(datasets.feature_names) # 판다스에서 clolumns 와 동일
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (150, 4) (150,)
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y))    # y의 라벨값 :  [0 1 2]

################# 이 지점에서 원핫을 해줘야 함 #######################

y = to_categorical(y)
# print(y)
print(y.shape)      # (150, 3)

## y를 (150, ) -> (150, 3)

# 판다스에 겟더미




# 사이킷런에 원핫인코더

#####################################################################


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state= 333, 
    train_size = 0.8, 
    stratify = y        # 데이터를 일정 비율로 분배해라 
)
print(y_train)
print(np.unique(y_train, return_counts= True))

# 전처리 (정규화)는 데이터를 나눈 후 한다
# scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0



# #2. 모델 구성
model = Sequential()
model.add(Dense(50, activation = 'relu', input_dim = 4))
model.add(Dropout(0.3))
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax')) # 3개의 확률값의 합 = 1
# # 다중분류 문제는 마지막 레이어의 activation = softmax, 주의: 모델 마지막 부분(output)은 y의 라벨값의 개수만큼

#(함수형) 모델 구성
# input1 = Input(shape=(4,)) # 인풋명시, 
# dense1 = Dense(50)(input1)  # Dense 모델 구성 후, 이 모델은 어디에서 시작해서 어디에서 끝나는지 연결
# dense2 = Dense(40, activation = 'relu')(dense1)
# dense3 = Dense(40, activation = 'relu')(dense2)
# dense4 = Dense(10, activation = 'relu')(dense3)
# output1 = Dense(3, activation = 'softmax')(dense4)
# model = Model(inputs = input1, outputs = output1)



#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics= ['acc'])

import datetime
date = datetime.datetime.now()
print(date)  # 2023-03-14 11:11:30.046663
date = date.strftime("%m%d_%H%M") # 시간을 문자로 (월, 일, 시간, 분)
print(date)  # 0314_1116

filepath = './_save/MCP/keras28/'
filename = '{epoch:04d}-{val_loss:4f}.hdf5'


# 정의하기
es = EarlyStopping(monitor = 'val_loss', patience = 55, mode = 'min',
                   verbose = 1,
                    restore_best_weights = True)

mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
        verbose = 1, 
        save_best_only= True,
        filepath="".join([filepath, 'k27_', date, '_', filename]))


model.fit(x_train, y_train, epochs = 1000, batch_size = 10,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])



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
loss :  [0.31838056445121765, 0.8999999761581421]
r2 스코어 :  0.6743670164191142
RMSE :  0.24169633

'''
