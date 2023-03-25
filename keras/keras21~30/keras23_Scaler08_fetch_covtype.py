import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical   # (케라스에 원핫)

#1. 데이터
datasets = fetch_covtype()          # 오류나면 sklearn 삭제하고 다시 설치, cmd 창에 pip uninstall scikit-learn
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x.shape, y.shape)     # (581012, 54) (581012,)
print(x)
print(y)
print('y의 라벨값 : ', np.unique(y))        # y의 라벨값 :  [1 2 3 4 5 6 7]

######################### 원핫 #####################
y = to_categorical(y)       # y의 라벨값이 1부터 시작하는 데이터에 0이 자동으로 추가됨
y=np.delete(y, 0, axis=1)   # 앞에 자동으로 추가된 열 삭제
# print(y)
print(y.shape)          # (581012, 8)

#####################################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, random_state= 2000,
    train_size= 0.8, stratify = y
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

#2. 모델 구성
model = Sequential()
model.add(Dense(50,activation = 'relu', input_dim = 54 ))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(7, activation = 'softmax'))

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])

# 정의하기
es = EarlyStopping(monitor = 'val_loss', patience = 55, mode = 'min',
                   verbose = 1,
                   restore_best_weights = True)


model.fit(x_train, y_train, epochs = 2000, batch_size = 5000,
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
# scaler = MinMaxScaler() 
loss :  [0.24726928770542145, 0.9012417793273926]
r2 스코어 :  0.6788020244994685
RMSE :  0.14291944
 
# scaler = StandardScaler() 
loss :  [0.19262652099132538, 0.9255871176719666]
r2 스코어 :  0.7516323143039403
RMSE :  0.12495616

# scaler = MaxAbsScaler()
loss :  [0.2575257420539856, 0.8986514806747437]
r2 스코어 :  0.6820219033250622
RMSE :  0.14556749

# scaler = RobustScaler()
loss :  [0.19338463246822357, 0.9262927770614624]
r2 스코어 :  0.7675418032355454
RMSE :  0.124693565

'''
