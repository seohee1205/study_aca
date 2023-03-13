#  전처리 (정규화)
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping

#1. 데이터
path = './_data/dacon_diabetes/'
path_save = './_save/dacon_diabetes/'

train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)

print(train_csv)
print(train_csv.shape)  # (652, 9)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)

print(test_csv)
print(test_csv.shape)   # (116, 8)



# 보스턴 임포트 못 하는 사람 (1.2 부터 안 되니까 1.1 버전 설치)
# pip uninstall scikit-learn     # 사이킷런 삭제
# pip install scikit- learn==1.1.0      # 1.1 버전 설치

# print(np.min(x), np.max(x)) # 0.0 711.0
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)     # 변환
# print(np.min(x), np.max(x)) # 0.0 1.0

# 결측치 제거
print(train_csv.info())     # 결측치 없음

x = train_csv.drop(['Outcome'], axis = 1)
print(x)
y = train_csv['Outcome']
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.8, random_state= 333
)

print(x_train.shape, x_test.shape)  # (521, 8) (131, 8)
print(y_train.shape, y_test.shape)  # (521,) (131,)



# 전처리 (정규화)는 데이터를 나눈 후 한다
# scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0

test_csv = scaler.transform(test_csv)   # test_csv에도 sclaer해줘야 함


#2. 모델 
model = Sequential()
model.add(Dense(6, input_dim = 8))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(1, activation = 'relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics= ['accuracy'])

# 정의하기
es = EarlyStopping(monitor = 'val_accuracy', patience = 152, mode = 'max',
                   verbose = 1,
                    restore_best_weights = True)

hist = model.fit(x_train, y_train, epochs = 5000, batch_size = 15,
          validation_split = 0.3,
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
results :  [6.358375072479248, 0.5877862572669983]
acc :  0.5877862595419847
RMSE :  0.6420387375057796

 
# scaler = StandardScaler() 
loss :  [0.6096569895744324, 0.7404580116271973]
r2 스코어 :  -2.0460350383613424
RMSE :  0.4468081998142009


# scaler = MaxAbsScaler()
loss :  [6.358375072479248, 0.5877862572669983]
r2 스코어 :  -69.42264185226927
RMSE :  0.7628957373034344


# scaler = RobustScaler()
loss :  [6.358375072479248, 0.5877862572669983]
r2 스코어 :  -2262.44405741446
RMSE :  0.6527500237984166

'''
