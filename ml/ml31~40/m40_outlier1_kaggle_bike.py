# 이상치 결측치 처리해서 기존 결과와 비교

import numpy as np
from sklearn.covariance import EllipticEnvelope
import pandas as pd
from xgboost import XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['count', 'casual', 'registered'], axis = 1)
# print(x)
y = train_csv['count']
# print(y)

# 결측치 처리
imputer = IterativeImputer(estimator=XGBRegressor())
x = imputer.fit_transform(x)

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75], axis=0)
    print('1사분위 : ', quartile_1) 
    print('q2 : ', q2) 
    print('3사분위 : ', quartile_3) 
    iqr = quartile_3 - quartile_1 
    print('iqr : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5) 
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
outliers_loc = outliers(x)
print('이상치의 위치 : ', list((outliers_loc[0], outliers_loc[1])))

x[outliers_loc] = 999999999

# import matplotlib.pyplot as plt
# plt.boxplot(x)
# plt.show()


############################### train_csv 데이터에서 x와 y 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size = 0.95, random_state= 1912
)

print(x_train.shape, x_test.shape)  # (7620, 8) (3266, 8)
print(y_train.shape, y_test.shape)  # (7620,) (3266,)

# 전처리 (정규화)는 데이터를 나눈 후 한다
# scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0

test_csv = scaler.transform(test_csv)   # test_csv에도 sclaer해줘야 함

#2. 모델 구성
model = Sequential()
model.add(Dense(20, input_dim = 8))
model.add(Dense(15, activation = 'relu'))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(11, activation = 'relu'))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

# 정의하기

es = EarlyStopping(monitor= 'val_loss', patience = 35, mode= 'min',     # if mode = auto: 자동으로 min 또는 max로 맞춰줌 
              verbose= 1,    
              restore_best_weights= True)  


hist = model.fit(x_train, y_train, epochs = 300, batch_size = 35,
                 validation_split = 0.1,
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


# loss :  28118.9765625
# r2 스코어 :  -3.8579545478655977
# RMSE :  167.6871404250946