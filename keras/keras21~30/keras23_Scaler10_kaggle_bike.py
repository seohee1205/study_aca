import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)          # datetime은 colume 0번째이니까 index 처리하면 데이터 취급 X

print(train_csv)
print(train_csv.shape)      # (10886, 11)


test_csv = pd.read_csv(path + 'test.csv',
                       index_col = 0)
print(test_csv)
print(test_csv.shape)       # (6493, 8)

#==========================================
print(train_csv.columns)
print(test_csv.columns)
print(train_csv.describe())
print(type(train_csv))      # <class 'pandas.core.frame.DataFrame'>

###################### 결측치 처리 ##################
# 결측치 제거
print(train_csv.info())     # 결측치 없음


############################ train_csv 데이터에서 x와 y 분리
# train에는 있지만 test에는 없는 'casual'. 'registered'는 삭제

x = train_csv.drop(['count', 'casual', 'registered'], axis = 1)
print(x)
y = train_csv['count']
print(y)

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
scaler = RobustScaler()
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


y_submit = model.predict(test_csv)
# print(y_submit)

'''
# 파일 생성
submission = pd.read_csv(path + 'samplesubmission.csv', index_col = 0)
# print(submission)
submission['count'] = y_submit
# print(submission)

path_save = './_save/kaggle_bike/'
submission.to_csv(path_save + 'submit_0310_0752.csv')
'''


'''
# scaler = MinMaxScaler() 
loss :  24479.375
r2 스코어 :  -1.9464606339533783
RMSE :  156.4588559988911


# scaler = StandardScaler() 
loss :  24165.287109375
r2 스코어 :  -1.5670971607989488
RMSE :  155.4518722416699


# scaler = MaxAbsScaler()
loss :  24867.8828125
r2 스코어 :  -1.687211939484385
RMSE :  157.69553617348853


# scaler = RobustScaler()
loss :  23392.2578125
r2 스코어 :  -1.0825096521649828
RMSE :  152.94527605972337


'''

