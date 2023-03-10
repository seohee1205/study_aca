# 알리데이션: 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

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
# Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
#        'humidity', 'windspeed', 'casual', 'registered', 'count'],
#       dtype='object')

print(test_csv.columns)
#  0   season      10886 non-null  int64
#  1   holiday     10886 non-null  int64
#  2   workingday  10886 non-null  int64
#  3   weather     10886 non-null  int64
#  4   temp        10886 non-null  float64
#  5   atemp       10886 non-null  float64
#  6   humidity    10886 non-null  int64
#  7   windspeed   10886 non-null  float64
#  8   casual      10886 non-null  int64
#  9   registered  10886 non-null  int64
#  10  count       10886 non-null  int64


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
    x, y, shuffle = True, train_size = 0.7, random_state= 3000
)

print(x_train.shape, x_test.shape)  # (7620, 8) (3266, 8)
print(y_train.shape, y_test.shape)  # (7620,) (3266,)


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 8))
model.add(Dense(7))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(1))
# 한정(활성화) 함수, activation: 모델구성에서 값을 한정시키고 싶을 때 사용 (항상 양수가 됨)


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1500, batch_size = 50, verbose = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)              # RMSE 함수 사용
print("RMSE : ", rmse)

##### submission.csv를 만들어봅시다. #####
print(test_csv.isnull().sum())      # 여기도 결측치가 있음

y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'samplesubmission.csv', index_col = 0)
print(submission)
submission['count'] = y_submit
print(submission)

path_save = './_save/kaggle_bike/'
submission.to_csv(path_save + 'submit_0307_0345.csv')



