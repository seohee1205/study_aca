# 데이콘 따릉이 문제풀이
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

#1. 데이터
path = './_data/dacon_wine/'


train_csv = pd.read_csv(path + 'train.csv',         # = train_csv = pd.read_csv('./data/ddarung/train.csv')
                        index_col = 0)              # header와 index는 따로 계산하지 않는다.

print(train_csv)
print(train_csv.shape)      # (5497, 13)

test_csv = pd.read_csv(path + 'test.csv',         
                        index_col = 0)
print(test_csv)
print(test_csv.shape)      # (1000, 12)


# 분리형을 수치형으로 바꿔주기
from sklearn.preprocessing import LabelEncoder, RobustScaler          # 데이터 전처리?
le = LabelEncoder()
le.fit(train_csv['type'])   # 0과 1로 인정
aaa = le.transform(train_csv['type'])
print(aaa)
print(type(aaa))    # <class 'numpy.ndarray'>
print(aaa.shape)    # (5497,) 벡터 형태
print(np.unique(aaa, return_counts= True))      # 몇 개씩 있는지    (array([0, 1]), array([1338, 4159], dtype=int64)) => 0이 1338개, 1이 4159개

train_csv['type'] = aaa
print(train_csv)
test_csv['type'] = le.transform(test_csv['type'])

print(le.transform(['red', 'white']))       # [0 1]     => red = 0, white = 1 
# print(le.transform(['white', 'red']))       # [1 0]     => white = 1, red = 0



print(train_csv['quality'].value_counts())
# 6    2416
# 5    1788
# 7     924
# 4     186
# 8     152
# 3      26
# 9       5




'''
#===========================================================================================

print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64

print(train_csv.describe())

print(type(train_csv))             # <class 'pandas.core.frame.DataFrame'>


############################### 결측치 처리 ##################################
# 결측치 처리 1. 제거
# print(train_csv.isnull())
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()           ### 결측치 제거 ###
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)



############################### train_csv 데이터에서 x와 y를 분리 ########################################

x = train_csv.drop(['count'], axis = 1)     # [ ] 2개 이상 = 리스트, axis =0 이면 행, axis = 1이면 열
print(x)
       # train에는 있지만, test에는 없으므로 x에서 count 제외 후 y로 정해줌
y = train_csv['count']
print(y)

############################### train_csv 데이터에서 x와 y를 분리 #########################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size = 0.7, random_state = 777
)
                                                                # 결측치 제거 후
print(x_train.shape, x_test.shape)   # (1021, 9) (438, 9)   ->  (929, 9) (399, 9)
print(y_train.shape, y_test.shape)  # (1021,) (438,)        ->  (929,) (399,)


#2. 모델구성
model = Sequential()
model.add(Dense(12, input_dim = 9))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(13))
model.add(Dense(4))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 10, batch_size = 32,
          verbose = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

'''
