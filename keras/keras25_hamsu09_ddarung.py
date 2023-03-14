import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

#1. 데이터
path = './_data/ddarung/'

train_csv = pd.read_csv(path + 'train.csv',         # = train_csv = pd.read_csv('./data/ddarung/train.csv')
                        index_col = 0)              # header와 index는 따로 계산하지 않는다.

print(train_csv)
print(train_csv.shape)      # (1459, 10)

test_csv = pd.read_csv(path + 'test.csv',         
                        index_col = 0)
print(test_csv)
print(test_csv.shape)      # (715, 9)

#===========================================================================================

print(train_csv.columns)
print(train_csv.info())
print(train_csv.describe())
print(type(train_csv))            


############################### 결측치 처리 ##################################
# 결측치 처리 1. 제거
# print(train_csv.isnull())
print(train_csv.isnull().sum())         
train_csv = train_csv.dropna()           ### 결측치 제거 ###
print(train_csv.isnull().sum())         # 결측치 제거 후 확인용
print(train_csv.info())
print(train_csv.shape)

############################### train_csv 데이터에서 x와 y를 분리 ########################################
# x와 y분리 중요!!!!!!

x = train_csv.drop(['count'], axis = 1)     # [ ] 2개 이상 = 리스트, axis = 열
print(x)
                        # train에는 있지만, test에는 없으므로 x에서 count 제외 후 y로 정해줌
y = train_csv['count']
print(y)

############################### train_csv 데이터에서 x와 y를 분리 #########################################

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size = 0.95, random_state = 3748
)
                                                                # 결측치 제거 후
print(x_train.shape, x_test.shape)   # (1021, 9) (438, 9)   ->  (929, 9) (399, 9)
print(y_train.shape, y_test.shape)  # (1021,) (438,)        ->  (929,) (399,)

# 전처리 (정규화)는 데이터를 나눈 후 한다
scaler = MinMaxScaler()   # 하나로 모아줄 때  
# scaler = StandardScaler()   # 표준 정규표를 만들 때
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)     # fit의 범위: x_train
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
print(np.min(x_test), np.max(x_test)) # 0.0 1.0

test_csv = scaler.transform(test_csv)   # test_csv에도 sclaer해줘야 함

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(50, input_dim = 9))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dense(38, activation = 'relu'))
# model.add(Dense(20, activation = 'relu'))
# model.add(Dense(13, activation = 'relu'))
# model.add(Dense(1))


#(함수형) 모델 구성
input1 = Input(shape=(9,)) # 인풋명시, 
dense1 = Dense(50, activation = 'relu')(input1)  # Dense 모델 구성 후, 이 모델은 어디에서 시작해서 어디에서 끝나는지 연결
dense2 = Dense(30, activation = 'relu')(dense1)
dense3 = Dense(38, activation = 'relu')(dense2)
dense4 = Dense(20, activation = 'relu')(dense3)
dense5 = Dense(13, activation = 'relu')(dense4)
output1 = Dense(1,)(dense5)
model = Model(inputs = input1, outputs = output1)


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

# 정의하기

es = EarlyStopping(monitor= 'val_loss', patience = 34, mode= 'min',     # if mode = auto: 자동으로 min 또는 max로 맞춰줌 
              verbose= 1,    
              restore_best_weights= True)  


hist = model.fit(x_train, y_train, epochs = 1200, batch_size = 17,
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
# 파일 생성
path_save = './_save/ddarung/'
submission.to_csv(path_save + 'submit_0310_0725.csv')
'''


'''
# scaler = MinMaxScaler() 

 
# scaler = StandardScaler() 


# scaler = MaxAbsScaler()



# scaler = RobustScaler()


'''



