import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd

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
    x, y, shuffle = True, train_size = 0.8, random_state = 650
)
                                                                # 결측치 제거 후
print(x_train.shape, x_test.shape)   # (1021, 9) (438, 9)   ->  (929, 9) (399, 9)
print(y_train.shape, y_test.shape)  # (1021,) (438,)        ->  (929,) (399,)


#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim = 9))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

# 정의하기

es = EarlyStopping(monitor= 'val_loss', patience = 20, mode= 'min',     # if mode = auto: 자동으로 min 또는 max로 맞춰줌 
              verbose= 1,    
              restore_best_weights= True)  


hist = model.fit(x_train, y_train, epochs = 1000, batch_size = 24,
                 validation_split = 0.2, 
                 verbose = 1,
                 callbacks= [es]    # es 호출
                 )


# print("=============================================")
# print(hist)
# # <tensorflow.python.keras.callbacks.History object at 0x00000227C99A01F0>
# print("=============================================")
# print(hist.history)
# print("=============================================")
# print(hist.history['loss'])
# print("======================발로스=======================")
# print(hist.history['val_loss'])
# print("======================발로스=======================")


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'    # 한글 깨짐 방지 / 앞으로 나눔체로 쓰기 

plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')      # 선 긋기 / 순서대로 할 때는 x를 명시하지 않아도 됨.
plt.plot(hist.history['val_loss'], marker = '.', c= 'blue', label = 'val_loss')
plt.title('따릉이')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()    # 선에 이름 표시
plt.grid()      # 격자
plt.show()

# val_loss가 loss보다 높은 위치에 있음

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


# 파일 생성
y_submit = model.predict(test_csv)
print(y_submit)

submission = pd.read_csv(path + 'submission.csv', index_col = 0)
print(submission)
submission['count'] = y_submit
print(submission)

path_save = './_save/ddarung/'
submission.to_csv(path_save + 'submit_0308_0735.csv')



# loss :  3112.440185546875
# r2 스코어 :  0.28345819422862617
# RMSE :  55.78924758144579

