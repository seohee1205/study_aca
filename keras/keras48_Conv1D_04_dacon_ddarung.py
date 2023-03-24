from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, Input
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import time

#1. 데이터
path = './_data/ddarung/'
path_save='./_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv',         # = train_csv = pd.read_csv('./data/ddarung/train.csv')
                        index_col = 0)

print(train_csv)        # [1459 rows x 10 columns]
print(train_csv.shape)  # (1459, 10)

test_csv = pd.read_csv(path + 'test.csv',         
                        index_col = 0)

# 결측치
print(train_csv.isnull().sum())         
train_csv = train_csv.dropna() # 결측치 제거
print(train_csv.isnull().sum())  

# x, y 분리
x = train_csv.drop(['count'], axis = 1)     # [ ] 2개 이상 = 리스트, axis = 열
print(x)
y = train_csv['count']
print(y)

# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size = 0.8, random_state = 777
)
# print(x_train.shape, y_train.shape)      # (1062, 9) (1062,)
# print(x_test.shape, y_test.shape)      # (266, 9) (266,)

# scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = x_train.reshape(-1, 9, 1) 
x_test = x_test.reshape(-1, 9, 1)
test_csv = test_csv.reshape(-1, 9, 1)
# -1인 이유: 알아서 최대값으로 들어감

#2. 모델
input1 = Input(shape=(9, 1))
Conv1 = Conv1D(54, 2, activation='relu')(input1)
Conv2 = Conv1D(34, 2, activation='relu')(Conv1)
Flat1 = Flatten()(Conv2)
dense1 = Dense(16,activation='relu')(Flat1)
dense2 = Dense(12,activation='relu')(dense1)
output1 = Dense(1)(dense2)
model = Model(inputs=input1, outputs=output1)

# model.summary()

#3. 컴파일, 훈련
start = time.time()
model.compile(loss = 'mse', optimizer = 'adam')

es = EarlyStopping(monitor = 'loss', patience = 60, mode = 'auto',
                   verbose = 1, restore_best_weights= True)

# mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto',
#         verbose = 1, 
#         save_best_only= True,
#         filepath="".join([filepath, 'k27_', date, '_', filename]))

model.fit(x_train, y_train, epochs = 500, batch_size = 50,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

end= time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)


#5 파일 저장
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'submission.csv', index_col = 0)
submission['count'] = y_submit
path_save = './_save/ddarung/'
submission.to_csv(path_save + 'submit_0323_0550.csv')
print('걸린 시간 : ', np.round(end-start, 2))


# result :  1985.6859130859375
# r2 스코어 :  0.6997628178791967
# 걸린 시간 :  73.27

