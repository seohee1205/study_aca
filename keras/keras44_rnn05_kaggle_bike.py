from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, GRU, Input, Dropout, MaxPooling2D
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler 
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import pandas as pd
import time

#1. 데이터
path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv',
                        index_col = 0)   
print(train_csv.shape)      # (10886, 11)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col = 0)

# 결측치
print(train_csv.isnull().sum()) # 결측치 없음

# x, y 분리
x = train_csv.drop(['count', 'casual', 'registered'], axis = 1)
print(x)
y = train_csv['count']
print(y)

# print(type(x))
# train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle = True, train_size = 0.95, random_state= 868
)

print(x_train.shape, y_train.shape)     # (10341, 8) (10341,)
print(x_test.shape, y_test.shape)       # (545, 8) (545,)

# print(type(x_train))

# scaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

x_train = x_train.reshape(-1, 8, 1)
x_test = x_test.reshape(-1, 8, 1)
test_csv = test_csv.reshape(-1, 8, 1)

#2. 모델
input1 = Input(shape=(8, 1))
GRU1 = GRU(54, activation='relu',  return_sequences = True)(input1)
GRU2 = GRU(34, activation='relu',  return_sequences = True)(GRU1)
GRU3 = GRU(24)(GRU2)
dense1 = Dense(16,activation='relu')(GRU3)
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

model.fit(x_train, y_train, epochs = 500, batch_size = 180,
          validation_split = 0.2,
          verbose = 1,
          callbacks = [es])

end = time.time()

#4. 평가, 예측
result = model.evaluate(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 스코어 : ', r2)

#5 파일 저장
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col = 0)
submission['count'] = y_submit
path_save = './_save/kaggle_bike/'
submission.to_csv(path_save + 'submit_0323_1135.csv')
print('걸린 시간 : ', np.round(end-start, 2))


# result :  25726.896484375
# r2 스코어 :  0.3368206401659041
# 걸린 시간 :  8663.87