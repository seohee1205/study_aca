import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
import datetime


#1. 데이터
path = 'd:/study_data/_data/dacon/'
path_save = 'd:/study_data/_save/dacon/'

train_csv = pd.read_csv(path + 'train.csv', index_col = 0)

print(train_csv)        # [7500 rows x 10 columns]
print(train_csv.shape)  # (7500, 10)

test_csv = pd.read_csv(path + 'test.csv', index_col = 0)

print(test_csv)         # [7500 rows x 9 columns]
print(test_csv.shape)   # (7500, 9)

#1-2. 결측치 처리
# print(train_csv.isnull().sum())   # 결측치 없음

#1-4. 라벨인코더
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
train_csv['Weight_Status'] = encoder1.fit_transform(train_csv['Weight_Status'])

train_csv['Gender'] = encoder2.fit_transform(train_csv['Gender'])

test_csv['Weight_Status'] = encoder1.fit_transform(test_csv['Weight_Status'])

test_csv['Gender'] = encoder2.fit_transform(test_csv['Gender'])

#1-3. x, y 분리
x = train_csv.drop(['Calories_Burned'], axis= 1)
print(x)
y = train_csv['Calories_Burned']
print(y)

#1-5. x, y 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, train_size= 0.9, random_state= 337
)

# print(x_train.shape, x_test.shape)  # (6000, 9) (1500, 9)
# print(y_train.shape, y_test.shape)  # (6000,) (1500,)

#1-4. 전처리
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

test_csv = scaler.transform(test_csv)

#2. 모델 구성
model = Sequential()
model.add(Dense(256, input_dim = 9))
model.add(Dense(128, activation = 'selu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'selu'))
model.add(Dense(64, activation = 'selu'))
model.add(Dense(80, activation = 'selu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation = 'selu'))
model.add(Dense(64, activation = 'selu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation = 'selu'))
model.add(Dense(16, activation = 'selu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'min',
                   verbose = 1)

model.fit(x_train, y_train, epochs = 1000, batch_size = 24,
          validation_split= 0.2,
          verbose = 1,
          callbacks = [es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)              # RMSE 함수 사용
print("RMSE : ", rmse)

#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

# 파일 생성
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sample_submission.csv')
submission['Calories_Burned'] = y_submit

submission.to_csv(path_save + date + 'sample_submission.csv', index=False)


