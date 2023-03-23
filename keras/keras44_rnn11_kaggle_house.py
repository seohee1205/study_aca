from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, GRU, Input, Dropout, MaxPooling2D
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # preprocessing: 전처리 / MinMaxScaler: 정규화 / StandardScaler: 평균점을 중심으로 데이터를 가운데로 모은다
from sklearn.preprocessing import MaxAbsScaler, RobustScaler,  LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import time

#1. 데이터
path = './_data/kaggle_house/'
path_save = './_save/kaggle_house/'

train_csv = pd.read_csv(path + 'train.csv',
                    index_col = 0)

print(train_csv.shape)  # (1121, 80)

test_csv = pd.read_csv(path + 'test.csv',
                       index_col= 0)
print(test_csv.shape)   # (1121, 78)

#1-1. 결측치
print(train_csv.isnull().sum())  

df = pd.read_csv(path + 'train.csv')

#1-2. 라벨인코딩
le=LabelEncoder()
for i in train_csv.columns:
    if train_csv[i].dtype=='object':
        train_csv[i] = le.fit_transform(train_csv[i])
        test_csv[i] = le.fit_transform(test_csv[i])
print(len(train_csv.columns))
print(train_csv.info())
train_csv=train_csv.dropna()
print(train_csv.shape)

#1-3. x, y 데이터 분리
x = train_csv.drop(['SalePrice', 'LotFrontage'], axis = 1)
print(x.shape)
y = train_csv['SalePrice']
print(y.shape)
test_csv = test_csv.drop(['LotFrontage'], axis =1)

#1-4. train, test 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size= 0.9, random_state= 888
)

print(x_train.shape, y_train.shape)     # (1008, 78) (1008,)
print(x_test.shape, y_test.shape)      # (113, 78) (113,)

#1-5. scaler
scaler = MinMaxScaler()                # fit의 범위: x_train
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # x_train의 변화 비율에 맞춰하기 때문에 scaler에 fit을 할 필요가 없음(변환만 해줌)
# print(np.min(x_test), np.max(x_test))   # 0과 1이 나오는지 확인해
test_csv = scaler.transform(test_csv)   # test_csv에도 scaler해줘야 함

x_train = x_train.reshape(-1, 78, 1) 
x_test = x_test.reshape(-1, 78, 1)
test_csv = test_csv.reshape(-1, 78, 1)

#2. 모델
input1 = Input(shape=(78, 1))
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

model.fit(x_train, y_train, epochs = 10, batch_size = 70,
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
submission = pd.read_csv(path + 'sample_submission.csv', index_col = 0)
submission['count'] = y_submit
path_save = './_save/kaggle_house/'
submission.to_csv(path_save + 'submit_0323_1245.csv')
print('걸린 시간 : ', np.round(end-start, 2))

# result :  42690211840.0
# r2 스코어 :  -3.975465500591917
# 걸린 시간 :  433.96